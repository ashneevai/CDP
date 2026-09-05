"""Import pinned public healthcare reference releases into indexed local snapshots.

The importer never trusts a mutable URL alone: callers must supply the expected
SHA-256 published/recorded by their release process. Raw source rows are not
printed, and the generated SQLite snapshot is immutable at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import httpx

from packages.reference_data.snapshot import SnapshotManifest
from packages.reference_enrichment.xlsx_reader import read_sheet

OFFICIAL_HOSTS = {
    "cms.gov",
    "www.cms.gov",
    "download.cms.gov",
    "cdc.gov",
    "www.cdc.gov",
    "ftp.cdc.gov",
}
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    source_name: str
    supported_fields: tuple[str, ...]


DATASETS = {
    "nppes": DatasetSpec(
        "AUTHORIZED_PROVIDER",
        "CMS NPPES",
        (
            "npi",
            "provider_npi",
            "billing_provider_npi",
            "rendering_provider_npi",
            "provider_name",
            "billing_provider_name",
            "rendering_provider_name",
        ),
    ),
    "icd10cm": DatasetSpec(
        "ICD10CM",
        "CDC ICD-10-CM",
        ("principal_diagnosis", "diagnosis_code"),
    ),
    "hcpcs-level-ii": DatasetSpec(
        "HCPCS_LEVEL_II",
        "CMS HCPCS Level II",
        ("hcpcs", "cpt_hcpcs"),
    ),
    "place-of-service": DatasetSpec(
        "PLACE_OF_SERVICE",
        "CMS Place of Service",
        ("place_of_service",),
    ),
}


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() == "tr":
            self._row = []
        elif tag.casefold() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag.casefold() == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in OFFICIAL_HOSTS:
        raise ValueError("public reference URL must use HTTPS on an approved CMS/CDC host")
    if parsed.username or parsed.password:
        raise ValueError("credentials are prohibited in public reference URLs")


def download(url: str, destination: Path, expected_sha256: str) -> None:
    current = url
    with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=False) as client:
        for _ in range(5):
            _validate_public_url(current)
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError("reference download redirect omitted Location")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                total = 0
                digest = hashlib.sha256()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError("reference download exceeded size limit")
                        digest.update(chunk)
                        handle.write(chunk)
                if digest.hexdigest().lower() != expected_sha256.lower():
                    destination.unlink(missing_ok=True)
                    raise ValueError("source artifact checksum mismatch")
                return
        raise RuntimeError("too many reference download redirects")


def _header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _value(row: dict[str, str], *names: str) -> str:
    indexed = {_header(key): str(value or "").strip() for key, value in row.items()}
    return next((indexed[_header(name)] for name in names if indexed.get(_header(name))), "")


def _first_sheet(path: Path) -> list[dict[str, str]]:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{main}}}sheet")
    if sheet is None:
        return []
    return read_sheet(path, sheet.attrib["name"])


def _archive_member(archive: ZipFile, dataset: str) -> str:
    candidates = [item for item in archive.infolist() if not item.is_dir()]
    extensions = {
        "nppes": (".csv",),
        "icd10cm": (".txt", ".csv"),
        "hcpcs-level-ii": (".xlsx", ".csv"),
        "place-of-service": (".xlsx", ".csv"),
    }[dataset]
    candidates = [item for item in candidates if item.filename.lower().endswith(extensions)]
    if dataset == "icd10cm":
        order = [item for item in candidates if "order" in item.filename.casefold()]
        candidates = order or candidates
    if dataset == "nppes":
        primary = [item for item in candidates if "npidata_pfile" in item.filename.casefold()]
        candidates = primary or candidates
    if not candidates:
        raise ValueError(f"archive contains no supported {dataset} data file")
    return max(candidates, key=lambda item: item.file_size).filename


def _table_rows(path: Path, dataset: str):
    suffix = path.suffix.casefold()
    if suffix == ".zip":
        with ZipFile(path) as archive:
            member = _archive_member(archive, dataset)
            member_info = archive.getinfo(member)
            if member_info.file_size > MAX_EXTRACTED_BYTES:
                raise ValueError("reference archive member exceeded extracted size limit")
            member_suffix = Path(member).suffix.casefold()
            with tempfile.TemporaryDirectory() as directory:
                extracted = Path(directory) / f"source{member_suffix}"
                total = 0
                with archive.open(member_info) as source, extracted.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > MAX_EXTRACTED_BYTES:
                            raise ValueError(
                                "reference archive member exceeded extracted size limit"
                            )
                        target.write(chunk)
                yield from _table_rows(extracted, dataset)
        return
    if suffix == ".xlsx":
        yield from _first_sheet(path)
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if dataset == "place-of-service" and suffix in {"", ".htm", ".html"}:
        parser = _HtmlTableParser()
        parser.feed(path.read_text("utf-8", errors="replace"))
        for cells in parser.rows:
            if len(cells) >= 3 and re.fullmatch(r"\d{2}", cells[0]):
                yield {
                    "Place of Service Code": cells[0],
                    "Place of Service Name": cells[1],
                    "Description": cells[2],
                }
        return
    if suffix == ".txt" and dataset == "icd10cm":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                parts = line.strip().split(None, 4)
                if len(parts) >= 4 and parts[0].isdigit() and parts[2] in {"0", "1"}:
                    yield {
                        "code": parts[1],
                        "billable": parts[2],
                        "short_description": parts[3],
                        "long_description": parts[4] if len(parts) > 4 else parts[3],
                    }
                elif len(parts) >= 2:
                    yield {
                        "code": parts[0],
                        "billable": "",
                        "long_description": " ".join(parts[1:]),
                    }
        return
    raise ValueError(f"unsupported source format for {dataset}: {suffix}")


def _public_record(dataset: str, row: dict[str, str], artifact_sha256: str) -> dict | None:
    lineage = [f"official-public-release:{artifact_sha256}"]
    if dataset == "nppes":
        npi = _value(row, "NPI")
        if not re.fullmatch(r"\d{10}", npi):
            return None
        organization = _value(row, "Provider Organization Name (Legal Business Name)")
        person = " ".join(
            filter(
                None,
                [
                    _value(row, "Provider First Name"),
                    _value(row, "Provider Middle Name"),
                    _value(row, "Provider Last Name (Legal Name)"),
                ],
            )
        )
        name = organization or person
        deactivated = _value(row, "NPI Deactivation Date")
        reactivated = _value(row, "NPI Reactivation Date")
        status = "ACTIVE" if not deactivated or reactivated else "INACTIVE"
        fields = {field: npi for field in DATASETS[dataset].supported_fields if "npi" in field}
        fields.update(
            {field: name for field in DATASETS[dataset].supported_fields if "name" in field}
        )
        return {
            "identity_key": npi,
            "source_record_id": f"NPI:{npi}",
            "source_lineage": lineage,
            "reference_attributes": {
                "npi": npi,
                "provider_name": name,
                "address": _value(row, "Provider First Line Business Practice Location Address"),
                "city": _value(row, "Provider Business Practice Location Address City Name"),
                "state": _value(row, "Provider Business Practice Location Address State Name"),
                "zip": _value(row, "Provider Business Practice Location Address Postal Code")[:9],
                "taxonomy": _value(row, "Healthcare Provider Taxonomy Code_1"),
                "entity_type": _value(row, "Entity Type Code"),
            },
            "field_values": fields,
            "record_status": status,
        }
    if dataset == "icd10cm":
        raw = _value(row, "code", "ICD-10-CM CODE", "ICD10CMCode").upper().replace(".", "")
        if not re.fullmatch(r"[A-TV-Z][0-9A-Z]{2,6}", raw):
            return None
        display = raw if len(raw) == 3 else f"{raw[:3]}.{raw[3:]}"
        description = _value(row, "long_description", "Long Description", "Description")
        fields = {field: display for field in DATASETS[dataset].supported_fields}
        return {
            "identity_key": raw,
            "source_record_id": f"ICD10CM:{raw}",
            "source_lineage": lineage,
            "reference_attributes": {
                "description": description,
                "billable": _value(row, "billable", "Header"),
            },
            "field_values": fields,
            "record_status": "VALID",
        }
    if dataset == "hcpcs-level-ii":
        code = _value(row, "HCPCS Code", "HCPC", "HCPCS", "Code").upper()
        if not re.fullmatch(r"[A-Z]\d{4}", code):
            return None
        termination = _value(row, "Termination Date", "Term Date")
        fields = {field: code for field in DATASETS[dataset].supported_fields}
        return {
            "identity_key": code,
            "source_record_id": f"HCPCS:{code}",
            "source_lineage": lineage,
            "reference_attributes": {
                "description": _value(row, "Long Description", "Long Description1", "Description"),
                "effective_date": _value(row, "Effective Date", "Add Date"),
                "termination_date": termination,
            },
            "field_values": fields,
            "record_status": "INACTIVE" if termination else "VALID",
        }
    code = _value(row, "Place of Service Code", "POS Code", "Code").zfill(2)
    if not re.fullmatch(r"\d{2}", code):
        return None
    return {
        "identity_key": code,
        "source_record_id": f"POS:{code}",
        "source_lineage": lineage,
        "reference_attributes": {
            "name": _value(row, "Place of Service Name", "Name"),
            "description": _value(row, "Description"),
        },
        "field_values": {"place_of_service": code},
        "record_status": "VALID",
    }


def import_snapshot(
    source: Path,
    destination: Path,
    *,
    dataset: str,
    version: str,
    source_url: str,
    expected_sha256: str,
    expires_at: datetime,
    source_published_at: datetime | None = None,
    effective_from: datetime | None = None,
    effective_through: datetime | None = None,
    source_contract_id: str,
    approved_by: str,
) -> dict:
    if destination.exists():
        raise ValueError("snapshot destination already exists")
    _validate_public_url(source_url)
    actual_sha = sha256_file(source)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError("source artifact checksum mismatch")
    if expires_at.tzinfo is None:
        raise ValueError("expires_at must include a timezone")
    spec = DATASETS[dataset]
    destination.mkdir(parents=True)
    database = destination / "records.sqlite3"
    count = 0
    try:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE reference_records (identity_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )
            for row in _table_rows(source, dataset):
                record = _public_record(dataset, row, actual_sha)
                if record is None:
                    continue
                payload = dict(record)
                payload["response_hash"] = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                connection.execute(
                    "INSERT INTO reference_records(identity_key, payload_json) VALUES (?, ?)",
                    (
                        record["identity_key"],
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    ),
                )
                count += 1
            if not count:
                raise ValueError("source produced no valid reference records")
            connection.execute("PRAGMA optimize")
        records_sha = sha256_file(database)
        now = datetime.now(UTC)
        manifest_payload = {
            "source_name": spec.source_name,
            "reference_domain": spec.domain,
            "version": version,
            "snapshot_timestamp": now.isoformat(),
            "records_file": database.name,
            "storage_format": "SQLITE",
            "record_count": count,
            "records_sha256": records_sha,
            "authorized": True,
            "independent_truth": True,
            "non_circular_lineage": True,
            "source_contract_id": source_contract_id,
            "approved_by": approved_by,
            "approved_at": now.isoformat(),
            "supported_fields": list(spec.supported_fields),
            "source_url": source_url,
            "source_artifact_sha256": actual_sha,
            "source_published_at": source_published_at.isoformat() if source_published_at else None,
            "effective_from": effective_from.isoformat() if effective_from else None,
            "effective_through": effective_through.isoformat() if effective_through else None,
            "expires_at": expires_at.isoformat(),
        }
        manifest = SnapshotManifest.model_validate(manifest_payload)
        (destination / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(database, 0o444)
        return {
            "dataset": dataset,
            "version": version,
            "records": count,
            "records_sha256": records_sha,
            "source_artifact_sha256": actual_sha,
        }
    except Exception:
        database.unlink(missing_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        destination.rmdir()
        raise


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--url")
    parser.add_argument("--source-url")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expires-at", type=_datetime, required=True)
    parser.add_argument("--source-published-at", type=_datetime)
    parser.add_argument("--effective-from", type=_datetime)
    parser.add_argument("--effective-through", type=_datetime)
    parser.add_argument("--source-contract-id", required=True)
    parser.add_argument("--approved-by", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        if args.url:
            local = Path(directory) / Path(urlparse(args.url).path).name
            download(args.url, local, args.expected_sha256)
            provenance_url = args.url
        else:
            local = args.source
            provenance_url = args.source_url
        if not provenance_url:
            parser.error("--source-url is required when importing a local file")
        report = import_snapshot(
            local,
            args.destination,
            dataset=args.dataset,
            version=args.version,
            source_url=provenance_url,
            expected_sha256=args.expected_sha256,
            expires_at=args.expires_at,
            source_published_at=args.source_published_at,
            effective_from=args.effective_from,
            effective_through=args.effective_through,
            source_contract_id=args.source_contract_id,
            approved_by=args.approved_by,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
