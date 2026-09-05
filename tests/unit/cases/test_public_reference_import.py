import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest

from packages.evidence_router import ReferenceSourceState
from packages.reference_data import LocalSnapshotProvider
from packages.reference_enrichment.contracts import ReferenceLookupRequest
from packages.reference_enrichment.evidence_adapter import ReferenceEvidenceService
from scripts.import_public_reference import import_snapshot
from workers.validation.consumer import reference_source_state


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(field: str, identity: str, requested_at: datetime) -> ReferenceLookupRequest:
    return ReferenceLookupRequest(
        request_id="request-1",
        identity_key=identity,
        document_id="document-1",
        page_number=1,
        document_family="CMS1500",
        field_name=field,
        criticality="CRITICAL",
        current_candidate=identity,
        available_claim_attributes={},
        requested_at=requested_at,
        policy_version="test-v1",
    )


@pytest.mark.parametrize(
    ("dataset", "row", "field", "identity", "expected"),
    [
        (
            "icd10cm",
            {"code": "E11.9", "billable": "1", "long_description": "Type 2 diabetes"},
            "diagnosis_code",
            "E11.9",
            "E11.9",
        ),
        (
            "hcpcs-level-ii",
            {"HCPC": "A0428", "Long Description": "Ambulance service", "Termination Date": ""},
            "cpt_hcpcs",
            "a0428",
            "A0428",
        ),
        (
            "place-of-service",
            {
                "Place of Service Code": "2",
                "Place of Service Name": "Telehealth",
                "Description": "Other than home",
            },
            "place_of_service",
            "2",
            "02",
        ),
    ],
)
def test_public_code_import_creates_indexed_field_scoped_snapshot(
    tmp_path: Path,
    dataset: str,
    row: dict[str, str],
    field: str,
    identity: str,
    expected: str,
) -> None:
    source = tmp_path / "source.csv"
    checksum = _write_csv(source, [row])
    destination = tmp_path / "snapshot"
    import_snapshot(
        source,
        destination,
        dataset=dataset,
        version="2026-test",
        source_url="https://www.cms.gov/reference-release.zip",
        expected_sha256=checksum,
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        source_contract_id="public-release-policy-v1",
        approved_by="data-governance",
    )
    provider = LocalSnapshotProvider(destination, test_only=False)
    records = provider.lookup(_request(field, identity, datetime(2026, 10, 1, tzinfo=UTC)))
    assert len(records) == 1
    assert records[0].field_values[field] == expected
    assert records[0].snapshot_checksum == provider.manifest.records_sha256
    assert provider.manifest.storage_format == "SQLITE"
    assert provider.manifest.record_count == 1
    assert (
        provider.lookup(_request("unrelated_field", identity, datetime(2026, 10, 1, tzinfo=UTC)))
        == []
    )
    assert provider.supports(field)
    assert not provider.supports("unrelated_field")
    service = ReferenceEvidenceService([provider])
    assert reference_source_state(service, field) is ReferenceSourceState.AUTHORIZED
    assert reference_source_state(service, "unrelated_field") is ReferenceSourceState.DISABLED


def test_nppes_import_excludes_tin_and_preserves_public_provenance(tmp_path: Path) -> None:
    source = tmp_path / "nppes.csv"
    checksum = _write_csv(
        source,
        [
            {
                "NPI": "1234567893",
                "Entity Type Code": "1",
                "Provider First Name": "ASHISH",
                "Provider Middle Name": "",
                "Provider Last Name (Legal Name)": "SINGH",
                "Provider Organization Name (Legal Business Name)": "",
                "Employer Identification Number (EIN)": "123456789",
                "Provider First Line Business Practice Location Address": "1 MAIN ST",
                "Provider Business Practice Location Address City Name": "BOSTON",
                "Provider Business Practice Location Address State Name": "MA",
                "Provider Business Practice Location Address Postal Code": "02110",
                "Healthcare Provider Taxonomy Code_1": "207Q00000X",
                "NPI Deactivation Date": "",
                "NPI Reactivation Date": "",
            }
        ],
    )
    destination = tmp_path / "nppes"
    report = import_snapshot(
        source,
        destination,
        dataset="nppes",
        version="2026-09",
        source_url="https://download.cms.gov/nppes/NPI_Files.html",
        expected_sha256=checksum,
        expires_at=datetime(2026, 11, 1, tzinfo=UTC),
        source_contract_id="nppes-policy-v1",
        approved_by="data-governance",
    )
    provider = LocalSnapshotProvider(destination, test_only=False)
    record = provider.lookup(
        _request("provider_name", "1234567893", datetime(2026, 10, 1, tzinfo=UTC))
    )[0]
    assert record.field_values["provider_name"] == "ASHISH SINGH"
    assert "tin" not in record.reference_attributes
    assert "ein" not in record.reference_attributes
    assert report["source_artifact_sha256"] == checksum


def test_official_icd_description_zip_and_pos_html_formats_are_supported(tmp_path: Path) -> None:
    icd_zip = tmp_path / "icd.zip"
    with ZipFile(icd_zip, "w") as archive:
        archive.writestr("icd10cm_codes_2026.txt", "E119 Type 2 diabetes mellitus\n")
    icd_checksum = hashlib.sha256(icd_zip.read_bytes()).hexdigest()
    icd_destination = tmp_path / "icd"
    import_snapshot(
        icd_zip,
        icd_destination,
        dataset="icd10cm",
        version="FY2026",
        source_url="https://ftp.cdc.gov/icd10cm-Code-Descriptions-2026.zip",
        expected_sha256=icd_checksum,
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        source_contract_id="public-release-policy-v1",
        approved_by="data-governance",
    )
    icd = LocalSnapshotProvider(icd_destination, test_only=False).lookup(
        _request("diagnosis_code", "E11.9", datetime(2026, 9, 1, tzinfo=UTC))
    )
    assert icd[0].field_values["diagnosis_code"] == "E11.9"

    pos_html = tmp_path / "code-sets"
    pos_html.write_text(
        "<table><tr><th>Code</th><th>Name</th><th>Description</th></tr>"
        "<tr><td>11</td><td>Office</td><td>Professional office</td></tr></table>",
        "utf-8",
    )
    pos_checksum = hashlib.sha256(pos_html.read_bytes()).hexdigest()
    pos_destination = tmp_path / "pos-html"
    import_snapshot(
        pos_html,
        pos_destination,
        dataset="place-of-service",
        version="2026-02-17",
        source_url="https://www.cms.gov/medicare/coding-billing/place-of-service-codes/code-sets",
        expected_sha256=pos_checksum,
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        source_contract_id="public-release-policy-v1",
        approved_by="data-governance",
    )
    pos = LocalSnapshotProvider(pos_destination, test_only=False).lookup(
        _request("place_of_service", "11", datetime(2026, 9, 1, tzinfo=UTC))
    )
    assert pos[0].reference_attributes["name"] == "Office"


def test_expired_or_tampered_public_snapshot_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "pos.csv"
    checksum = _write_csv(source, [{"Code": "11", "Name": "Office", "Description": "Office"}])
    destination = tmp_path / "pos"
    import_snapshot(
        source,
        destination,
        dataset="place-of-service",
        version="2026",
        source_url="https://www.cms.gov/pos.csv",
        expected_sha256=checksum,
        expires_at=datetime(2026, 10, 1, tzinfo=UTC),
        source_contract_id="pos-policy-v1",
        approved_by="data-governance",
    )
    provider = LocalSnapshotProvider(destination, test_only=False)
    with pytest.raises(RuntimeError, match="expired"):
        provider.lookup(_request("place_of_service", "11", datetime(2026, 10, 1, tzinfo=UTC)))
    assert not provider.is_available("place_of_service", datetime(2026, 10, 1, tzinfo=UTC))
    database = destination / "records.sqlite3"
    database.chmod(0o644)
    database.write_bytes(database.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        provider.lookup(_request("place_of_service", "11", datetime(2026, 9, 1, tzinfo=UTC)))
    assert not provider.is_available("place_of_service", datetime(2026, 9, 1, tzinfo=UTC))


def test_import_rejects_unpinned_or_unofficial_source(tmp_path: Path) -> None:
    source = tmp_path / "pos.csv"
    checksum = _write_csv(source, [{"Code": "11", "Name": "Office"}])
    with pytest.raises(ValueError, match="approved CMS/CDC host"):
        import_snapshot(
            source,
            tmp_path / "bad",
            dataset="place-of-service",
            version="v1",
            source_url="https://example.com/pos.csv",
            expected_sha256=checksum,
            expires_at=datetime(2027, 1, 1, tzinfo=UTC),
            source_contract_id="policy",
            approved_by="governance",
        )
    with pytest.raises(ValueError, match="source artifact checksum"):
        import_snapshot(
            source,
            tmp_path / "bad-sha",
            dataset="place-of-service",
            version="v1",
            source_url="https://www.cms.gov/pos.csv",
            expected_sha256="0" * 64,
            expires_at=datetime(2027, 1, 1, tzinfo=UTC),
            source_contract_id="policy",
            approved_by="governance",
        )
