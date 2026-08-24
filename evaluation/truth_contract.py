"""Independent truth contract for frozen external-corpus promotion scoring.

Truth values may contain PHI and must remain outside Git. The contract validates
identity and shape only; safe aggregate metrics are produced by the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TruthField:
    value: str | None
    critical: bool = False


@dataclass(frozen=True, slots=True)
class TruthRecord:
    document_id: str
    package_id: str
    document_type: str | None
    fields: dict[str, TruthField]


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_truth(path: str | Path) -> Iterator[TruthRecord]:
    truth_path = Path(path)
    seen: set[str] = set()
    with truth_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"INVALID_TRUTH_JSONL:line={line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"INVALID_TRUTH_RECORD:line={line_no}")
            document_id = str(raw.get("document_id") or "")
            package_id = str(raw.get("package_id") or "")
            if not document_id or not package_id:
                raise ValueError(f"TRUTH_IDENTITY_MISSING:line={line_no}")
            if document_id in seen:
                raise ValueError(f"DUPLICATE_TRUTH_DOCUMENT:{document_id}")
            seen.add(document_id)
            raw_fields = raw.get("fields") or {}
            if not isinstance(raw_fields, dict):
                raise ValueError(f"INVALID_TRUTH_FIELDS:line={line_no}")
            fields: dict[str, TruthField] = {}
            for name, field in raw_fields.items():
                if isinstance(field, dict):
                    value = field.get("value")
                    critical = bool(field.get("critical", False))
                else:
                    value = field
                    critical = False
                fields[str(name)] = TruthField(
                    value=None if value is None else str(value),
                    critical=critical,
                )
            yield TruthRecord(
                document_id=document_id,
                package_id=package_id,
                document_type=(str(raw["document_type"]) if raw.get("document_type") else None),
                fields=fields,
            )


def truth_fingerprint(path: str | Path) -> dict[str, object]:
    truth_path = Path(path)
    records = list(iter_truth(truth_path))
    return {
        "truth_sha256": _sha_file(truth_path),
        "truth_records": len(records),
        "truth_documents": len({record.document_id for record in records}),
        "truth_packages": len({record.package_id for record in records}),
    }
