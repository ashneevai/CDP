"""Prediction-blind dual-annotation and adjudication validation utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_DOCUMENT_TYPES = {"CMS1500", "UB04", "UNKNOWN_STRUCTURED",
                           "UNKNOWN_UNSTRUCTURED", "NON_CLAIM"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"INVALID_RECORD:line={line_no}")
        rows.append(value)
    return rows


def validate_annotations(*, manifest: Path, annotations: Path) -> dict[str, Any]:
    expected = {row["document_id"]: row for row in _read_jsonl(manifest)}
    records = _read_jsonl(annotations)
    seen: set[str] = set()
    errors: list[str] = []
    disagreements = 0
    for row in records:
        document_id = str(row.get("document_id") or "")
        if document_id in seen:
            errors.append(f"DUPLICATE_DOCUMENT:{document_id}")
            continue
        seen.add(document_id)
        if document_id not in expected:
            errors.append(f"UNKNOWN_DOCUMENT:{document_id}")
            continue
        if row.get("package_id") != expected[document_id].get("package_id"):
            errors.append(f"PACKAGE_ID_MISMATCH:{document_id}")
        if row.get("document_type") not in REQUIRED_DOCUMENT_TYPES:
            errors.append(f"INVALID_DOCUMENT_TYPE:{document_id}")
        fields = row.get("fields")
        if not isinstance(fields, dict):
            errors.append(f"INVALID_FIELDS:{document_id}")
            continue
        for name, field in fields.items():
            if not isinstance(field, dict) or "critical" not in field:
                errors.append(f"INVALID_FIELD:{document_id}:{name}")
                continue
            annotations_by_id = field.get("annotations") or {}
            if field["critical"] and set(annotations_by_id) != {"annotator_a", "annotator_b"}:
                errors.append(f"CRITICAL_FIELD_REQUIRES_DUAL_ANNOTATION:{document_id}:{name}")
                continue
            values = {str(value) for value in annotations_by_id.values()}
            if len(values) > 1:
                disagreements += 1
                if field.get("adjudicated_value") is None or not field.get("adjudicator_id"):
                    errors.append(f"DISAGREEMENT_REQUIRES_ADJUDICATION:{document_id}:{name}")
    missing = sorted(set(expected) - seen)
    errors.extend(f"MISSING_DOCUMENT:{document_id}" for document_id in missing)
    return {
        "valid": not errors, "manifest_documents": len(expected),
        "annotation_documents": len(seen), "coverage_complete": not missing,
        "disagreements": disagreements, "errors": errors,
        "predictions_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    report = validate_annotations(manifest=args.manifest, annotations=args.annotations)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
