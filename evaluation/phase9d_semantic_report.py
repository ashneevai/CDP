"""Create private field-level diffs and a PHI-safe aggregate report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text("utf-8"))
    return {
        row["document_id"]: row["prediction"]
        for row in rows
        if row.get("prediction") is not None
    }


def _norm(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _disposition(field: dict[str, Any]) -> str | None:
    return (field.get("decision") or {}).get("disposition")


def _root_cause(name: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    if not after:
        return "FIELD_DEFINITION_CHANGE"
    if not before:
        return "PRIMARY_OCR_MISS"
    if _norm(before.get("value")) == _norm(after.get("value")):
        return "EXTRACTION_METHOD_ONLY"
    compact_name = _norm(name)
    if compact_name and compact_name in _norm(after.get("raw")):
        return "LABEL_INCLUDED_AS_VALUE"
    if before.get("bbox") != after.get("bbox"):
        return "FIELD_SPAN_SELECTION"
    return "NORMALIZATION_CHANGE"


def _classification(before: dict[str, Any], after: dict[str, Any]) -> str:
    if _norm(before.get("value")) == _norm(after.get("value")):
        return "PROVENANCE_ONLY"
    # Phase 9A is only a regression reference. Without independent truth,
    # changed values and missing/populated transitions remain ambiguous.
    return "AMBIGUOUS"


def build(
    baseline_path: Path,
    candidate_path: Path,
    private_output: Path,
    safe_output: Path,
) -> dict[str, Any]:
    baseline = _predictions(baseline_path)
    candidate = _predictions(candidate_path)
    details: list[dict[str, Any]] = []
    for document_id in sorted(set(baseline) | set(candidate)):
        left = baseline.get(document_id, {})
        right = candidate.get(document_id, {})
        family = left.get("schema") or right.get("schema")
        if family not in {"CMS1500", "UB04"}:
            continue
        left_fields = left.get("fields") or {}
        right_fields = right.get("fields") or {}
        diagnostics = right.get("ocr_diagnostics") or {}
        for name in sorted(set(left_fields) | set(right_fields)):
            before = left_fields.get(name, {})
            after = right_fields.get(name, {})
            if before == after:
                continue
            cause = _root_cause(name, before, after)
            classification = _classification(before, after)
            details.append({
                "document_id": document_id,
                "document_family": family,
                "field_name": name,
                "phase9a": {
                    "raw_value": before.get("raw"),
                    "normalized_value": before.get("value"),
                    "confidence": before.get("confidence"),
                    "bbox": before.get("bbox"),
                    "evidence_source": before.get("extraction_method"),
                    "disposition": _disposition(before),
                },
                "candidate": {
                    "raw_value": after.get("raw"),
                    "normalized_value": after.get("value"),
                    "confidence": after.get("confidence"),
                    "bbox": after.get("bbox"),
                    "evidence_source": after.get("extraction_method"),
                    "disposition": _disposition(after),
                },
                "roi": after.get("bbox"),
                "observation_tokens_inside_roi": None,
                "overlapping_tokens": diagnostics.get("tokens_overlapping_multiple_fields"),
                "regional_ocr_triggered": bool(
                    after.get("extraction_method") == "REGIONAL_RAPIDOCR"
                ),
                "regional_trigger_reason": None,
                "classification": classification,
                "root_cause": cause,
            })
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in details),
        "utf-8",
    )
    by_cause = Counter(row["root_cause"] for row in details)
    by_field = Counter(row["field_name"] for row in details)
    by_family = Counter(row["document_family"] or "UNKNOWN" for row in details)
    by_class = Counter(row["classification"] for row in details)
    potential_regressions = sum(
        bool(_norm(row["phase9a"].get("normalized_value")))
        and _norm(row["phase9a"].get("normalized_value"))
        != _norm(row["candidate"].get("normalized_value"))
        for row in details
    )
    potential_causes = Counter(
        row["root_cause"]
        for row in details
        if bool(_norm(row["phase9a"].get("normalized_value")))
        and _norm(row["phase9a"].get("normalized_value"))
        != _norm(row["candidate"].get("normalized_value"))
    )
    # Criticality cannot be truthfully reconstructed from historical result
    # rows; report the limitation instead of guessing.
    impact = defaultdict(int)
    for row in details:
        impact[row["root_cause"]] += 1
    safe = {
        "schema_version": "1.0",
        "changed_fields": len(details),
        "potential_regressions": potential_regressions,
        "classification": dict(sorted(by_class.items())),
        "root_causes": dict(by_cause.most_common()),
        "potential_regression_root_causes": dict(potential_causes.most_common()),
        "fields": dict(by_field.most_common()),
        "families": dict(by_family.most_common()),
        "criticality_breakdown": "UNAVAILABLE_IN_HISTORICAL_RESULTS",
        "business_impact_ranking": [
            {"root_cause": name, "changed_fields": count}
            for name, count in sorted(impact.items(), key=lambda item: (-item[1], item[0]))
        ],
        "independent_truth_used": False,
        "contains_phi_values": False,
        "private_detail_committed": False,
    }
    safe_output.parent.mkdir(parents=True, exist_ok=True)
    safe_output.write_text(json.dumps(safe, indent=2) + "\n", "utf-8")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.baseline,
        args.candidate,
        args.private_output,
        args.safe_output,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
