"""Phase 9E private accuracy qualification controls.

The module intentionally separates prediction freezing, blind truth validation,
and scoring.  Private inputs/outputs are caller supplied and must remain outside
Git.  It never reads truth while producing or freezing predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable


DOCUMENT_TYPES = {
    "CMS_1500", "UB_04", "CUSTOM_STRUCTURED", "CLAIM_ATTACHMENT", "NON_CLAIM", "OTHER"
}
TRUTH_STATUSES = {"APPLICABLE", "NOT_APPLICABLE", "ILLEGIBLE", "UNCERTAIN"}
ACCEPTED = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED", "HUMAN_CONFIRMED"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"INVALID_JSONL_RECORD:{path}:{number}")
            rows.append(value)
    return rows


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_selection(path: Path, *, expected_pages: int = 150) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    if len(rows) != expected_pages:
        raise ValueError(f"SELECTION_SIZE_MISMATCH:expected={expected_pages}:actual={len(rows)}")
    ids = [str(row.get("document_id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("MISSING_OR_DUPLICATE_DOCUMENT_ID")
    packages = [str(row.get("package_id") or "") for row in rows]
    if any(not value for value in packages):
        raise ValueError("MISSING_PACKAGE_ID")
    for row in rows:
        source = Path(str(row.get("path") or ""))
        if not source.is_file() or _sha(source) != row.get("sha256"):
            raise ValueError(f"SOURCE_INTEGRITY_FAILURE:{row['document_id']}")
    return rows


def freeze_existing_predictions(
    selection: Path, predictions: Path, runtime_manifest: Path, output: Path, *, code_sha: str,
    expected_pages: int = 150,
) -> dict[str, Any]:
    """Freeze predictions without accepting or inspecting any truth path."""
    selected = validate_selection(selection, expected_pages=expected_pages)
    predicted = _jsonl(predictions)
    expected_ids = [row["document_id"] for row in selected]
    actual_ids = [row.get("document_id") for row in predicted]
    if actual_ids != expected_ids:
        raise ValueError("PREDICTION_ORDER_OR_MEMBERSHIP_MISMATCH")
    if output.exists():
        raise FileExistsError(f"PREDICTION_FREEZE_ALREADY_EXISTS:{output}")
    runtime = json.loads(runtime_manifest.read_text("utf-8"))
    freeze = {
        "schema_version": "1.0",
        "dataset_id": "CDP_ACCURACY_QUALIFICATION_V1",
        "baseline_git_sha": code_sha,
        "pages": len(selected),
        "selection_manifest_sha256": _sha(selection),
        "ordered_document_ids": expected_ids,
        "ordered_document_ids_sha256": _canonical_sha(expected_ids),
        "predictions_sha256": _sha(predictions),
        "runtime_manifest_sha256": _sha(runtime_manifest),
        "runtime_manifest": runtime,
        "configuration_sha256": _canonical_sha(runtime),
        "truth_present": False,
        "scoring_allowed": False,
        "immutable": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, indent=2) + "\n", "utf-8")
    return freeze


def export_ordered_predictions(results: Path, output: Path) -> int:
    """Export successful harness predictions once, preserving measured order."""
    if output.exists():
        raise FileExistsError(f"PREDICTIONS_ALREADY_EXIST:{output}")
    rows = json.loads(results.read_text("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("EMPTY_OR_INVALID_ORDERED_RESULTS")
    predictions = []
    for expected_ordinal, row in enumerate(rows):
        if row.get("ordinal") != expected_ordinal or row.get("error") is not None:
            raise ValueError(f"FAILED_OR_UNORDERED_RESULT:{expected_ordinal}")
        prediction = row.get("prediction")
        if not isinstance(prediction, dict) or prediction.get("document_id") != row.get("document_id"):
            raise ValueError(f"INVALID_PREDICTION:{expected_ordinal}")
        predictions.append(prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in predictions), "utf-8")
    return len(predictions)


def validate_truth(
    selection: Path, truth: Path, critical_fields: set[str], *, expected_pages: int = 150
) -> dict[str, Any]:
    selected = validate_selection(selection, expected_pages=expected_pages)
    selected_ids = {row["document_id"] for row in selected}
    rows = _jsonl(truth)
    by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        document_id = row.get("document_id")
        if document_id in by_id:
            errors.append(f"DUPLICATE_DOCUMENT_ID:{document_id}")
        by_id[document_id] = row
        if row.get("document_type") not in DOCUMENT_TYPES:
            errors.append(f"INVALID_DOCUMENT_TYPE:{document_id}")
        if not row.get("package_id"):
            errors.append(f"MISSING_PACKAGE_ID:{document_id}")
        fields = row.get("fields") or {}
        for name, field in fields.items():
            if field.get("status") not in TRUTH_STATUSES:
                errors.append(f"INVALID_FIELD_STATUS:{document_id}:{name}")
            if name in critical_fields and field.get("status") == "APPLICABLE":
                a, b = field.get("annotator_a"), field.get("annotator_b")
                if not a or not b or a.get("annotator_id") == b.get("annotator_id"):
                    errors.append(f"CRITICAL_DUAL_ANNOTATION_MISSING:{document_id}:{name}")
                elif a.get("value") != b.get("value") and not field.get("adjudication"):
                    errors.append(f"CRITICAL_ADJUDICATION_MISSING:{document_id}:{name}")
                if field.get("final_value") is None:
                    errors.append(f"CRITICAL_FINAL_VALUE_MISSING:{document_id}:{name}")
    errors.extend(f"MISSING_DOCUMENT:{value}" for value in sorted(selected_ids - set(by_id)))
    errors.extend(f"UNKNOWN_DOCUMENT:{value}" for value in sorted(set(by_id) - selected_ids))
    report = {
        "complete": not errors and len(by_id) == expected_pages,
        "documents": len(by_id),
        "expected_documents": expected_pages,
        "errors": errors,
        "truth_sha256": _sha(truth) if not errors else None,
    }
    return report


def normalize(value: Any) -> str:
    return re.sub(r"[^A-Z0-9.]", "", str(value or "").upper())


def score(
    selection: Path, predictions: Path, freeze: Path, truth: Path, critical_fields: set[str], *,
    expected_pages: int = 150,
) -> dict[str, Any]:
    quality = validate_truth(selection, truth, critical_fields, expected_pages=expected_pages)
    if not quality["complete"]:
        raise ValueError("GROUND_TRUTH_QUALITY_GATE_FAILED:" + ";".join(quality["errors"][:10]))
    frozen = json.loads(freeze.read_text("utf-8"))
    if _sha(predictions) != frozen.get("predictions_sha256"):
        raise ValueError("FROZEN_PREDICTIONS_TAMPERED")
    predicted = {row["document_id"]: row for row in _jsonl(predictions)}
    truths = {row["document_id"]: row for row in _jsonl(truth)}
    selected = validate_selection(selection, expected_pages=expected_pages)
    route_correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    fields: list[dict[str, Any]] = []
    package_ok: dict[str, bool] = defaultdict(lambda: True)
    package_review: dict[str, bool] = defaultdict(bool)
    for item in selected:
        doc_id, package_id = item["document_id"], item["package_id"]
        pred, actual = predicted[doc_id], truths[doc_id]
        pred_type = {"CMS1500": "CMS_1500", "UB04": "UB_04"}.get(pred.get("route"), pred.get("route"))
        truth_type = actual["document_type"]
        is_route_correct = pred_type == truth_type
        route_correct += int(is_route_correct)
        confusion[(truth_type, str(pred_type))] += 1
        package_ok[package_id] &= is_route_correct and bool(actual.get("package_assembly_correct", True))
        pred_fields = pred.get("fields") or {}
        for name, target in (actual.get("fields") or {}).items():
            if target.get("status") != "APPLICABLE":
                continue
            candidate = pred_fields.get(name) or {}
            decision = candidate.get("decision") or {}
            value = candidate.get("value")
            exact = str(value or "") == str(target.get("final_value") or "")
            normalized = normalize(value) == normalize(target.get("final_value"))
            accepted = decision.get("disposition") in ACCEPTED
            review = not accepted
            critical = name in critical_fields
            package_ok[package_id] &= normalized
            package_review[package_id] |= review
            fields.append({"document_id": doc_id, "package_id": package_id, "family": truth_type,
                           "field": name, "critical": critical, "exact": exact,
                           "normalized": normalized, "accepted": accepted, "review": review,
                           "false_accept": accepted and not normalized})
    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None
    critical = [row for row in fields if row["critical"]]
    accepted = [row for row in fields if row["accepted"]]
    critical_accepted = [row for row in critical if row["accepted"]]
    packages = set(package_ok)
    report = {
        "dataset_id": "CDP_ACCURACY_QUALIFICATION_V1",
        "truth_sha256": quality["truth_sha256"],
        "routing_accuracy": ratio(route_correct, len(selected)),
        "routing_confusion_matrix": {f"{a}->{b}": count for (a, b), count in sorted(confusion.items())},
        "overall_exact_match": ratio(sum(r["exact"] for r in fields), len(fields)),
        "normalized_field_accuracy": ratio(sum(r["normalized"] for r in fields), len(fields)),
        "critical_field_accuracy": ratio(sum(r["normalized"] for r in critical), len(critical)),
        "accepted_precision": ratio(sum(r["normalized"] for r in accepted), len(accepted)),
        "critical_accepted_precision": ratio(sum(r["normalized"] for r in critical_accepted), len(critical_accepted)),
        "false_accepts": sum(r["false_accept"] for r in fields),
        "critical_false_accepts": sum(r["false_accept"] for r in critical),
        "field_hitl": ratio(sum(r["review"] for r in fields), len(fields)),
        "critical_field_hitl": ratio(sum(r["review"] for r in critical), len(critical)),
        "claim_hitl": ratio(sum(package_review[p] for p in packages), len(packages)),
        "true_claim_stp": ratio(sum(package_ok[p] and not package_review[p] for p in packages), len(packages)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-truth")
    validate.add_argument("--selection", type=Path, required=True)
    validate.add_argument("--truth", type=Path, required=True)
    validate.add_argument("--critical-field", action="append", default=[])
    export = sub.add_parser("export-predictions")
    export.add_argument("--results", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument("--runtime-manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    if args.command == "validate-truth":
        print(json.dumps(validate_truth(args.selection, args.truth, set(args.critical_field)), indent=2))
    elif args.command == "export-predictions":
        print(json.dumps({"predictions": export_ordered_predictions(args.results, args.output)}))
    elif args.command == "freeze":
        print(json.dumps(freeze_existing_predictions(
            args.selection, args.predictions, args.runtime_manifest, args.output,
            code_sha=args.code_sha,
        ), indent=2))


if __name__ == "__main__":
    main()
