"""Score frozen external-corpus predictions against independently created truth.

The scorer fails closed unless prediction/corpus/runtime identities match the
pre-truth freeze. Raw truth and predictions may contain PHI and must remain
outside Git. Only the returned aggregate report is Git-safe.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from evaluation.truth_contract import iter_truth, truth_fingerprint
from packages.promotion_gates import PromotionGate, PromotionMetrics


_ACCEPTED = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED", "HUMAN_CONFIRMED"}


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return re.sub(r"\s+", " ", text)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"INVALID_PREDICTION_JSONL:line={line_no}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"INVALID_PREDICTION_RECORD:line={line_no}")
            rows.append(row)
    return rows


def _decision_disposition(field: dict[str, Any]) -> str | None:
    decision = field.get("decision") or {}
    return decision.get("disposition") if isinstance(decision, dict) else None


def _prediction_value(field: dict[str, Any]) -> Any:
    for key in ("normalized_value", "value", "candidate", "text"):
        if key in field:
            return field[key]
    decision = field.get("decision") or {}
    if isinstance(decision, dict):
        for key in ("normalized_value", "value"):
            if key in decision:
                return decision[key]
    return None


def _assert_freeze_identity(
    freeze: dict[str, Any],
    predictions_path: Path,
    runtime_manifest_id: str | None,
    corpus_sha256: str | None,
) -> None:
    actual_prediction_sha = _sha_file(predictions_path)
    if actual_prediction_sha != freeze.get("prediction_sha256"):
        raise ValueError("PREDICTION_HASH_MISMATCH")
    if (
        runtime_manifest_id
        and runtime_manifest_id != freeze.get("runtime_manifest_id")
    ):
        raise ValueError("RUNTIME_MANIFEST_MISMATCH")
    if corpus_sha256 and corpus_sha256 != freeze.get("corpus_sha256"):
        raise ValueError("CORPUS_HASH_MISMATCH")
    if freeze.get("truth_present"):
        raise ValueError("INVALID_FREEZE_TRUTH_ALREADY_PRESENT")


def score(
    *,
    predictions_jsonl: str | Path,
    prediction_freeze_json: str | Path,
    truth_jsonl: str | Path,
    output_json: str | Path,
    runtime_manifest_id: str | None = None,
    corpus_sha256: str | None = None,
    fully_loaded_cost_usd: float | None = None,
) -> dict[str, Any]:
    predictions_path = Path(predictions_jsonl)
    freeze = json.loads(
        Path(prediction_freeze_json).read_text(encoding="utf-8")
    )
    _assert_freeze_identity(
        freeze,
        predictions_path,
        runtime_manifest_id,
        corpus_sha256,
    )

    predictions = _read_jsonl(predictions_path)
    prediction_by_id = {
        str(row.get("document_id")): row for row in predictions
    }
    if len(prediction_by_id) != len(predictions):
        raise ValueError("DUPLICATE_PREDICTION_DOCUMENT")

    truths = list(iter_truth(truth_jsonl))
    truth_by_id = {record.document_id: record for record in truths}
    if set(prediction_by_id) != set(truth_by_id):
        missing_predictions = len(set(truth_by_id) - set(prediction_by_id))
        missing_truth = len(set(prediction_by_id) - set(truth_by_id))
        raise ValueError(
            "PREDICTION_TRUTH_COVERAGE_MISMATCH:"
            f"missing_predictions={missing_predictions}:"
            f"missing_truth={missing_truth}"
        )

    total_fields = 0
    correct_fields = 0
    critical_total = 0
    critical_correct = 0
    accepted_total = 0
    accepted_correct = 0
    critical_accepted = 0
    critical_accepted_correct = 0
    critical_false_accepts = 0
    reviewed_fields = 0
    field_failure_reasons: Counter[str] = Counter()
    routing_total = 0
    routing_correct = 0
    group_fields: dict[str, Counter[str]] = defaultdict(Counter)
    package_review: dict[str, bool] = defaultdict(lambda: False)
    wall_seconds: list[float] = []
    cloud_cost = 0.0

    for document_id, truth in truth_by_id.items():
        prediction = prediction_by_id[document_id]
        group = str(prediction.get("group") or "UNKNOWN")
        if truth.document_type:
            routing_total += 1
            predicted_type = str(
                prediction.get("schema") or prediction.get("route") or ""
            )
            if _normalize(predicted_type) == _normalize(truth.document_type):
                routing_correct += 1

        predicted_fields = prediction.get("fields") or {}
        if not isinstance(predicted_fields, dict):
            predicted_fields = {}

        for field_name, truth_field in truth.fields.items():
            total_fields += 1
            group_fields[group]["total"] += 1
            predicted_field = predicted_fields.get(field_name)
            if not isinstance(predicted_field, dict):
                predicted_field = {}
            predicted_value = _prediction_value(predicted_field)
            correct = _normalize(predicted_value) == _normalize(truth_field.value)
            disposition = _decision_disposition(predicted_field)
            accepted = disposition in _ACCEPTED

            if correct:
                correct_fields += 1
                group_fields[group]["correct"] += 1
            elif predicted_value in (None, ""):
                field_failure_reasons["MISSING_EXTRACTION"] += 1
            else:
                field_failure_reasons["WRONG_VALUE"] += 1

            if truth_field.critical:
                critical_total += 1
                if correct:
                    critical_correct += 1

            if accepted:
                accepted_total += 1
                if correct:
                    accepted_correct += 1
                else:
                    field_failure_reasons["FALSE_ACCEPT"] += 1
                if truth_field.critical:
                    critical_accepted += 1
                    if correct:
                        critical_accepted_correct += 1
                    else:
                        critical_false_accepts += 1
            else:
                reviewed_fields += 1
                package_review[truth.package_id] = True
                if correct:
                    field_failure_reasons["CORRECT_BUT_REVIEWED"] += 1

        wall = prediction.get("wall_seconds")
        if isinstance(wall, (int, float)):
            wall_seconds.append(float(wall))
        cloud_cost += float(prediction.get("cloud_cost_usd") or 0.0)

    packages = {truth.package_id for truth in truths}
    claim_hitl_count = sum(
        1 for package_id in packages if package_review[package_id]
    )
    claim_stp_count = len(packages) - claim_hitl_count
    page_count = len(predictions)
    total_cost = (
        fully_loaded_cost_usd
        if fully_loaded_cost_usd is not None
        else cloud_cost
    )

    overall_accuracy = correct_fields / total_fields if total_fields else 0.0
    critical_accuracy = critical_correct / critical_total if critical_total else 0.0
    accepted_precision = accepted_correct / accepted_total if accepted_total else 0.0
    critical_precision = (
        critical_accepted_correct / critical_accepted if critical_accepted else 0.0
    )
    field_hitl_rate = reviewed_fields / total_fields if total_fields else 0.0
    claim_hitl_rate = claim_hitl_count / len(packages) if packages else 0.0
    claim_stp_rate = claim_stp_count / len(packages) if packages else 0.0
    routing_accuracy = routing_correct / routing_total if routing_total else None
    p95 = _percentile(wall_seconds, 0.95) or 0.0
    cost_per_page = total_cost / page_count if page_count else 0.0

    promotion_metrics = PromotionMetrics(
        accepted_precision=accepted_precision,
        critical_field_precision=critical_precision,
        critical_false_accepts=critical_false_accepts,
        field_hitl_rate=field_hitl_rate,
        claim_hitl_rate=claim_hitl_rate,
        claim_stp_rate=claim_stp_rate,
        p95_seconds_per_page=p95,
        cost_usd_per_page=cost_per_page,
        sample_size=page_count,
        overall_field_accuracy=overall_accuracy,
        critical_field_accuracy=critical_accuracy,
        routing_accuracy=routing_accuracy,
    )
    promotion = PromotionGate().evaluate(promotion_metrics)

    group_report = {}
    for group, counts in sorted(group_fields.items()):
        total = counts["total"]
        group_report[group] = {
            "fields": total,
            "field_accuracy": counts["correct"] / total if total else None,
        }

    report = {
        "qualification": {
            "truth_blind": False,
            "accuracy_scored": True,
            "prediction_frozen_before_truth": True,
            "raw_truth_git_safe": False,
            "raw_predictions_git_safe": False,
            "aggregate_report_git_safe": True,
        },
        "identity": {
            "corpus_id": freeze.get("corpus_id"),
            "corpus_sha256": freeze.get("corpus_sha256"),
            "runtime_manifest_id": freeze.get("runtime_manifest_id"),
            "prediction_sha256": freeze.get("prediction_sha256"),
            **truth_fingerprint(truth_jsonl),
        },
        "volume": {
            "pages": page_count,
            "packages": len(packages),
            "truth_fields": total_fields,
            "critical_fields": critical_total,
        },
        "quality": {
            "overall_field_accuracy": overall_accuracy,
            "critical_field_accuracy": critical_accuracy,
            "accepted_precision": accepted_precision,
            "critical_accepted_precision": critical_precision,
            "critical_false_accepts": critical_false_accepts,
            "routing_accuracy": routing_accuracy,
        },
        "automation": {
            "field_hitl_rate": field_hitl_rate,
            "claim_hitl_rate": claim_hitl_rate,
            "claim_stp_rate": claim_stp_rate,
            "claim_stp_count": claim_stp_count,
        },
        "latency_seconds": {
            "mean": statistics.fmean(wall_seconds) if wall_seconds else None,
            "p50": _percentile(wall_seconds, 0.50),
            "p95": _percentile(wall_seconds, 0.95),
            "p99": _percentile(wall_seconds, 0.99),
        },
        "cost": {
            "cloud_cost_usd": cloud_cost,
            "fully_loaded_cost_usd": total_cost,
            "cost_usd_per_page": cost_per_page,
        },
        "failure_taxonomy": dict(sorted(field_failure_reasons.items())),
        "per_group": group_report,
        "promotion": {
            "promote": promotion.promote,
            "reasons": list(promotion.reasons),
        },
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
