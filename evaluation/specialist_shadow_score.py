"""Compare canonical CDP fields with specialist shadow proposals against truth.

This scorer never mutates canonical decisions. It measures whether specialist
proposals improve extraction enough to justify a later controlled activation.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from packages.specialist_activation import (
    SpecialistActivationGate,
    SpecialistMetrics,
)


def _norm(value: Any) -> str:
    return re.sub(r"[^A-Z0-9.]", "", str(value or "").upper())


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_shadow(*, predictions_jsonl: str | Path, truth_jsonl: str | Path) -> dict[str, Any]:
    preds = {str(r["document_id"]): r for r in _load_jsonl(predictions_jsonl)}
    truths = {str(r["document_id"]): r for r in _load_jsonl(truth_jsonl)}
    if set(preds) != set(truths):
        raise ValueError("PREDICTION_TRUTH_COVERAGE_MISMATCH")

    counts = Counter()
    per_field: dict[str, Counter[str]] = {}
    for document_id, truth in truths.items():
        prediction = preds[document_id]
        canonical_fields = prediction.get("fields") or {}
        shadow = prediction.get("specialist_shadow") or {}
        proposed = shadow.get("proposals") or {}
        truth_fields = truth.get("fields") or {}
        for field_name, truth_payload in truth_fields.items():
            truth_value = truth_payload.get("value") if isinstance(truth_payload, dict) else truth_payload
            canonical = canonical_fields.get(field_name) or {}
            canonical_value = canonical.get("normalized_value", canonical.get("value"))
            specialist = proposed.get(field_name) or {}
            specialist_value = specialist.get("value", canonical_value)
            baseline_ok = _norm(canonical_value) == _norm(truth_value)
            specialist_ok = _norm(specialist_value) == _norm(truth_value)
            counts["total"] += 1
            counts["baseline_correct"] += int(baseline_ok)
            counts["specialist_correct"] += int(specialist_ok)
            counts["improved"] += int((not baseline_ok) and specialist_ok)
            counts["regressed"] += int(baseline_ok and (not specialist_ok))
            bucket = per_field.setdefault(field_name, Counter())
            bucket["total"] += 1
            bucket["baseline_correct"] += int(baseline_ok)
            bucket["specialist_correct"] += int(specialist_ok)

    total = counts["total"]
    report = {
        "fields": total,
        "baseline_accuracy": counts["baseline_correct"] / total if total else 0.0,
        "specialist_accuracy": counts["specialist_correct"] / total if total else 0.0,
        "improved_fields": counts["improved"],
        "regressed_fields": counts["regressed"],
        "per_field": {
            name: {
                "total": c["total"],
                "baseline_accuracy": c["baseline_correct"] / c["total"] if c["total"] else None,
                "specialist_accuracy": c["specialist_correct"] / c["total"] if c["total"] else None,
            }
            for name, c in sorted(per_field.items())
        },
    }
    return report


def evaluate_activation(*, shadow_report: dict[str, Any], safety_report: dict[str, Any]) -> dict[str, Any]:
    metrics = SpecialistMetrics(
        sample_size=int(safety_report["volume"]["pages"]),
        baseline_accuracy=float(shadow_report["baseline_accuracy"]),
        specialist_accuracy=float(shadow_report["specialist_accuracy"]),
        accepted_precision=float(safety_report["quality"]["accepted_precision"]),
        critical_accepted_precision=float(safety_report["quality"]["critical_accepted_precision"]),
        critical_false_accepts=int(safety_report["quality"]["critical_false_accepts"]),
        field_hitl_rate=float(safety_report["automation"]["field_hitl_rate"]),
        p95_seconds_per_page=float(safety_report["latency_seconds"]["p95"] or 0.0),
        cost_usd_per_page=float(safety_report["cost"]["cost_usd_per_page"]),
    )
    decision = SpecialistActivationGate().evaluate(metrics)
    return {"activate": decision.activate, "reasons": list(decision.reasons)}
