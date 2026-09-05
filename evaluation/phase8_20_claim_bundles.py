"""Phase 8.20 claim-bundle diagnosis on the frozen governed baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.claim_bundle_analysis import build_claim_blocker_analysis, failure_category
from evaluation.phase8_11_competition_frontier import BASELINE, _field_rows, _read, _rows

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def run(output: Path) -> dict:
    baseline = _read(BASELINE / "summary.json")
    fields = _field_rows()
    claims = []
    field_scope = {row["document_id"]: row for row in fields}
    for source in ("SOURCE_A", "SOURCE_B", "SOURCE_C"):
        for claim in _rows(BASELINE / source.lower() / "claim_decisions.jsonl"):
            scope = field_scope[claim["claim_id"]]
            claims.append({**claim, "source": source, "family": scope["family"]})
    analysis = build_claim_blocker_analysis(claims, fields)
    taxonomy = Counter(failure_category(row) for row in fields if not row["exact"])
    segments = {}
    for source in ("SOURCE_A", "SOURCE_B", "SOURCE_C"):
        scoped_claims = [row for row in analysis["claim_blocker_matrix"] if row["source"] == source]
        scoped_fields = [row for row in fields if row["source"] == source]
        segments[source] = {
            "quality_segment": "UNKNOWN_NOT_CAPTURED",
            "claims": len(scoped_claims),
            "claim_hitl": sum(row["blocker_count"] > 0 for row in scoped_claims) / max(1, len(scoped_claims)),
            "claim_stp": sum(row["blocker_count"] == 0 for row in scoped_claims) / max(1, len(scoped_claims)),
            "raw_accuracy": sum(row["exact"] for row in scoped_fields) / max(1, len(scoped_fields)),
        }
    scorecard = {
        "authority": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "before": {
            "raw_accuracy": baseline["accuracy"]["overall"],
            "critical_raw_accuracy": baseline["accuracy"]["critical"],
            "field_hitl": baseline["safety_and_automation"]["field_hitl"],
            "claim_hitl": baseline["safety_and_automation"]["claim_hitl"],
            "claim_stp": baseline["safety_and_automation"]["claim_stp"],
            "accepted_precision": baseline["safety_and_automation"]["accepted_precision"],
            "critical_false_accepts": baseline["safety_and_automation"]["critical_false_accepts"],
            "wrong_crop_recall": baseline["wrong_crop"]["recall"],
            "p95_latency_seconds": 9.221308400010457,
            "cost_per_page_usd": baseline["cost"]["fully_loaded_cost_per_page_usd"],
        },
        "after": None,
        "actual_complete_claims_unlocked": 0,
        "thresholds_changed": False,
        "production_decision": "NEEDS_MORE_DATA",
        "reason": "No bundle candidate has qualified acceptance authority or real-source shadow evidence.",
    }
    _write(output / "claim_blocker_matrix.json", analysis["claim_blocker_matrix"])
    _write(output / "bundle_pareto.json", {
        "blocker_count_distribution": analysis["blocker_count_distribution"],
        "top_blocker_combinations": analysis["top_blocker_combinations"],
        "correct_but_reviewed_combinations": analysis["correct_but_reviewed_combinations"],
        "incorrect_extraction_combinations": analysis["incorrect_extraction_combinations"],
        "bundle_pareto": analysis["bundle_pareto"],
    })
    _write(output / "before_after_scorecard.json", scorecard)
    _write(output / "segment_scorecard.json", segments)
    _write(output / "claims_unlocked.json", {
        "actual_complete_claims_unlocked": 0,
        "opportunities": analysis["bundle_pareto"],
        "production_authority": False,
    })
    _write(output / "accuracy_error_taxonomy.json", dict(sorted(taxonomy.items())))
    _write(output / "latency_cost_comparison.json", {
        "before": {"p95_seconds": 9.221308400010457, "cost_per_page_usd": scorecard["before"]["cost_per_page_usd"]},
        "after": None,
        "measurement_status": "NO_QUALIFIED_RUNTIME_CANDIDATE",
    })
    return {"scorecard": scorecard, "analysis": analysis, "segments": segments, "taxonomy": dict(taxonomy)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results/phase8_20")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
