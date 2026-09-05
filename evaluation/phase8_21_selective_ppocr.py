"""Truth-preserving Phase 8.21 comparative report.

The report refuses to manufacture an after score when fresh challenger
evidence is not joinable to the frozen Source-B replay population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "evaluation_results/phase8_20_rerun/before_after_scorecard.json"
DEFAULT_SHADOW = ROOT / "evaluation_results/ocr_shadow_bakeoff/evaluation_fresh/metrics.json"
DEFAULT_OUTPUT = ROOT / "evaluation_results/phase8_21/comparative_report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def run(*, baseline_path: Path, shadow_path: Path, output_path: Path) -> dict:
    baseline = _load(baseline_path)
    shadow = _load(shadow_path)
    before = baseline["before"]
    incremental = int(shadow.get("incremental_correct_candidates", 0))
    comparable_source_b = bool(shadow.get("source_b_frozen_population", False))
    after = dict(before) if comparable_source_b and incremental == 0 else None
    gates = {
        "critical_false_accepts_zero": shadow.get("critical_false_accepts") == 0,
        "accepted_precision_not_degraded": None if after is None else (
            after["accepted_precision"] >= before["accepted_precision"]
        ),
        "source_b_accuracy_improved": None if after is None else (
            after["raw_accuracy"] > before["raw_accuracy"]
        ),
        "hitl_decreased": None if after is None else (
            after["claim_hitl"] < before["claim_hitl"]
        ),
        "challenger_budget_at_most_30_percent": None,
        "latency_cost_satisfied": None,
    }
    passed = all(value is True for value in gates.values())
    report = {
        "phase": "8.21",
        "authority": "FROZEN_REPLAY_COMPARATIVE",
        "inputs": {
            _label(baseline_path): _sha256(baseline_path),
            _label(shadow_path): _sha256(shadow_path),
        },
        "thresholds_changed": False,
        "before": before,
        "after": after,
        "challenger_evidence": {
            "evaluated_fields": shadow.get("evaluated_fields"),
            "incremental_correct_candidates": incremental,
            "critical_false_accepts": shadow.get("critical_false_accepts"),
            "production_values_overwritten": shadow.get("production_values_overwritten"),
            "candidate_authority": shadow.get("candidate_authority"),
            "comparable_frozen_source_b_population": comparable_source_b,
        },
        "acceptance_gates": gates,
        "verdict": "PASS" if passed else "NEEDS_MORE_DATA",
        "remaining_blockers": [] if passed else [
            "Fresh PP-OCRv5 evidence is not keyed to the frozen Phase 8.20 Source-B corpus.",
            "Aggregate eligible/challenged counts and claim-unlock attribution are unavailable.",
            "A production-authority after score cannot be inferred from review-only shadow crops.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(
        baseline_path=args.baseline,
        shadow_path=args.shadow,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
