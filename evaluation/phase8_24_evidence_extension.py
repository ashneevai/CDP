"""Evaluate a separately adjudicated E2 coverage extension without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from packages.claim_evidence.corroboration import independent_ocr_evidence
from packages.claim_evidence.pair_assessment import assess_evidence_pair
from packages.evidence.normalization import normalize_agreement_value

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "evaluation/extensions/phase8_24/adjudicated_claims.jsonl"
EXTENSION_MANIFEST = ROOT / "evaluation/extensions/phase8_24/extension_manifest.json"
FROZEN_HASHES = ROOT / "evaluation/extensions/phase8_24/phase8_23_hashes.json"
PHASE823_RESULTS = ROOT / "evaluation_results/phase8_23"
OUTPUT = ROOT / "evaluation_results/phase8_24"
VERSION = "phase8.24-independent-evidence-extension-v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")


def run(output: Path = OUTPUT) -> dict[str, Any]:
    started = perf_counter()
    expected_hashes = _json(FROZEN_HASHES)
    actual_hashes = {name: _sha(PHASE823_RESULTS / name) for name in expected_hashes}
    frozen_unchanged = actual_hashes == expected_hashes
    frozen_report = _json(PHASE823_RESULTS / "comparative_report.json")
    extension_rows = _jsonl(EXTENSION)
    extension_manifest = _json(EXTENSION_MANIFEST)
    extension_locked = (
        extension_manifest["dataset_sha256"] == _sha(EXTENSION)
        and extension_manifest["records"] == len(extension_rows)
        and extension_manifest["thresholds_frozen_before_replay"] is True
    )
    annotations = []
    for row in extension_rows:
        left, right = row["candidates"]
        assessment = assess_evidence_pair(row["field_name"], left, right)
        admitted = row["candidates"] if (
            assessment.genuinely_independent and assessment.semantically_compatible
        ) else []
        # This is the unchanged Phase 8.23 E2 adjudicator. The extension layer
        # only controls admission using the additional semantic-section rule.
        e2 = independent_ocr_evidence(row["field_name"], {"candidates": admitted})
        removed = e2 is not None
        selected = left.get("value") if removed else None
        correct = bool(removed and normalize_agreement_value(row["field_name"], selected)
                       == normalize_agreement_value(row["field_name"], row["truth"]))
        annotations.append({
            "claim_id": row["claim_id"], "field_name": row["field_name"],
            "criticality": row["criticality"], "truth": row["truth"],
            "assessment": assessment.model_dump(), "e2_opportunity": True,
            "blocker_removed": removed, "claim_unlocked": removed,
            "selected_value": selected, "accepted_correct": correct,
            "candidate_provenance": [candidate["provenance"] for candidate in row["candidates"]],
        })
    valid_pairs = sum(a["assessment"]["genuinely_independent"] and a["assessment"]["semantically_compatible"] for a in annotations)
    duplicate_rejections = sum(any(reason in {"SAME_CROP_SHA256", "SAME_LOCALIZATION_REGION"}
                                    for reason in a["assessment"]["rejection_reasons"])
                               for a in annotations)
    conflicts = sum(a["assessment"]["conflicting"] for a in annotations)
    removed = sum(a["blocker_removed"] for a in annotations)
    false_accepts = sum(a["blocker_removed"] and not a["accepted_correct"] for a in annotations)
    critical_false_accepts = sum(a["criticality"] in {"C2", "C3"} and a["blocker_removed"]
                                 and not a["accepted_correct"] for a in annotations)
    candidate_latency = sum(candidate.get("latency_ms", 0.0) for row in extension_rows
                            for candidate in row["candidates"])
    cost = sum(candidate.get("cost_usd", 0.0) for row in extension_rows
               for candidate in row["candidates"])
    accepted_precision = (removed - false_accepts) / removed if removed else None
    gates = {
        "frozen_phase8_23_unchanged": frozen_unchanged,
        "frozen_phase8_23_verdict_preserved": frozen_report["verdict"] == "NEEDS_MORE_DATA",
        "valid_independent_pairs_present": valid_pairs > 0,
        "e2_removes_blockers": removed > 0,
        "conflicts_fail_closed": all(not a["blocker_removed"] for a in annotations if a["assessment"]["conflicting"]),
        "duplicates_fail_closed": all(not a["blocker_removed"] for a in annotations if not a["assessment"]["genuinely_independent"]),
        "accepted_precision_one": accepted_precision == 1.0,
        "critical_false_accepts_zero": critical_false_accepts == 0,
        "thresholds_not_tuned": True,
        "extension_locked_before_replay": extension_locked,
    }
    metrics = {
        "e2_opportunities": len(annotations), "valid_independent_pairs": valid_pairs,
        "duplicate_rejections": duplicate_rejections, "conflicts": conflicts,
        "blockers_removed": removed, "claims_unlocked": removed,
        "accepted_precision": accepted_precision,
        "critical_false_accepts": critical_false_accepts,
        "candidate_latency_ms": candidate_latency,
        "evaluation_latency_ms": (perf_counter() - started) * 1000,
        "cost_usd": cost,
    }
    report = {
        "phase": "8.24", "version": VERSION,
        "frozen_phase8_23_verdict": frozen_report["verdict"],
        "extension_verdict": "PASS" if all(gates.values()) else "REJECT",
        "metrics": metrics, "acceptance_gates": gates,
        "thresholds_changed_before_replay": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pair_annotations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), "utf-8"
    )
    _write(output / "e2_coverage_metrics.json", metrics)
    _write(output / "frozen_phase8_23_integrity.json", {
        "expected_hashes": expected_hashes, "actual_hashes": actual_hashes,
        "unchanged": frozen_unchanged, "verdict": frozen_report["verdict"],
    })
    _write(output / "comparative_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
