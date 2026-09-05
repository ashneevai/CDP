import json

from evaluation.phase8_20_claim_bundles import run


def test_phase8_20_emits_all_required_diagnostic_deliverables(tmp_path):
    report = run(tmp_path)
    expected = {
        "claim_blocker_matrix.json", "bundle_pareto.json", "before_after_scorecard.json",
        "segment_scorecard.json", "claims_unlocked.json", "accuracy_error_taxonomy.json",
        "latency_cost_comparison.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    scorecard = json.loads((tmp_path / "before_after_scorecard.json").read_text())
    assert scorecard["production_decision"] == "NEEDS_MORE_DATA"
    assert scorecard["actual_complete_claims_unlocked"] == 0
    assert scorecard["thresholds_changed"] is False
    assert report["analysis"]["claim_blocker_matrix"]


def test_phase8_20_does_not_invent_quality_segments_or_acceptance_authority(tmp_path):
    report = run(tmp_path)
    assert all(
        row["quality_segment"] == "UNKNOWN_NOT_CAPTURED"
        for row in report["analysis"]["claim_blocker_matrix"]
    )
    assert all(
        item["production_authority"] is False
        for item in report["analysis"]["bundle_pareto"]
    )
