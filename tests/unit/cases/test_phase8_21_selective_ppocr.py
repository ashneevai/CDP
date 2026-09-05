import json

from evaluation.phase8_21_selective_ppocr import run


def test_report_fails_closed_when_shadow_is_not_joinable_to_source_b(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    shadow_path = tmp_path / "shadow.json"
    output_path = tmp_path / "report.json"
    baseline_path.write_text(json.dumps({"before": {
        "accepted_precision": 1.0, "claim_hitl": 1.0, "raw_accuracy": .8,
        "critical_false_accepts": 0,
    }}))
    shadow_path.write_text(json.dumps({
        "evaluated_fields": 23, "incremental_correct_candidates": 0,
        "critical_false_accepts": 0, "production_values_overwritten": 0,
        "candidate_authority": "REVIEW_ONLY",
    }))
    report = run(
        baseline_path=baseline_path,
        shadow_path=shadow_path,
        output_path=output_path,
    )
    assert report["after"] is None
    assert report["verdict"] == "NEEDS_MORE_DATA"
    assert report["thresholds_changed"] is False
    assert output_path.exists()


def test_report_never_treats_review_only_shadow_as_production_after(tmp_path):
    baseline = tmp_path / "baseline.json"
    shadow = tmp_path / "shadow.json"
    baseline.write_text(json.dumps({"before": {"accepted_precision": 1, "claim_hitl": 1, "raw_accuracy": .5}}))
    shadow.write_text(json.dumps({"incremental_correct_candidates": 10, "critical_false_accepts": 0}))
    assert run(baseline_path=baseline, shadow_path=shadow, output_path=tmp_path / "out.json")["after"] is None
