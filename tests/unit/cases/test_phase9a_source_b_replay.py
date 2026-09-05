from evaluation.phase9a_source_b_replay import run


def test_phase9a_frozen_replay_keeps_only_safe_measured_policy(tmp_path):
    report = run(tmp_path)
    metrics = report["metrics"]
    assert report["verdict"] == "PASS"
    assert report["experiments_retained"] == ["9A-1"]
    assert metrics["raw_accuracy_after"] > metrics["raw_accuracy_before"]
    assert metrics["field_hitl_after"] < metrics["field_hitl_before"]
    assert metrics["accepted_precision_after"] == 1.0
    assert metrics["critical_false_accepts_after"] == 0
    assert metrics["policy_blockers_removed"] == 30


def test_phase9a_reports_all_required_artifacts_and_no_regressions(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "blocker_cohort_analysis.json",
        "localization_analysis.json",
        "token_geometry_analysis.json",
        "field_extractor_metrics.json",
        "evidence_policy_metrics.json",
        "regression_analysis.json",
        "claim_unlock_analysis.json",
        "comparative_report.json",
    }
    import json

    regression = json.loads((tmp_path / "regression_analysis.json").read_text("utf-8"))
    assert regression["regression_count"] == 0
