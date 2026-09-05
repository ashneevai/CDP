import json

from evaluation.phase9b_name_replay import run

EXPECTED_ARTIFACTS = {
    "name_error_cohort.json",
    "provider_name_analysis.json",
    "patient_insured_name_analysis.json",
    "token_geometry_metrics.json",
    "localization_metrics.json",
    "regression_analysis.json",
    "claim_unlock_distance.json",
    "comparative_report.json",
}


def test_phase9b_frozen_name_replay_meets_safety_gates(tmp_path):
    report = run(tmp_path)
    metrics = report["metrics"]

    assert report["verdict"] == "PASS"
    assert all(report["acceptance_gates"].values())
    assert metrics["accepted_precision"] >= 0.995
    assert metrics["critical_false_accepts"] == 0
    assert metrics["regressions"] == 0
    assert metrics["provider_name_errors_after"] == 2
    assert metrics["patient_name_errors_after"] == 1
    assert metrics["insured_name_errors_after"] == 0
    assert metrics["blockers_removed"] == 1
    assert metrics["ocr_calls_per_claim_after"] == metrics["ocr_calls_per_claim_before"]
    assert metrics["cost_per_page_after"] == 0.0


def test_phase9b_reports_complete_frozen_cohort_and_artifacts(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_ARTIFACTS

    cohort = json.loads((tmp_path / "name_error_cohort.json").read_text("utf-8"))
    assert cohort["frozen_size"] == 13
    assert len(cohort["records"]) == 13
    assert sum(record["after_correctness"] for record in cohort["records"]) == 10
    assert all(record["selected_tokens"] for record in cohort["records"])

    distances = json.loads((tmp_path / "claim_unlock_distance.json").read_text("utf-8"))
    assert distances["claims_unlocked"] == []
    assert all(
        row["minimum_remaining_blockers_to_unlock"] == row["blockers_remaining"]
        for row in distances["claims"]
    )
