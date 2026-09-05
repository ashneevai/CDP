import json

from evaluation.phase9c_claim_unlock import TARGETS, run

EXPECTED_ARTIFACTS = {
    "claim_unlock_baseline.json",
    "target_claim_blockers.json",
    "member_id_analysis.json",
    "total_charge_reconciliation.json",
    "claim_unlock_distance.json",
    "regression_analysis.json",
    "source_evidence_limitations.json",
    "comparative_report.json",
}


def test_phase9c_moves_all_target_claims_from_distance_four_to_three(tmp_path):
    report = run(tmp_path)
    metrics = report["metrics"]

    assert report["verdict"] == "PASS"
    assert all(report["acceptance_gates"].values())
    assert metrics["blockers_removed"] == 7
    assert metrics["claims_advanced_but_not_unlocked"] == 7
    assert metrics["claims_unlocked"] == 0
    assert metrics["field_hitl_after"] < metrics["field_hitl_before"]
    assert metrics["accepted_precision"] == 1.0
    assert metrics["critical_false_accepts"] == 0
    assert metrics["distance_before"]["distance_4_claims"] == 7
    assert metrics["distance_after"]["distance_4_claims"] == 0
    assert metrics["distance_after"]["distance_3_claims"] == 7


def test_phase9c_charge_acceptance_has_exact_arithmetic_and_provenance(tmp_path):
    run(tmp_path)
    charge = json.loads((tmp_path / "total_charge_reconciliation.json").read_text("utf-8"))

    assert len(charge["records"]) == len(TARGETS)
    assert all(row["reconciliation_status"] == "EXACT_MATCH" for row in charge["records"])
    assert all(row["difference"] == "0.00" for row in charge["records"])
    assert all(row["header_observation_provenance"]["crop_sha256"] for row in charge["records"])
    assert all(row["service_line_observation_provenance"] for row in charge["records"])
    assert all(row["accepted"] for row in charge["records"])


def test_phase9c_member_id_fails_closed_without_second_observation(tmp_path):
    run(tmp_path)
    member = json.loads((tmp_path / "member_id_analysis.json").read_text("utf-8"))

    assert member["result"] == "REVERTED_NO_ADMISSIBLE_ACCEPTANCE"
    assert all(not row["accepted"] for row in member["records"])
    assert all(row["reason"] == "SOURCE_EVIDENCE_REQUIRED" for row in member["records"])


def test_phase9c_writes_complete_claim_level_artifact_set(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_ARTIFACTS

    baseline = json.loads((tmp_path / "claim_unlock_baseline.json").read_text("utf-8"))
    target_rows = [row for row in baseline["claims"] if row["claim_id"] in TARGETS]
    assert len(target_rows) == 7
    assert all(row["claim_unlock_distance"] == 4 for row in target_rows)
