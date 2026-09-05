from evaluation.phase8_26_evidence_availability import run


def test_all_remaining_e2_blockers_are_classified_once(tmp_path):
    report = run(tmp_path)
    assert report["remaining_e2_blockers"] == 61
    assert sum(report["root_cause_distribution"].values()) == 61
    assert report["acceptance_gates"]["all_e2_blockers_classified_once"]
    assert report["acceptance_decisions_created"] == 0


def test_missing_raw_bundles_yield_needs_more_data_not_invented_capture_gap(tmp_path):
    report = run(tmp_path)
    assert report["verdict"] == "NEEDS_MORE_DATA"
    assert report["root_cause_distribution"]["UNKNOWN_REQUIRES_SOURCE_AUDIT"] > 0
    distribution = __import__("json").loads(
        (tmp_path / "root_cause_distribution.json").read_text("utf-8")
    )
    assert distribution["process_capture_gap_blockers"] == 0


def test_phase8_26_writes_six_required_artifacts(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "evidence_availability_inventory.json", "ingestion_gap_analysis.json",
        "attachment_opportunity_analysis.json", "root_cause_distribution.json",
        "claim_unlock_potential.json", "comparative_report.json",
    }
