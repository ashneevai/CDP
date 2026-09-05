from evaluation.phase8_25_evidence_discovery import run


def test_real_bundle_discovery_rejects_shared_crop_and_preserves_safety(tmp_path):
    report = run(tmp_path)
    metrics = report["metrics"]
    assert report["verdict"] == "REJECT"
    assert metrics["remaining_blockers_before"] == 134
    assert metrics["remaining_blockers_after"] == 134
    assert metrics["valid_independent_pairs"] == 0
    assert metrics["duplicate_rejections"] == 42
    assert metrics["blockers_removed"] == 0
    assert metrics["claims_unlocked"] == 0
    assert metrics["critical_false_accepts_after"] == 0
    assert report["acceptance_gates"]["shared_provenance_rejected"]


def test_phase8_25_writes_exact_required_artifacts(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "blocker_inventory.json", "evidence_opportunity_analysis.json",
        "evidence_discovery_replay.jsonl", "independence_metrics.json",
        "claim_unlock_analysis.json", "comparative_report.json",
    }
