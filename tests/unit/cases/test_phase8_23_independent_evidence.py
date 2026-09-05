from evaluation.phase8_23_independent_evidence import run


def test_phase8_23_frozen_replay_is_provenance_safe(tmp_path):
    report = run(tmp_path)
    assert report["metrics"]["target_blockers"] == 110
    assert report["metrics"]["critical_false_accepts"] == 0
    assert report["metrics"]["blockers_removed"] > 0
    assert report["acceptance_gates"]["shared_crop_never_counted_twice"]
    assert report["verdict"] == "NEEDS_MORE_DATA"
    assert report["evidence_class_metrics"]["E2"]["satisfied"] == 0
    assert len(list(tmp_path.glob("*.json"))) == 6
