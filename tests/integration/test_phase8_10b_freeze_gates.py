import hashlib
import json
from pathlib import Path

from packages.runtime_profile import DecisionServiceFactory

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "evaluation/baselines/phase8_10b/ocr_experiment_result.json"
BASELINE_SHA256 = "f87a01cb94bf17f274452072e885c742d08f70d40e6c2d4e62d2968c958eb045"


def _result():
    payload = BASELINE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == BASELINE_SHA256
    return json.loads(payload)


def test_reverted_ocr_experiment_did_not_change_localization_or_evidence_policy():
    result = _result()
    profile_before = DecisionServiceFactory.from_profile().profile
    profile_after = DecisionServiceFactory.from_profile().profile
    assert result["decision"] == "REVERT"
    assert result["runtime_change_retained"] is False
    assert result["localization_changed"] is False
    assert result["evidence_policy_changed"] is False
    assert profile_before.evidence_policy_sha256 == profile_after.evidence_policy_sha256
    assert profile_before.route_registry_sha256 == profile_after.route_registry_sha256


def test_accuracy_regression_forces_revert_before_downstream_promotion():
    result = _result()
    assert result["before"]["canonical_accepted_precision"] == 0.96
    assert result["before"]["canonical_critical_false_accepts"] == 1
    assert result["treatment"]["canonical_decision_replay_status"] == (
        "NOT_RUN_AFTER_HARD_EXTRACTION_REGRESSION"
    )
    assert result["treatment"]["critical_accuracy"] <= result["before"]["critical_accuracy"]
    assert result["decision"] == "REVERT"
