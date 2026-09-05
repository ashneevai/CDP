import json
import sys

import pytest

from evaluation.phase8_15_shadow_qualification import main
from packages.shadow_evaluation import (
    AppendOnlyShadowClaimSink,
    ClaimShadowObservation,
    qualify_shadow_claims,
)


def observation(index: int, **changes) -> ClaimShadowObservation:
    values = {
        "claim_id": f"claim-{index}",
        "source_group_id": f"source-{index}",
        "source_segment": "CMS1500_SCANNER_A",
        "production_requires_review": True,
        "shadow_requires_review": index < 50,
        "evaluated_field_decisions": 10,
        "correct_field_decisions": 10,
        "evaluated_critical_field_decisions": 3,
        "correct_critical_field_decisions": 3,
        "accepted_field_decisions": 10,
        "accepted_critical_field_decisions": 3,
        "correct_accepted_field_decisions": 10,
        "correct_accepted_critical_field_decisions": 3,
        "false_accepts": 0,
        "critical_false_accepts": 0,
        "wrong_crops": 1,
        "wrong_crops_detected": 1,
        "runtime_latency_ms": 100,
        "cost_usd": .01,
        "runtime_decision_parity": True,
        "route_governance_passed": True,
    }
    values.update(changes)
    return ClaimShadowObservation(**values)


def test_qualifies_only_with_buffered_hitl_and_required_volume():
    report = qualify_shadow_claims([observation(index) for index in range(1000)])
    assert report.status == "QUALIFIED"
    assert report.claim_hitl == .05
    assert report.claim_stp == .95
    assert report.claim_hitl_upper_95 < .10
    assert report.promotion_authority is False


def test_segment_regression_blocks_even_when_overall_hitl_passes():
    rows = [observation(index) for index in range(1000)]
    for index in range(100):
        rows[index] = observation(
            index,
            source_segment="FAX_LOW_QUALITY",
            shadow_requires_review=index < 20,
        )
    report = qualify_shadow_claims(rows)
    assert report.claim_hitl < .08
    assert report.maximum_segment_claim_hitl == .20
    assert "SEGMENT_CLAIM_HITL" in report.blocking_reasons


def test_critical_raw_accuracy_is_independent_of_accepted_precision():
    report = qualify_shadow_claims([
        observation(
            1,
            evaluated_critical_field_decisions=4,
            correct_critical_field_decisions=3,
            accepted_critical_field_decisions=2,
            correct_accepted_critical_field_decisions=2,
        )
    ])
    assert report.critical_accuracy == .75
    assert report.critical_accepted_precision == 1.0
    assert not report.gates["critical_raw_accuracy"]


def test_raw_accuracy_and_accepted_precision_use_independent_denominators():
    report = qualify_shadow_claims([
        observation(
            1,
            evaluated_field_decisions=10,
            correct_field_decisions=8,
            accepted_field_decisions=5,
            correct_accepted_field_decisions=5,
        )
    ])
    assert report.overall_raw_accuracy == .8
    assert report.accepted_precision == 1.0
    assert not report.gates["overall_raw_accuracy"]
    assert report.gates["accepted_precision"]


def test_inconsistent_false_accept_counts_are_rejected():
    with pytest.raises(ValueError, match="do not reconcile"):
        observation(
            1,
            accepted_field_decisions=10,
            correct_accepted_field_decisions=9,
            false_accepts=0,
        )


def test_training_source_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        qualify_shadow_claims(
            [observation(1)], prohibited_source_groups={"source-1"}
        )


def test_shadow_record_cannot_claim_serving_authority():
    with pytest.raises(ValueError, match="serving authority"):
        qualify_shadow_claims([observation(1, shadow_only=False)])


def test_impossible_adjudication_counts_are_rejected():
    with pytest.raises(ValueError, match="wrong crops detected"):
        observation(1, wrong_crops=0, wrong_crops_detected=1)


def test_cli_rejects_correction_overlap_for_deidentified_ledger(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "shadow.jsonl"
    AppendOnlyShadowClaimSink(ledger, identity_key=b"secret").append(observation(1))
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    (corrections / "train.jsonl").write_text(
        json.dumps({"source_group_id": "source-1"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("SHADOW_IDENTITY_KEY", "secret")
    monkeypatch.setattr(sys, "argv", [
        "phase8_15_shadow_qualification.py",
        str(ledger),
        "--correction-dataset",
        str(corrections),
    ])
    with pytest.raises(ValueError, match="overlaps"):
        main()


def test_cli_requires_identity_key_for_ledger_overlap_check(tmp_path, monkeypatch):
    ledger = tmp_path / "shadow.jsonl"
    AppendOnlyShadowClaimSink(ledger, identity_key=b"secret").append(observation(1))
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    monkeypatch.delenv("SHADOW_IDENTITY_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "phase8_15_shadow_qualification.py",
        str(ledger),
        "--correction-dataset",
        str(corrections),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
