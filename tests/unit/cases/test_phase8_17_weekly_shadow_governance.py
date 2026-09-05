import json

import pytest

from evaluation.phase8_17_weekly_shadow_governance import (
    generate_weekly_governance,
)
from packages.production_readiness_gate import ReadinessEvidence
from packages.shadow_evaluation import (
    AppendOnlyShadowClaimSink,
    ClaimShadowObservation,
    fingerprinted_source_groups,
)


def observation(index: int) -> ClaimShadowObservation:
    return ClaimShadowObservation(
        claim_id=f"claim-{index}", source_group_id=f"source-{index}",
        source_segment="CMS1500_SCANNER_A", production_requires_review=True,
        shadow_requires_review=index < 50, evaluated_field_decisions=10,
        correct_field_decisions=10, evaluated_critical_field_decisions=3,
        correct_critical_field_decisions=3, accepted_field_decisions=10,
        accepted_critical_field_decisions=3, correct_accepted_field_decisions=10,
        correct_accepted_critical_field_decisions=3, false_accepts=0,
        critical_false_accepts=0, wrong_crops=1, wrong_crops_detected=1,
        runtime_latency_ms=100, cost_usd=.01, runtime_decision_parity=True,
        route_governance_passed=True,
    )


def operational_evidence() -> ReadinessEvidence:
    return ReadinessEvidence(
        full_suite_passed=True, security_passed=True,
        database_and_events_passed=True, load_and_keda_passed=True,
        failure_injection_passed=True,
    )


def test_weekly_artifact_is_hash_addressed_and_non_authoritative(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    sink = AppendOnlyShadowClaimSink(ledger, identity_key=b"secret")
    for index in range(1000):
        sink.append(observation(index))
    first = generate_weekly_governance(
        ledger, as_of_week="2026-W36", base_evidence=operational_evidence()
    )
    second = generate_weekly_governance(
        ledger, as_of_week="2026-W36", base_evidence=operational_evidence()
    )
    assert first.exit_code == 0
    assert first.artifact["artifact_sha256"] == second.artifact["artifact_sha256"]
    assert first.artifact["promotion_authority"] is False
    assert first.artifact["shadow_qualification"]["claim_hitl"] == .05
    assert first.artifact["shadow_qualification"]["ocr_only_processing_rate"] == 1.0
    assert first.artifact["shadow_qualification"]["llm_escalation_rate"] == 0.0
    # Weekly shadow evidence is deliberately non-authoritative. Even perfect
    # metrics cannot bypass the separately signed production approval chain.
    assert first.artifact["production_readiness"]["decision"] == "PROMOTE_TO_SHADOW"
    blockers = first.artifact["production_readiness"]["blocking_reasons"]
    assert "NAMED_RELEASE_APPROVALS" in blockers
    assert "APPROVAL_SIGNATURES" in blockers


def test_insufficient_week_fails_closed(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    AppendOnlyShadowClaimSink(ledger, identity_key=b"secret").append(observation(1))
    result = generate_weekly_governance(ledger, as_of_week="2026-W36")
    assert result.exit_code == 2
    assert result.artifact["shadow_qualification"]["status"] == "NEEDS_MORE_DATA"
    assert result.artifact["production_readiness"]["decision"] == "NEEDS_MORE_DATA"


def test_more_than_one_percent_llm_escalation_fails_shadow_qualification(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    sink = AppendOnlyShadowClaimSink(ledger, identity_key=b"secret")
    for index in range(1000):
        sink.append(observation(index).model_copy(update={"llm_escalated": index < 11}))
    result = generate_weekly_governance(ledger, as_of_week="2026-W36")
    assert result.artifact["shadow_qualification"]["llm_escalation_rate"] == .011
    assert "LLM_ESCALATION" in result.artifact["shadow_qualification"]["blocking_reasons"]


def test_weekly_governance_rejects_correction_source_overlap(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    sink = AppendOnlyShadowClaimSink(ledger, identity_key=b"secret")
    sink.append(observation(1))
    captured_source = sink.observations()[0].source_group_id
    with pytest.raises(ValueError, match="overlaps"):
        generate_weekly_governance(
            ledger,
            as_of_week="2026-W36",
            prohibited_source_groups={captured_source},
        )


def test_correction_source_groups_use_capture_fingerprints(tmp_path):
    correction_dataset = tmp_path / "corrections"
    correction_dataset.mkdir()
    (correction_dataset / "train.jsonl").write_text(
        json.dumps({"source_group_id": "source-1"}) + "\n", encoding="utf-8"
    )
    groups = fingerprinted_source_groups(
        correction_dataset, {"train", "calibration"}, identity_key=b"secret"
    )
    ledger = tmp_path / "shadow.jsonl"
    sink = AppendOnlyShadowClaimSink(ledger, identity_key=b"secret")
    sink.append(observation(1))
    assert groups == {sink.observations()[0].source_group_id}


def test_correction_source_groups_fail_closed_without_source_identity(tmp_path):
    correction_dataset = tmp_path / "corrections"
    correction_dataset.mkdir()
    (correction_dataset / "train.jsonl").write_text(
        json.dumps({"document_id": "source-1"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing source_group_id"):
        fingerprinted_source_groups(
            correction_dataset, {"train"}, identity_key=b"secret"
        )
