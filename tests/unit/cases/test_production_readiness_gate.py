from datetime import UTC, datetime

from packages.production_readiness_gate import (
    ApprovalRecord,
    ApprovalRole,
    ProductionReadinessGate,
    ReadinessDecision,
    ReadinessEvidence,
)

RELEASE_SHA = "a" * 40
EVIDENCE_SHA = "b" * 64


def approvals():
    return [
        ApprovalRecord(
            role=role,
            approver_id=f"approver-{index}",
            approved_at=datetime.now(UTC),
            release_commit_sha=RELEASE_SHA,
            evidence_bundle_sha256=EVIDENCE_SHA,
            approval_signature_sha256=(f"{index:x}" * 64)[:64],
            signature_verified=True,
        )
        for index, role in enumerate(ApprovalRole, 1)
    ]


def passing(**changes):
    values = {
        "holdout_frozen": True, "holdout_independent": True,
        "holdout_documents": 5000, "holdout_fields": 15000,
        "full_suite_passed": True,
        "overall_raw_accuracy": .96, "critical_accuracy": .99,
        "total_false_accept_rate": 0, "critical_false_accept_count": 0,
        "safe_field_coverage": .95, "accepted_precision": .999,
        "claim_stp": .95, "claim_hitl": .05,
        "ocr_only_processing_rate": .995, "llm_escalation_rate": .005,
        "claim_hitl_count": 250, "accepted_critical_field_decisions": 4000,
        "critical_accepted_precision": .999, "wrong_crop_recall": .97,
        "maximum_segment_claim_hitl": .10,
        "p95_latency_ms": 1000, "cost_per_document_usd": .01,
        "runtime_parity_passed": True, "route_governance_passed": True,
        "security_passed": True, "database_and_events_passed": True,
        "load_and_keda_passed": True, "shadow_validation_passed": True,
        "failure_injection_passed": True,
        "release_commit_sha": RELEASE_SHA, "evidence_bundle_sha256": EVIDENCE_SHA,
        "approvals": approvals(),
    }
    values.update(changes)
    return ReadinessEvidence(**values)


def test_all_gates_promote_to_production():
    result = ProductionReadinessGate.load().evaluate(passing())
    assert result.decision is ReadinessDecision.PROMOTE_TO_PRODUCTION


def test_holdout_only_can_promote_to_shadow_but_not_production():
    result = ProductionReadinessGate.load().evaluate(passing(
        security_passed=False, database_and_events_passed=False,
        load_and_keda_passed=False, shadow_validation_passed=False,
    ))
    assert result.decision is ReadinessDecision.PROMOTE_TO_SHADOW


def test_missing_holdout_needs_more_data():
    result = ProductionReadinessGate.load().evaluate(ReadinessEvidence(
        runtime_parity_passed=True, route_governance_passed=True,
    ))
    assert result.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_observed_critical_false_accept_rejects():
    result = ProductionReadinessGate.load().evaluate(passing(
        critical_false_accept_count=1,
    ))
    assert result.decision is ReadinessDecision.REJECT


def test_overall_accepted_precision_is_an_independent_gate():
    result = ProductionReadinessGate.load().evaluate(passing(
        accepted_precision=.994, critical_accepted_precision=1.0,
    ))
    assert not result.gates["accepted_precision"]
    assert result.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_point_estimate_cannot_bypass_confidence_bound():
    result = ProductionReadinessGate.load().evaluate(passing(
        holdout_documents=100, holdout_fields=3000, claim_hitl=.08,
        claim_hitl_count=8,
    ))
    assert not result.gates["claim_hitl_upper_confidence"]
    assert result.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_measured_cost_above_ceiling_cannot_promote():
    result = ProductionReadinessGate.load().evaluate(
        passing(cost_per_document_usd=0.030001)
    )
    assert result.gates["measured_cost"]
    assert not result.gates["cost_ceiling"]
    assert result.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_p95_above_five_seconds_cannot_promote():
    result = ProductionReadinessGate.load().evaluate(passing(p95_latency_ms=5000.01))
    assert not result.gates["p95_latency"]
    assert result.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_ocr_first_objective_is_a_binding_production_gate():
    low_ocr = ProductionReadinessGate.load().evaluate(
        passing(ocr_only_processing_rate=.989)
    )
    assert not low_ocr.gates["ocr_only_processing"]
    assert low_ocr.decision is ReadinessDecision.NEEDS_MORE_DATA

    high_llm = ProductionReadinessGate.load().evaluate(
        passing(llm_escalation_rate=.011)
    )
    assert not high_llm.gates["llm_escalation"]
    assert high_llm.decision is ReadinessDecision.NEEDS_MORE_DATA


def test_missing_named_release_approvals_cannot_promote_beyond_shadow():
    result = ProductionReadinessGate.load().evaluate(
        passing(release_commit_sha=None, approvals=[])
    )
    assert not result.gates["named_release_approvals"]
    assert not result.gates["approval_signatures"]
    assert result.decision is ReadinessDecision.PROMOTE_TO_SHADOW


def test_one_person_cannot_satisfy_all_independent_approval_roles():
    records = [record.model_copy(update={"approver_id": "same-person"}) for record in approvals()]
    result = ProductionReadinessGate.load().evaluate(passing(approvals=records))
    assert not result.gates["named_release_approvals"]
    assert result.decision is ReadinessDecision.PROMOTE_TO_SHADOW


def test_approvals_for_another_evidence_bundle_cannot_promote():
    result = ProductionReadinessGate.load().evaluate(
        passing(evidence_bundle_sha256="c" * 64)
    )
    assert not result.gates["named_release_approvals"]
    assert result.decision is ReadinessDecision.PROMOTE_TO_SHADOW
