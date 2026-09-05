from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import StrEnum
from math import sqrt
from pathlib import Path

import yaml
from pydantic import ConfigDict, Field, field_validator

from packages.domain.common import DomainModel


class ReadinessDecision(StrEnum):
    PROMOTE_TO_SHADOW = "PROMOTE_TO_SHADOW"
    PROMOTE_TO_PRODUCTION = "PROMOTE_TO_PRODUCTION"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"
    REJECT = "REJECT"


class ApprovalRole(StrEnum):
    OPERATIONS = "OPERATIONS"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"
    DATA_GOVERNANCE = "DATA_GOVERNANCE"
    RELEASE = "RELEASE"


class ApprovalRecord(DomainModel):
    role: ApprovalRole
    approver_id: str = Field(min_length=1, max_length=128)
    approved_at: datetime
    release_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    evidence_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature_verified: bool = False

    @field_validator("approved_at")
    @classmethod
    def approval_timestamp_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("APPROVAL_TIMESTAMP_MUST_BE_TIMEZONE_AWARE")
        return value


class ReadinessEvidence(DomainModel):
    model_config = ConfigDict(extra="forbid")
    holdout_frozen: bool = False
    holdout_independent: bool = False
    holdout_documents: int = Field(default=0, ge=0)
    holdout_fields: int = Field(default=0, ge=0)
    full_suite_passed: bool = False
    overall_raw_accuracy: float | None = Field(default=None, ge=0, le=1)
    critical_accuracy: float | None = Field(default=None, ge=0, le=1)
    total_false_accept_rate: float | None = Field(default=None, ge=0, le=1)
    critical_false_accept_count: int | None = Field(default=None, ge=0)
    safe_field_coverage: float | None = Field(default=None, ge=0, le=1)
    accepted_precision: float | None = Field(default=None, ge=0, le=1)
    claim_stp: float | None = Field(default=None, ge=0, le=1)
    claim_hitl: float | None = Field(default=None, ge=0, le=1)
    claim_hitl_count: int | None = Field(default=None, ge=0)
    ocr_only_processing_rate: float | None = Field(default=None, ge=0, le=1)
    llm_escalation_rate: float | None = Field(default=None, ge=0, le=1)
    accepted_critical_field_decisions: int = Field(default=0, ge=0)
    critical_accepted_precision: float | None = Field(default=None, ge=0, le=1)
    wrong_crop_recall: float | None = Field(default=None, ge=0, le=1)
    maximum_segment_claim_hitl: float | None = Field(default=None, ge=0, le=1)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    cost_per_document_usd: float | None = Field(default=None, ge=0)
    runtime_parity_passed: bool = False
    route_governance_passed: bool = False
    security_passed: bool = False
    database_and_events_passed: bool = False
    load_and_keda_passed: bool = False
    shadow_validation_passed: bool = False
    failure_injection_passed: bool = False
    release_commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    evidence_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approvals: list[ApprovalRecord] = Field(default_factory=list)


class ReadinessResult(DomainModel):
    decision: ReadinessDecision
    policy_version: str
    gates: dict[str, bool]
    blocking_reasons: list[str]
    evidence_status: dict[str, str]


class ProductionReadinessGate:
    def __init__(self, config: dict) -> None:
        self.config = config

    @classmethod
    def load(
        cls, path: str | Path = "config/production_readiness_gate.yaml",
    ) -> ProductionReadinessGate:
        return cls(yaml.safe_load(Path(path).read_text("utf-8")))

    def evaluate(self, evidence: ReadinessEvidence) -> ReadinessResult:
        limits = self.config["requirements"]
        required_approval_roles = {
            ApprovalRole(value) for value in limits["required_approval_roles"]
        }
        release_approvals = [
            approval
            for approval in evidence.approvals
            if evidence.release_commit_sha is not None
            and evidence.evidence_bundle_sha256 is not None
            and approval.release_commit_sha == evidence.release_commit_sha
            and approval.evidence_bundle_sha256 == evidence.evidence_bundle_sha256
        ]
        approval_roles = {approval.role for approval in release_approvals}
        approver_ids = {
            " ".join(unicodedata.normalize("NFKC", approval.approver_id).casefold().split())
            for approval in release_approvals
        }
        approvals_bound = (
            evidence.release_commit_sha is not None
            and evidence.evidence_bundle_sha256 is not None
            and required_approval_roles.issubset(approval_roles)
        )
        approvals_distinct = (
            not limits["require_distinct_approvers"]
            or len(approver_ids) >= len(required_approval_roles)
        )
        approval_signatures_verified = approvals_bound and all(
            approval.signature_verified for approval in release_approvals
        )
        claim_hitl_upper = None
        if evidence.claim_hitl_count is not None and evidence.holdout_documents:
            n = evidence.holdout_documents
            p = evidence.claim_hitl_count / n
            z = 1.959963984540054
            denominator = 1 + z * z / n
            centre = p + z * z / (2 * n)
            margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            claim_hitl_upper = (centre + margin) / denominator
        sufficient_sample = (
            evidence.holdout_documents >= limits["minimum_documents"]
            and evidence.holdout_fields >= limits["minimum_fields"]
        )
        gates = {
            "full_regression_suite": evidence.full_suite_passed,
            "independent_frozen_holdout": evidence.holdout_frozen and evidence.holdout_independent,
            "sample_size": sufficient_sample,
            "accepted_critical_sample_size": evidence.accepted_critical_field_decisions >= limits["minimum_accepted_critical_field_decisions"],
            "overall_raw_accuracy": evidence.overall_raw_accuracy is not None and evidence.overall_raw_accuracy >= limits["minimum_overall_raw_accuracy"],
            "critical_accuracy": evidence.critical_accuracy is not None and evidence.critical_accuracy >= limits["minimum_critical_accuracy"],
            "total_false_accept_rate": evidence.total_false_accept_rate is not None and evidence.total_false_accept_rate <= limits["maximum_total_false_accept_rate"],
            "critical_false_accepts": evidence.critical_false_accept_count is not None and evidence.critical_false_accept_count <= limits["maximum_critical_false_accept_count"],
            "safe_field_coverage": evidence.safe_field_coverage is not None and evidence.safe_field_coverage >= limits["minimum_safe_field_coverage"],
            "accepted_precision": evidence.accepted_precision is not None and evidence.accepted_precision >= limits["minimum_accepted_precision"],
            "claim_stp": evidence.claim_stp is not None and evidence.claim_stp >= limits["minimum_claim_stp"],
            "claim_hitl": evidence.claim_hitl is not None and evidence.claim_hitl <= limits["maximum_claim_hitl"],
            "ocr_only_processing": evidence.ocr_only_processing_rate is not None and evidence.ocr_only_processing_rate >= limits["minimum_ocr_only_processing_rate"],
            "llm_escalation": evidence.llm_escalation_rate is not None and evidence.llm_escalation_rate <= limits["maximum_llm_escalation_rate"],
            "claim_hitl_upper_confidence": claim_hitl_upper is not None and claim_hitl_upper < limits["maximum_claim_hitl_upper_95"],
            "critical_accepted_precision": evidence.critical_accepted_precision is not None and evidence.critical_accepted_precision >= limits["minimum_critical_accepted_precision"],
            "wrong_crop_recall": evidence.wrong_crop_recall is not None and evidence.wrong_crop_recall >= limits["minimum_wrong_crop_recall"],
            "segment_claim_hitl": evidence.maximum_segment_claim_hitl is not None and evidence.maximum_segment_claim_hitl <= limits["maximum_segment_claim_hitl"],
            "p95_latency": evidence.p95_latency_ms is not None and evidence.p95_latency_ms <= limits["maximum_p95_latency_ms"],
            "measured_cost": evidence.cost_per_document_usd is not None,
            "cost_ceiling": (
                evidence.cost_per_document_usd is not None
                and evidence.cost_per_document_usd
                <= limits["maximum_cost_per_document_usd"]
            ),
            "runtime_parity": evidence.runtime_parity_passed,
            "route_governance": evidence.route_governance_passed,
            "security": evidence.security_passed,
            "database_and_events": evidence.database_and_events_passed,
            "load_and_keda": evidence.load_and_keda_passed,
            "shadow_validation": evidence.shadow_validation_passed,
            "failure_injection": evidence.failure_injection_passed,
            "named_release_approvals": approvals_bound and approvals_distinct,
            "approval_signatures": approval_signatures_verified,
        }
        safety_reject = (
            evidence.critical_false_accept_count is not None
            and evidence.critical_false_accept_count > limits["maximum_critical_false_accept_count"]
        ) or (
            evidence.total_false_accept_rate is not None
            and evidence.total_false_accept_rate > limits["maximum_total_false_accept_rate"]
        )
        missing_holdout = not gates["independent_frozen_holdout"] or not gates["sample_size"]
        shadow_gate_names = {
            "full_regression_suite",
            "independent_frozen_holdout", "sample_size", "overall_raw_accuracy",
            "accepted_critical_sample_size",
            "critical_accuracy", "total_false_accept_rate", "critical_false_accepts",
            "accepted_precision", "critical_accepted_precision", "safe_field_coverage", "claim_stp", "claim_hitl",
            "ocr_only_processing", "llm_escalation",
            "claim_hitl_upper_confidence", "wrong_crop_recall", "segment_claim_hitl", "p95_latency",
            "measured_cost", "cost_ceiling", "runtime_parity", "route_governance",
        }
        shadow_ready = all(gates[name] for name in shadow_gate_names)
        production_ready = all(gates.values())
        if safety_reject:
            decision = ReadinessDecision.REJECT
        elif missing_holdout:
            decision = ReadinessDecision.NEEDS_MORE_DATA
        elif production_ready:
            decision = ReadinessDecision.PROMOTE_TO_PRODUCTION
        elif shadow_ready:
            decision = ReadinessDecision.PROMOTE_TO_SHADOW
        else:
            decision = ReadinessDecision.NEEDS_MORE_DATA
        evidence_status = {
            name: "PASS" if passed else "NOT_RUN_OR_FAILED"
            for name, passed in gates.items()
        }
        return ReadinessResult(
            decision=decision,
            policy_version=self.config["version"],
            gates=gates,
            blocking_reasons=[name.upper() for name, passed in gates.items() if not passed],
            evidence_status=evidence_status,
        )
