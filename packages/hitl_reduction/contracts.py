from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import ConfigDict, Field, field_validator, model_validator

from packages.domain.common import DomainModel
from packages.evidence.models import EvidenceItem
from packages.evidence_decision import DecisionContext
from packages.production_readiness_gate import ApprovalRecord

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class OperationalEvidence(DomainModel):
    """Non-accuracy evidence required by the production readiness authority."""

    full_suite_passed: bool = False
    wrong_crop_recall: float | None = Field(default=None, ge=0, le=1)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    cost_per_document_usd: float | None = Field(default=None, ge=0)
    runtime_parity_passed: bool = False
    route_governance_passed: bool = False
    security_passed: bool = False
    database_and_events_passed: bool = False
    load_and_keda_passed: bool = False
    shadow_validation_passed: bool = False
    failure_injection_passed: bool = False
    route_shadow_samples: dict[str, int] = Field(default_factory=dict)
    route_operational_reliability: dict[str, float] = Field(default_factory=dict)
    route_cost_per_call_usd: dict[str, float] = Field(default_factory=dict)
    release_commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    evidence_bundle_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    approvals: list[ApprovalRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def route_metrics_are_bounded(self):
        if any(value < 0 for value in self.route_shadow_samples.values()):
            raise ValueError("ROUTE_SHADOW_SAMPLES_MUST_BE_NONNEGATIVE")
        if any(value < 0 for value in self.route_cost_per_call_usd.values()):
            raise ValueError("ROUTE_COST_MUST_BE_NONNEGATIVE")
        if any(value < 0 or value > 1 for value in self.route_operational_reliability.values()):
            raise ValueError("ROUTE_RELIABILITY_MUST_BE_BETWEEN_ZERO_AND_ONE")
        return self


class FieldRuntimeRecord(DomainModel):
    field_instance_id: str
    document_id: str
    page_id: str
    page_sha256: str = Field(pattern=SHA256_PATTERN)
    crop_sha256: str = Field(pattern=SHA256_PATTERN)
    crop_reference: str
    source_segment: str = "UNKNOWN_NOT_CAPTURED"
    quality_segment: str = "UNKNOWN_NOT_CAPTURED"
    latency_ms: float = Field(ge=0)
    decision_context: DecisionContext

    @model_validator(mode="after")
    def exact_observation_lineage(self):
        if self.decision_context.field_id != self.field_instance_id:
            raise ValueError("FIELD_INSTANCE_ID_CONTEXT_MISMATCH")
        for candidate in self.decision_context.candidates:
            provenance = candidate.provenance
            if provenance is None:
                raise ValueError("CANDIDATE_PROVENANCE_REQUIRED")
            if provenance.page_sha256 != self.page_sha256:
                raise ValueError("CANDIDATE_PAGE_SHA256_MISMATCH")
            if not provenance.crop_sha256:
                raise ValueError("CANDIDATE_CROP_SHA256_REQUIRED")
        if self.decision_context.candidates and not any(
            candidate.provenance and candidate.provenance.crop_sha256 == self.crop_sha256
            for candidate in self.decision_context.candidates
        ):
            raise ValueError("PRIMARY_CROP_NOT_REPRESENTED_BY_CANDIDATE")
        return self


class ClaimRuntimeRecord(DomainModel):
    claim_id: str
    document_family: str
    source_segment: str = "UNKNOWN_NOT_CAPTURED"
    quality_segment: str = "UNKNOWN_NOT_CAPTURED"
    fields: list[FieldRuntimeRecord] = Field(min_length=1)
    claim_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[EvidenceItem] = Field(default_factory=list)
    document_integrity_valid: bool = True
    template_integrity_valid: bool = True
    registration_integrity_valid: bool = True
    process_integrity_valid: bool = True
    structural_consistency_valid: bool = True
    dependent_field_groups: list[list[str]] = Field(default_factory=list)
    enforce_configured_required_fields: bool = True

    @model_validator(mode="after")
    def claim_lineage_is_consistent(self):
        if not self.fields:
            raise ValueError("CLAIM_REQUIRES_FIELD_RECORDS")
        field_ids = [field.field_instance_id for field in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("DUPLICATE_FIELD_INSTANCE_ID_IN_CLAIM")
        if any(
            field.decision_context.document_family != self.document_family for field in self.fields
        ):
            raise ValueError("FIELD_DOCUMENT_FAMILY_MISMATCH")
        if len({field.document_id for field in self.fields}) != 1:
            raise ValueError("CLAIM_FIELDS_MUST_SHARE_DOCUMENT_ID")
        return self


class HITLReductionInput(DomainModel):
    cohort_id: str
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    holdout_frozen: bool = False
    holdout_independent: bool = False
    claims: list[ClaimRuntimeRecord] = Field(min_length=1)
    operational_evidence: OperationalEvidence = Field(default_factory=OperationalEvidence)

    @model_validator(mode="after")
    def globally_unique_field_ids(self):
        field_ids = [field.field_instance_id for claim in self.claims for field in claim.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("DUPLICATE_FIELD_INSTANCE_ID_IN_COHORT")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("DUPLICATE_CLAIM_ID_IN_COHORT")
        return self


class LabelDisposition(StrEnum):
    VALUE = "VALUE"
    CONFIRMED_BLANK = "CONFIRMED_BLANK"
    UNREADABLE = "UNREADABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LabelAuthority(StrEnum):
    SOURCE_SYSTEM_GROUND_TRUTH = "SOURCE_SYSTEM_GROUND_TRUTH"
    HUMAN_ADJUDICATED = "HUMAN_ADJUDICATED"


class ReviewObservation(DomainModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    reviewed_at: datetime
    disposition: LabelDisposition
    value: str | None = None

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_id_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("REVIEWER_ID_REQUIRED")
        return normalized

    @model_validator(mode="after")
    def value_matches_disposition(self):
        if self.reviewed_at.tzinfo is None:
            raise ValueError("REVIEW_TIMESTAMP_MUST_BE_TIMEZONE_AWARE")
        if self.disposition is LabelDisposition.VALUE and not self.value:
            raise ValueError("VALUE_DISPOSITION_REQUIRES_VALUE")
        if self.disposition is LabelDisposition.CONFIRMED_BLANK and self.value:
            raise ValueError("CONFIRMED_BLANK_CANNOT_HAVE_VALUE")
        return self


class BlindReviewSubmission(ReviewObservation):
    """A review or adjudication bound to one sealed blind assignment."""

    blind_task_id: str = Field(pattern=SHA256_PATTERN)
    prediction_seal_sha256: str = Field(pattern=SHA256_PATTERN)
    review_assignment_seal_sha256: str = Field(pattern=SHA256_PATTERN)


class GovernedFieldLabel(DomainModel):
    """Truth record with exact sealed-observation binding."""

    model_config = ConfigDict(extra="forbid")

    field_instance_id: str
    document_id: str
    page_id: str
    page_sha256: str = Field(pattern=SHA256_PATTERN)
    crop_sha256: str = Field(pattern=SHA256_PATTERN)
    blind_task_id: str = Field(pattern=SHA256_PATTERN)
    prediction_seal_sha256: str = Field(pattern=SHA256_PATTERN)
    authority: LabelAuthority
    final_disposition: LabelDisposition
    final_value: str | None = None
    reviews: list[ReviewObservation] = Field(default_factory=list)
    adjudication: ReviewObservation | None = None
    source_system: str | None = None
    source_version: str | None = None
    source_snapshot_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_record_id: str | None = None
    source_independent: bool = False
    source_non_circular: bool = False
    derived_from_cdp: bool = False

    @model_validator(mode="after")
    def final_value_matches_disposition(self):
        if self.final_disposition is LabelDisposition.VALUE and not self.final_value:
            raise ValueError("FINAL_VALUE_REQUIRED")
        if self.final_disposition is LabelDisposition.CONFIRMED_BLANK and self.final_value:
            raise ValueError("CONFIRMED_BLANK_CANNOT_HAVE_FINAL_VALUE")
        return self
