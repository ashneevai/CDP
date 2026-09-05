from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from pydantic import ConfigDict, Field, model_validator

from packages.domain.common import DomainModel


class CandidateSnapshot(DomainModel):
    """PHI-safe persisted identity for a candidate; raw values stay out of metrics/events."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    value_sha256: str
    engine: str
    model_version: str

    @classmethod
    def from_candidate(cls, candidate) -> CandidateSnapshot:
        from packages.evidence.builder import candidate_identifier

        return cls(
            candidate_id=candidate_identifier(candidate),
            value_sha256=sha256((candidate.value or "").encode()).hexdigest(),
            engine=candidate.engine,
            model_version=candidate.model_version,
        )


class ShadowObservation(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    field_name: str
    document_family: str
    route_id: str
    route_status: str
    production_candidate: CandidateSnapshot
    shadow_candidate: CandidateSnapshot | None = None
    agreement: bool | None = None
    runtime_latency_ms: float = Field(ge=0)
    additional_cpu_ms: float = Field(ge=0)
    additional_memory_bytes: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    execution_status: str
    truth_status: str = "UNAVAILABLE"
    truth_value_sha256: str | None = None
    shadow_correct: bool | None = None


class ShadowResult(DomainModel):
    canonical_candidate_id: str
    canonical_value: str | None
    canonical_unchanged: bool
    observation: ShadowObservation


class ClaimShadowObservation(DomainModel):
    """One adjudicated real-source claim evaluated without serving authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str
    source_group_id: str
    source_segment: str
    production_requires_review: bool
    shadow_requires_review: bool
    evaluated_field_decisions: int = Field(ge=0)
    correct_field_decisions: int = Field(ge=0)
    evaluated_critical_field_decisions: int = Field(ge=0)
    correct_critical_field_decisions: int = Field(ge=0)
    accepted_field_decisions: int = Field(ge=0)
    accepted_critical_field_decisions: int = Field(ge=0)
    correct_accepted_field_decisions: int = Field(ge=0)
    correct_accepted_critical_field_decisions: int = Field(ge=0)
    false_accepts: int = Field(ge=0)
    critical_false_accepts: int = Field(ge=0)
    wrong_crops: int = Field(default=0, ge=0)
    wrong_crops_detected: int = Field(default=0, ge=0)
    runtime_latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    runtime_decision_parity: bool
    route_governance_passed: bool
    llm_escalated: bool = False
    locked_holdout: bool = True
    shadow_only: bool = True

    @model_validator(mode="after")
    def validate_counts(self) -> ClaimShadowObservation:
        if self.correct_field_decisions > self.evaluated_field_decisions:
            raise ValueError("correct decisions exceed evaluated decisions")
        if self.evaluated_critical_field_decisions > self.evaluated_field_decisions:
            raise ValueError("evaluated critical decisions exceed evaluated decisions")
        if self.correct_critical_field_decisions > self.evaluated_critical_field_decisions:
            raise ValueError("correct critical decisions exceed evaluated critical decisions")
        if self.accepted_field_decisions > self.evaluated_field_decisions:
            raise ValueError("accepted decisions exceed evaluated decisions")
        if self.accepted_critical_field_decisions > self.accepted_field_decisions:
            raise ValueError("accepted critical decisions exceed accepted decisions")
        if self.correct_accepted_field_decisions > self.accepted_field_decisions:
            raise ValueError("correct accepted decisions exceed accepted decisions")
        if (
            self.correct_accepted_critical_field_decisions
            > self.accepted_critical_field_decisions
        ):
            raise ValueError("correct critical decisions exceed accepted critical decisions")
        if self.false_accepts > self.accepted_field_decisions:
            raise ValueError("false accepts exceed accepted decisions")
        if self.critical_false_accepts > self.accepted_critical_field_decisions:
            raise ValueError("critical false accepts exceed accepted critical decisions")
        if self.wrong_crops_detected > self.wrong_crops:
            raise ValueError("wrong crops detected exceeds wrong crops")
        if self.false_accepts != (
            self.accepted_field_decisions - self.correct_accepted_field_decisions
        ):
            raise ValueError("false accepts do not reconcile with accepted decisions")
        if self.critical_false_accepts != (
            self.accepted_critical_field_decisions
            - self.correct_accepted_critical_field_decisions
        ):
            raise ValueError(
                "critical false accepts do not reconcile with accepted critical decisions"
            )
        return self
