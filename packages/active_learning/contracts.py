from __future__ import annotations

from enum import StrEnum
from pydantic import Field

from packages.domain.common import DomainModel


class CorrectionReason(StrEnum):
    ROUTING_ERROR = "ROUTING_ERROR"
    FORM_VERSION_ERROR = "FORM_VERSION_ERROR"
    OCR_ERROR = "OCR_ERROR"
    LOCALIZATION_ERROR = "LOCALIZATION_ERROR"
    TABLE_ERROR = "TABLE_ERROR"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    NORMALIZATION_ERROR = "NORMALIZATION_ERROR"
    REFERENCE_ERROR = "REFERENCE_ERROR"
    CROSS_FIELD_ERROR = "CROSS_FIELD_ERROR"
    CONFIDENCE_ERROR = "CONFIDENCE_ERROR"
    POLICY_TOO_CONSERVATIVE = "POLICY_TOO_CONSERVATIVE"
    TRUE_AMBIGUITY = "TRUE_AMBIGUITY"
    MISSING_SOURCE_DATA = "MISSING_SOURCE_DATA"


class CorrectionEvent(DomainModel):
    correction_id: str
    document_id: str
    field_name: str
    document_family: str
    before_value: str | None = None
    after_value: str | None = None
    reason: CorrectionReason
    runtime_manifest_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    reviewer_id_hash: str | None = None


class TrainingCandidate(DomainModel):
    target_component: str
    correction_ids: list[str]
    reason_counts: dict[str, int]
    requires_human_approval: bool = True
