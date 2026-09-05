"""Extraction jobs and field-level evidence/results."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from packages.domain.common import BoundingBox, DomainModel, ObjectRef, new_id, utcnow
from packages.domain.enums import ExtractionMethod, ValidationStatus
from packages.ocr.provenance import EvidenceProvenance


class FieldEvidence(DomainModel):
    """One candidate value for a field, produced by one extraction attempt.

    An ExtractedField may accumulate several of these across escalation
    (e.g. two OCR passes + a VLM candidate) before one is selected as the
    field's value; all candidates are retained for the review UI.
    """

    evidence_id: UUID = Field(default_factory=new_id)
    source: ExtractionMethod
    raw_text: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox | None = None
    crop_object: ObjectRef | None = None
    model_name: str | None = None
    model_version: str | None = None
    provenance: EvidenceProvenance | None = None
    tokens: tuple[dict[str, object], ...] = ()
    adjudication_metadata: dict[str, object] | None = None
    produced_at: datetime = Field(default_factory=utcnow)


class ExtractedField(DomainModel):
    """A single canonical field on a claim, with full evidence trail."""

    field_id: UUID = Field(default_factory=new_id)
    field_name: str
    raw_value: str
    normalized_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    page_number: int = Field(ge=1)
    bounding_box: BoundingBox
    crop_object_uri: str | None = None
    extraction_method: ExtractionMethod
    model_name: str | None = None
    model_version: str | None = None
    template_version: str | None = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_reasons: list[str] = Field(default_factory=list)
    candidates: list[FieldEvidence] = Field(default_factory=list)
    escalation_count: int = 0
    is_critical: bool = False
    disposition: str | None = None
    reference_evidence: dict | None = None


class ExtractionJob(DomainModel):
    job_id: UUID = Field(default_factory=new_id)
    document_id: UUID
    page_id: UUID
    claim_id: UUID | None = None
    template_id: str | None = None
    template_version: str | None = None
    attempt: int = Field(ge=1, default=1)
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    fields: list[ExtractedField] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
