from __future__ import annotations

from pydantic import Field

from packages.domain.common import DomainModel


class HITLFieldException(DomainModel):
    document_id: str
    claim_id: str | None = None
    field_name: str
    suggested_value: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    page_number: int
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    reference_status: str | None = None
    source_crop_ref: str | None = None


class HITLClaimExceptionQueue(DomainModel):
    claim_id: str
    document_family: str
    exceptions: list[HITLFieldException]
    runtime_manifest_id: str
    blocking_field_count: int
    total_field_count: int
