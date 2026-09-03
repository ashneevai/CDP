from __future__ import annotations

from enum import StrEnum
from pydantic import Field

from packages.domain.common import DomainModel


class PageKind(StrEnum):
    CLAIM = "CLAIM"
    NON_CLAIM = "NON_CLAIM"
    UNKNOWN = "UNKNOWN"


class StructureKind(StrEnum):
    STRUCTURED = "STRUCTURED"
    UNSTRUCTURED = "UNSTRUCTURED"
    UNKNOWN = "UNKNOWN"


class DocumentFamily(StrEnum):
    CMS_1500 = "CMS_1500"
    UB_04 = "UB_04"
    EOB = "EOB"
    MEDICAL_BILL = "MEDICAL_BILL"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    LAB = "LAB"
    AUTHORIZATION = "AUTHORIZATION"
    REFERRAL = "REFERRAL"
    COVER_SHEET = "COVER_SHEET"
    CORRESPONDENCE = "CORRESPONDENCE"
    UNKNOWN_STRUCTURED = "UNKNOWN_STRUCTURED"
    UNKNOWN_UNSTRUCTURED = "UNKNOWN_UNSTRUCTURED"
    NON_CLAIM = "NON_CLAIM"
    UNKNOWN = "UNKNOWN"


class ClassificationSignal(DomainModel):
    source: str
    label: str
    confidence: float = Field(ge=0, le=1)
    lineage_id: str
    reason_codes: list[str] = Field(default_factory=list)


class PageClassificationV5(DomainModel):
    page_id: str
    page_kind: PageKind
    structure_kind: StructureKind
    family: DocumentFamily
    form_version: str | None = None
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(default=0, ge=0, le=1)
    signals: list[ClassificationSignal] = Field(default_factory=list)
    route: str
    requires_review: bool = False
    reason_codes: list[str] = Field(default_factory=list)
