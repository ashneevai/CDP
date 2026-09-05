"""Field-level OCR request/candidate types used by every recognition engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from PIL import Image

from packages.domain.common import BoundingBox
from packages.domain.enums import ClaimFormType, FieldCriticality
from packages.ocr.provenance import EvidenceProvenance

if TYPE_CHECKING:
    from packages.field_localization.contracts import FieldLocationEvidence


@dataclass(frozen=True)
class OCRRequest:
    document_id: str
    page_number: int
    field_name: str
    field_type: str
    form_type: ClaimFormType
    image: Image.Image
    bounding_box: BoundingBox
    expected_pattern: str | None = None
    allowed_characters: str | None = None
    handwritten_probability: float = 0.0
    criticality: FieldCriticality = FieldCriticality.NON_CRITICAL
    scope: Literal["FIELD_CROP", "REGION_CROP", "FULL_PAGE"] = "FIELD_CROP"
    registration_failed: bool = False
    policy_allows_full_page: bool = False
    preprocessing_profile: str | None = None
    document_sha256: str | None = None
    page_sha256: str | None = None
    source_representation_id: str | None = None
    localization_evidence: FieldLocationEvidence | None = None


@dataclass(frozen=True)
class OCRToken:
    """One recognized token with its own confidence and page-space geometry."""

    text: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class OCRCandidate:
    value: str | None
    raw_value: str
    engine: str
    model_name: str
    model_version: str
    preprocessing_variant: str
    raw_confidence: float
    calibrated_confidence: float | None
    bounding_box: BoundingBox
    latency_ms: float
    validation_results: tuple[str, ...] = field(default_factory=tuple)
    evidence_reference: str | None = None
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    preprocessing_version: str = "unknown"
    registration_confidence: float | None = None
    image_quality_score: float | None = None
    provenance: EvidenceProvenance | None = None
    tokens: tuple[OCRToken, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OCRResult:
    candidates: tuple[OCRCandidate, ...]
    provider: str
    provider_version: str
    latency_ms: float
    execution_cache_key: str | None = None
    cache_hit: bool = False


class OCRProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    async def extract(self, request: OCRRequest) -> OCRResult: ...


class OCREngine(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def recognize(self, request: OCRRequest) -> list[OCRCandidate]: ...
