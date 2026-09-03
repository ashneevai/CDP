from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import Field

from packages.domain.common import DomainModel


class ProviderMode(StrEnum):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    DISABLED = "DISABLED"


class ProviderCapability(StrEnum):
    OCR = "OCR"
    STRUCTURED_EXTRACTION = "STRUCTURED_EXTRACTION"
    TABLE_EXTRACTION = "TABLE_EXTRACTION"
    MULTIMODAL = "MULTIMODAL"
    HANDWRITING = "HANDWRITING"


class ExtractionRequest(DomainModel):
    document_id: str
    page_id: str
    document_family: str
    field_names: list[str] = Field(default_factory=list)
    image_ref: str
    region: tuple[int, int, int, int] | None = None
    privacy_allows_cloud: bool = False
    latency_budget_ms: int | None = None
    cost_budget_usd: float | None = None


class ProviderCandidate(DomainModel):
    field_name: str
    value: str
    confidence: float = Field(ge=0, le=1)
    provider: str
    model_version: str
    lineage_id: str
    page_id: str
    bbox: tuple[float, float, float, float] | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    reason_codes: list[str] = Field(default_factory=list)


class ExtractionProvider(Protocol):
    name: str
    mode: ProviderMode
    capabilities: set[ProviderCapability]

    def available(self) -> bool: ...
    def extract(self, request: ExtractionRequest) -> list[ProviderCandidate]: ...
