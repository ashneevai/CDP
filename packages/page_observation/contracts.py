from __future__ import annotations

from pydantic import Field

from packages.domain.common import DomainModel


class ObservationToken(DomainModel):
    token_id: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    line_index: int = Field(ge=0)
    reading_order: int = Field(ge=0)


class StructuralLine(DomainModel):
    line_id: str
    orientation: str
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)


class StructuralRegion(DomainModel):
    region_id: str
    kind: str
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)


class ImageQualityEvidence(DomainModel):
    blur_score: float = Field(ge=0)
    contrast_score: float = Field(ge=0)
    foreground_ratio: float = Field(ge=0, le=1)
    quality_bucket: str
    skew_degrees: float = 0.0
    noise_estimate: float = Field(default=0.0, ge=0, le=1)
    resolution_width: int | None = Field(default=None, gt=0)
    resolution_height: int | None = Field(default=None, gt=0)
    resolution_dpi: float | None = Field(default=None, gt=0)
    writing_type: str = "UNKNOWN"
    handwriting_likelihood: float | None = Field(default=None, ge=0, le=1)
    source_channel: str = "UNKNOWN"
    dynamic_range: float | None = Field(default=None, ge=0, le=255)
    background_uniformity: float | None = Field(default=None, ge=0, le=1)
    orientation_degrees: int | None = None
    binarization_quality: float | None = Field(default=None, ge=0, le=1)
    text_density: float | None = Field(default=None, ge=0, le=1)
    edge_density: float | None = Field(default=None, ge=0, le=1)


class PageObservation(DomainModel):
    page_id: str
    page_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspect_ratio: float = Field(gt=0)
    image_quality: ImageQualityEvidence
    ocr_tokens: tuple[ObservationToken, ...]
    text_lines: tuple[str, ...]
    word_boxes: tuple[tuple[float, float, float, float], ...]
    horizontal_lines: tuple[StructuralLine, ...]
    vertical_lines: tuple[StructuralLine, ...]
    connected_components: tuple[tuple[int, int, int, int], ...]
    checkbox_candidates: tuple[tuple[int, int, int, int], ...]
    table_regions: tuple[StructuralRegion, ...]
    anchor_candidates: tuple[str, ...]
    structural_regions: tuple[StructuralRegion, ...]
    document_family_evidence: dict[str, float] = Field(default_factory=dict)
    ocr_model_version: str
    preprocessing_version: str
    full_page_ocr_calls: int = 1
    observation_version: str = "page-observation-v2-quality-bands"
