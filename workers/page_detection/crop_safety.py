"""Fail-closed geometry checks and bounded crop expansion for template fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

from packages.domain.registration import RegistrationEvidence
from packages.templates.models import FieldRegion
from workers.page_detection.local_crop_alignment import align_field_crop


class CropSafetyOutcome(StrEnum):
    CROP_SAFE = "CROP_SAFE"
    CROP_CLIPPED = "CROP_CLIPPED"
    WRONG_CROP_SUSPECTED = "WRONG_CROP_SUSPECTED"
    LABEL_CONTAMINATED = "LABEL_CONTAMINATED"
    EMPTY_CROP = "EMPTY_CROP"
    LOCALIZATION_UNCERTAIN = "LOCALIZATION_UNCERTAIN"


@dataclass(frozen=True)
class CropSafetyEvidence:
    accepted: bool
    reason_codes: tuple[str, ...]
    local_match_score: float
    local_alignment_accepted: bool
    crop_box: tuple[int, int, int, int]
    variant_boxes: tuple[tuple[int, int, int, int], ...]
    outcome: CropSafetyOutcome


def expanded_crop_boxes(
    region: FieldRegion,
    image_size: tuple[int, int],
    registration_confidence: float,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return at most 3 variants, and only in the 0.60 <= confidence < 0.80 band."""
    width, height = image_size
    ratios = (0.0, 0.05, 0.10) if 0.60 <= registration_confidence < 0.80 else (0.0,)
    boxes = []
    for ratio in ratios:
        dx = round((region.x1 - region.x0) * ratio)
        dy = round((region.y1 - region.y0) * ratio)
        boxes.append(
            (
                max(0, region.x0 - dx),
                max(0, region.y0 - dy),
                min(width, region.x1 + dx),
                min(height, region.y1 + dy),
            )
        )
    return tuple(boxes)


def validate_field_crop(
    aligned_page: Image.Image,
    reference_page: Image.Image,
    region: FieldRegion,
    registration: RegistrationEvidence | None,
    *,
    critical: bool,
) -> CropSafetyEvidence:
    reasons: list[str] = []
    width, height = aligned_page.size
    geometry_valid = (
        0 <= region.x0 < region.x1 <= width and 0 <= region.y0 < region.y1 <= height
    )
    if not geometry_valid:
        reasons.extend(("FIELD_GEOMETRY_INVALID", "CROP_CLIPPED"))
    confidence = registration.alignment_confidence if registration is not None else 0.0
    if registration is None or not registration.accepted or confidence < 0.60:
        reasons.append("LOW_REGISTRATION_CONFIDENCE")
    if registration is not None and registration.corner_validity is False:
        reasons.append("INVALID_REGISTRATION_CORNERS")

    local = align_field_crop(aligned_page, reference_page, region) if geometry_valid else None
    if local is None or not local.accepted:
        reasons.append("EXPECTED_LABEL_OR_NEIGHBOR_ANCHOR_MISMATCH")
    if critical and (local is None or local.match_score < 0.35):
        reasons.append("WRONG_CROP_SUSPECTED")
    empty = False
    if geometry_valid:
        crop = aligned_page.crop((region.x0, region.y0, region.x1, region.y1)).convert("L")
        extrema = crop.getextrema()
        empty = extrema is None or extrema[1] - extrema[0] < 8
        if empty:
            reasons.append("EMPTY_CROP")
    variants = expanded_crop_boxes(region, aligned_page.size, confidence)
    outcome = (
        CropSafetyOutcome.CROP_CLIPPED if not geometry_valid
        else CropSafetyOutcome.EMPTY_CROP if empty
        else CropSafetyOutcome.WRONG_CROP_SUSPECTED if "WRONG_CROP_SUSPECTED" in reasons
        else CropSafetyOutcome.LOCALIZATION_UNCERTAIN if reasons
        else CropSafetyOutcome.CROP_SAFE
    )
    return CropSafetyEvidence(
        accepted=not reasons,
        reason_codes=tuple(reasons),
        local_match_score=local.match_score if local is not None else 0.0,
        local_alignment_accepted=local.accepted if local is not None else False,
        crop_box=local.box if local is not None else variants[0],
        variant_boxes=variants,
        outcome=outcome,
    )
