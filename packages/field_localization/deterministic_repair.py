"""Fail-closed template-zone repair for unresolved standard-form fields."""

from __future__ import annotations

from dataclasses import dataclass

from packages.domain.common import BoundingBox
from packages.ocr.token_reconstruction import SpatialToken


@dataclass(frozen=True)
class LocalizationRepair:
    outcome: str
    bounding_box: BoundingBox | None
    tokens: tuple[SpatialToken, ...]
    score: float
    reason_codes: tuple[str, ...]


def repair_from_expected_zone(
    tokens: list[SpatialToken] | tuple[SpatialToken, ...],
    expected_zone: BoundingBox,
    *,
    minimum_confidence: float = .80,
) -> LocalizationRepair:
    """Recover only an unambiguous token row wholly centered in a registered zone."""
    selected = tuple(token for token in tokens if (
        expected_zone.x0 <= (token.bounding_box.x0 + token.bounding_box.x1) / 2 <= expected_zone.x1
        and expected_zone.y0 <= (token.bounding_box.y0 + token.bounding_box.y1) / 2 <= expected_zone.y1
        and token.confidence >= minimum_confidence
    ))
    if not selected:
        return LocalizationRepair("EMPTY_CROP", None, (), 0.0, ("EXPECTED_ZONE_EMPTY",))
    centers = [(token.bounding_box.y0 + token.bounding_box.y1) / 2 for token in selected]
    heights = [token.bounding_box.y1 - token.bounding_box.y0 for token in selected]
    if max(centers) - min(centers) > max(8.0, max(heights, default=0) * .75):
        return LocalizationRepair(
            "LOCALIZATION_UNCERTAIN", None, selected, 0.0,
            ("MULTIPLE_COMPETING_ROWS", "FAIL_CLOSED"),
        )
    x0, y0 = min(t.bounding_box.x0 for t in selected), min(t.bounding_box.y0 for t in selected)
    x1, y1 = max(t.bounding_box.x1 for t in selected), max(t.bounding_box.y1 for t in selected)
    score = sum(token.confidence for token in selected) / len(selected)
    return LocalizationRepair(
        "CROP_SAFE",
        BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1,
                    image_width=expected_zone.image_width, image_height=expected_zone.image_height),
        selected, score,
        ("REGISTERED_EXPECTED_ZONE", "SINGLE_TOKEN_ROW", "VALUE_GEOMETRY_CONFIRMED"),
    )
