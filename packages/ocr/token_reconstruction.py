"""Geometry-aware field token reconstruction without identity correction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from packages.domain.common import BoundingBox

NAME_FIELDS = frozenset({
    "patient_name", "insured_name", "provider_name", "rendering_provider_name",
    "referring_provider_name", "billing_provider_name",
})
KNOWN_LABELS = frozenset({
    "PATIENTNAME", "INSUREDNAME", "PROVIDERNAME", "RENDERINGPROVIDER",
    "REFERRINGPROVIDER", "BILLINGPROVIDER", "FACILITYPROVIDER",
})


@dataclass(frozen=True)
class SpatialToken:
    text: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class TokenReconstruction:
    value: str | None
    selected_tokens: tuple[SpatialToken, ...]
    rejected_tokens: tuple[SpatialToken, ...]
    reason_codes: tuple[str, ...]


def _normalized(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def _name_token(token: SpatialToken) -> bool:
    normalized = _normalized(token.text)
    return bool(normalized) and normalized not in KNOWN_LABELS and not any(char.isdigit() for char in token.text)


def reconstruct_field_tokens(
    field_name: str,
    tokens: list[SpatialToken] | tuple[SpatialToken, ...],
    *,
    region: BoundingBox | None = None,
) -> TokenReconstruction:
    """Select observed tokens in reading order; never invent or fuzzy-correct text."""
    inside, rejected = [], []
    for token in tokens:
        box = token.bounding_box
        center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        in_region = region is None or (
            region.x0 <= center[0] <= region.x1 and region.y0 <= center[1] <= region.y1
        )
        valid = _name_token(token) if field_name in NAME_FIELDS else bool(token.text.strip())
        (inside if in_region and valid else rejected).append(token)
    if not inside:
        return TokenReconstruction(None, (), tuple(rejected), ("NO_FIELD_TOKENS",))
    heights = sorted(max(1.0, item.bounding_box.y1 - item.bounding_box.y0) for item in inside)
    tolerance = max(4.0, heights[len(heights) // 2] * .65)
    rows: list[list[SpatialToken]] = []
    for token in sorted(inside, key=lambda item: (
        (item.bounding_box.y0 + item.bounding_box.y1) / 2, item.bounding_box.x0
    )):
        cy = (token.bounding_box.y0 + token.bounding_box.y1) / 2
        if not rows:
            rows.append([token])
            continue
        previous_y = sum((x.bounding_box.y0 + x.bounding_box.y1) / 2 for x in rows[-1]) / len(rows[-1])
        if abs(cy - previous_y) <= tolerance:
            rows[-1].append(token)
        else:
            rows.append([token])
    # A bounded field is expected to contain one value row. Ambiguity fails closed.
    usable = [row for row in rows if any(_normalized(token.text) for token in row)]
    if len(usable) != 1:
        return TokenReconstruction(None, (), tuple(rejected + inside), ("MULTIPLE_TOKEN_ROWS",))
    selected = tuple(sorted(usable[0], key=lambda token: token.bounding_box.x0))
    value = " ".join(token.text.strip() for token in selected if token.text.strip()).strip()
    return TokenReconstruction(
        value or None, selected, tuple(rejected),
        ("GEOMETRY_READING_ORDER", "KNOWN_LABELS_REMOVED", "OBSERVED_CHARACTERS_ONLY"),
    )
