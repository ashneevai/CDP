"""Deterministic token-to-ROI membership helpers for Phase 9D.

The production Phase 9B extractor currently assigns a page-observation token to
an ROI when the token centre lies inside that ROI. That rule is fast and stable,
but it can drop a value token whose bounding box meaningfully overlaps an ROI
while its centre falls just outside a resolved boundary.

Phase 9D introduces this helper first as a shadow-safe primitive. Production
selection should only adopt it after the recovered Phase 9 benchmark shows that
semantic accuracy improves without introducing label/value contamination.
"""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[int, int, int, int]
DEFAULT_MIN_TOKEN_OVERLAP = 0.40


@dataclass(frozen=True)
class TokenROIMembership:
    """Explain why a token is or is not assigned to an ROI."""

    accepted: bool
    center_inside: bool
    token_overlap_ratio: float
    reason: str


def _area(bbox: BBox) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def token_roi_overlap_ratio(token_bbox: BBox, roi_bbox: BBox) -> float:
    """Return the fraction of the token box covered by the ROI.

    The denominator is token area because the membership question is whether
    the observed token belongs to the field region. Degenerate token boxes fail
    closed with an overlap ratio of zero.
    """

    tx0, ty0, tx1, ty1 = token_bbox
    rx0, ry0, rx1, ry1 = roi_bbox
    token_area = _area(token_bbox)
    if token_area == 0:
        return 0.0

    ix0 = max(tx0, rx0)
    iy0 = max(ty0, ry0)
    ix1 = min(tx1, rx1)
    iy1 = min(ty1, ry1)
    intersection = _area((ix0, iy0, ix1, iy1))
    return intersection / token_area


def classify_token_roi_membership(
    token_bbox: BBox,
    roi_bbox: BBox,
    *,
    min_token_overlap: float = DEFAULT_MIN_TOKEN_OVERLAP,
) -> TokenROIMembership:
    """Classify token membership using centre containment plus overlap.

    Centre containment preserves Phase 9B behavior. An overlap fallback recovers
    boundary-clipped tokens. A threshold above 0.50 would be ineffective for the
    intended failure mode: once a token centre lies outside an axis-aligned ROI,
    less than half of that token can overlap on the clipped axis. The 0.40
    default is deliberately conservative and remains subject to benchmark
    qualification before production activation.
    """

    if not 0.0 <= min_token_overlap <= 1.0:
        raise ValueError("min_token_overlap must be between 0 and 1")

    tx0, ty0, tx1, ty1 = token_bbox
    rx0, ry0, rx1, ry1 = roi_bbox
    cx = (tx0 + tx1) / 2
    cy = (ty0 + ty1) / 2
    center_inside = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
    overlap_ratio = token_roi_overlap_ratio(token_bbox, roi_bbox)

    if center_inside:
        return TokenROIMembership(True, True, overlap_ratio, "TOKEN_CENTER_IN_ROI")
    if overlap_ratio >= min_token_overlap:
        return TokenROIMembership(True, False, overlap_ratio, "TOKEN_BOUNDARY_OVERLAP")
    return TokenROIMembership(False, False, overlap_ratio, "TOKEN_OUTSIDE_ROI")


def token_belongs_to_roi(
    token_bbox: BBox,
    roi_bbox: BBox,
    *,
    min_token_overlap: float = DEFAULT_MIN_TOKEN_OVERLAP,
) -> bool:
    """Boolean convenience wrapper for deterministic extraction code."""

    return classify_token_roi_membership(
        token_bbox,
        roi_bbox,
        min_token_overlap=min_token_overlap,
    ).accepted
