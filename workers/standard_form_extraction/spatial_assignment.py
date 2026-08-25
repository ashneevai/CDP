"""Deterministic, exclusive ownership of page-observation tokens."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from packages.page_observation import ObservationToken

BBox = tuple[float, float, float, float]


class TokenOwnership(StrEnum):
    UNASSIGNED = "UNASSIGNED"
    UNIQUE_FIELD = "UNIQUE_FIELD"
    AMBIGUOUS = "AMBIGUOUS"
    LABEL = "LABEL"
    VALUE = "VALUE"
    SHARED_STRUCTURAL = "SHARED_STRUCTURAL"


@dataclass(frozen=True)
class SpatialFeatures:
    centroid_contained: bool
    intersection_area: float
    token_overlap: float
    roi_overlap: float
    horizontal_overlap: float
    vertical_overlap: float
    baseline_distance: float
    nearest_neighbor_distance: float
    score: float


@dataclass(frozen=True)
class TokenAssignment:
    token: ObservationToken
    field_name: str | None
    ownership: TokenOwnership
    features: SpatialFeatures | None
    competing_fields: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _intersection(left: BBox, right: BBox) -> tuple[float, float, float]:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width, height, width * height


def spatial_features(token_bbox: BBox, roi: BBox, neighbors: tuple[BBox, ...]) -> SpatialFeatures:
    token_width = max(1.0, token_bbox[2] - token_bbox[0])
    token_height = max(1.0, token_bbox[3] - token_bbox[1])
    roi_width = max(1.0, roi[2] - roi[0])
    roi_height = max(1.0, roi[3] - roi[1])
    width, height, area = _intersection(token_bbox, roi)
    center_x = (token_bbox[0] + token_bbox[2]) / 2
    center_y = (token_bbox[1] + token_bbox[3]) / 2
    centroid = roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]
    token_overlap = area / (token_width * token_height)
    roi_overlap = area / (roi_width * roi_height)
    horizontal = width / token_width
    vertical = height / token_height
    baseline = abs(token_bbox[3] - roi[3]) / roi_height
    nearest = min(
        (
            math.hypot(
                center_x - (other[0] + other[2]) / 2,
                center_y - (other[1] + other[3]) / 2,
            )
            / max(roi_width, roi_height)
            for other in neighbors
        ),
        default=10.0,
    )
    # Fixed, auditable weights: overlap dominates; alignment breaks ties.
    score = (
        0.40 * token_overlap
        + 0.20 * horizontal
        + 0.15 * vertical
        + 0.10 * float(centroid)
        + 0.10 * max(0.0, 1.0 - baseline)
        + 0.05 * min(1.0, nearest)
    )
    return SpatialFeatures(
        centroid, area, token_overlap, roi_overlap, horizontal, vertical,
        baseline, nearest, score,
    )


def assign_tokens(
    tokens: tuple[ObservationToken, ...],
    rois: dict[str, BBox],
    *,
    labels_by_field: dict[str, set[str]] | None = None,
    minimum_score: float = 0.45,
    ambiguity_margin: float = 0.08,
) -> tuple[TokenAssignment, ...]:
    """Assign each token to at most one field, rejecting close conflicts."""
    labels_by_field = labels_by_field or {}
    normalized_labels = {
        field: {_normalize(label) for label in labels if _normalize(label)}
        for field, labels in labels_by_field.items()
    }
    assignments: list[TokenAssignment] = []
    for token in tokens:
        ranked: list[tuple[float, str, SpatialFeatures]] = []
        for field, roi in rois.items():
            neighbors = tuple(box for name, box in rois.items() if name != field)
            features = spatial_features(token.bbox, roi, neighbors)
            if features.intersection_area > 0:
                ranked.append((features.score, field, features))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or ranked[0][0] < minimum_score:
            assignments.append(TokenAssignment(token, None, TokenOwnership.UNASSIGNED, None))
            continue
        best_score, best_field, best = ranked[0]
        competitors = tuple(item[1] for item in ranked[1:] if best_score-item[0] < ambiguity_margin)
        if competitors:
            assignments.append(TokenAssignment(
                token, None, TokenOwnership.AMBIGUOUS, best,
                (best_field, *competitors),
            ))
            continue
        normalized = _normalize(token.text)
        is_label = any(
            normalized == label or (len(label) >= 4 and normalized.startswith(label))
            for label in normalized_labels.get(best_field, set())
        )
        assignments.append(TokenAssignment(
            token,
            best_field,
            TokenOwnership.LABEL if is_label else TokenOwnership.VALUE,
            best,
        ))
    return tuple(assignments)
