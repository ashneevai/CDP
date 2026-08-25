from __future__ import annotations

import pytest

from packages.spatial_token_assignment import (
    classify_token_roi_membership,
    token_belongs_to_roi,
    token_roi_overlap_ratio,
)


def test_center_inside_preserves_phase9b_membership():
    result = classify_token_roi_membership((10, 10, 30, 20), (0, 0, 40, 40))

    assert result.accepted is True
    assert result.center_inside is True
    assert result.reason == "TOKEN_CENTER_IN_ROI"


def test_boundary_overlap_without_center_containment_is_recovered():
    # Centre x=20 lies outside ROI x<=19, but 45% of token area still belongs
    # to the resolved field region. Phase 9B's centre-only rule drops it.
    result = classify_token_roi_membership((10, 10, 30, 20), (0, 0, 19, 40))

    assert result.accepted is True
    assert result.center_inside is False
    assert result.token_overlap_ratio == pytest.approx(0.45)
    assert result.reason == "TOKEN_BOUNDARY_OVERLAP"


def test_vertical_boundary_overlap_without_center_is_recovered():
    # Centre y=20 lies outside ROI y<=19 with 45% token overlap.
    result = classify_token_roi_membership((10, 10, 20, 30), (0, 0, 40, 19))

    assert result.accepted is True
    assert result.center_inside is False
    assert result.token_overlap_ratio == pytest.approx(0.45)


def test_partial_touch_does_not_pull_neighboring_label_into_field():
    result = classify_token_roi_membership((10, 10, 30, 20), (28, 0, 40, 40))

    assert result.accepted is False
    assert result.center_inside is False
    assert result.token_overlap_ratio == pytest.approx(0.10)
    assert result.reason == "TOKEN_OUTSIDE_ROI"


def test_overlap_threshold_is_inclusive():
    # Centre x=5 lies outside ROI x<=4, while exactly 40% overlaps.
    assert token_belongs_to_roi(
        (0, 0, 10, 10),
        (0, 0, 4, 10),
        min_token_overlap=0.40,
    )


def test_overlap_below_threshold_fails_closed():
    assert not token_belongs_to_roi(
        (0, 0, 10, 10),
        (0, 0, 3, 10),
        min_token_overlap=0.40,
    )


def test_degenerate_token_fails_closed():
    assert token_roi_overlap_ratio((10, 10, 10, 20), (0, 0, 30, 30)) == 0.0
    assert not token_belongs_to_roi((10, 10, 10, 20), (0, 0, 30, 30))


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_overlap_threshold_is_rejected(threshold: float):
    with pytest.raises(ValueError, match="between 0 and 1"):
        classify_token_roi_membership(
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            min_token_overlap=threshold,
        )
