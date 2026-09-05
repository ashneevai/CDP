"""Provenance-safe corroboration of repeated claim evidence."""

from __future__ import annotations

from typing import Any


def observations_are_independent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Reject shared OCR/crop dependencies before values can corroborate."""
    required = ("invocation_id", "crop_sha256", "localization_region_id")
    if any(not left.get(key) or not right.get(key) for key in required):
        return False
    if any(left[key] == right[key] for key in required):
        return False
    left_dependencies = set(left.get("shared_dependency_ids") or ())
    right_dependencies = set(right.get("shared_dependency_ids") or ())
    return not bool(left_dependencies & right_dependencies)


def has_independent_corroboration(provenances: list[dict[str, Any]]) -> bool:
    return any(
        observations_are_independent(left, right)
        for index, left in enumerate(provenances)
        for right in provenances[index + 1:]
    )
