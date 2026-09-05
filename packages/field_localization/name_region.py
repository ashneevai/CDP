"""Deterministic standard-form name region refinement over OCR tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NAME_VALUE_ZONES = {
    "CMS1500": {
        "patient_name": (0.04, 0.10, 0.30, 0.13),
        "insured_name": (0.04, 0.16, 0.30, 0.19),
        "provider_name": (0.04, 0.238, 0.36, 0.27),
    },
    "UB04": {
        "provider_name": (0.04, 0.098, 0.45, 0.132),
        "patient_name": (0.04, 0.16, 0.32, 0.194),
    },
}
LABELS = {
    "PATIENTNAME",
    "PATIENTSNAME",
    "INSUREDNAME",
    "INSUREDSNAME",
    "BILLINGPROVIDER",
    "RENDERINGPROVIDER",
    "REFERRINGPROVIDER",
    "NAME",
}


@dataclass(frozen=True)
class NameRegionResult:
    value: str | None
    selected_tokens: tuple[dict[str, Any], ...]
    crop_box: tuple[float, float, float, float] | None
    crop_safety_outcome: str
    reason_codes: tuple[str, ...]
    score_components: dict[str, float]


def resolve_name_region(form: str, field: str, observation: dict[str, Any]) -> NameRegionResult:
    zone = NAME_VALUE_ZONES.get(form, {}).get(field)
    if not zone:
        return NameRegionResult(
            None, (), None, "LOCALIZATION_UNCERTAIN", ("NO_STANDARD_NAME_ZONE",), {}
        )
    w, h = observation["width"], observation["height"]
    box = (zone[0] * w, zone[1] * h, zone[2] * w, zone[3] * h)
    selected = []
    for token in observation.get("ocr_tokens", []):
        x0, y0, x1, y1 = token["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        normalized = "".join(ch for ch in token["text"].upper() if ch.isalnum())
        if (
            box[0] <= cx <= box[2]
            and box[1] <= cy <= box[3]
            and normalized not in LABELS
            and not normalized.isdigit()
            and not any(ch.isdigit() for ch in normalized[:3])
        ):
            selected.append(token)
    if not selected:
        return NameRegionResult(
            None, (), box, "EMPTY_CROP", ("NO_NAME_TOKEN_IN_VALUE_ZONE",), {"geometry": 1.0}
        )

    # A registered standard-form zone can overlap a neighbouring row. Select
    # one physical text line before horizontal assembly so adjacent observations
    # cannot be combined into fabricated evidence.
    expected_y = (box[1] + box[3]) / 2
    line_by_token = {
        id(token): round(((token["bbox"][1] + token["bbox"][3]) / 2) / 12) for token in selected
    }
    lines = set(line_by_token.values())
    best_line = min(
        lines,
        key=lambda line: abs(
            sum(
                (token["bbox"][1] + token["bbox"][3]) / 2
                for token in selected
                if line_by_token[id(token)] == line
            )
            / sum(1 for token in selected if line_by_token[id(token)] == line)
            - expected_y
        ),
    )
    selected = [token for token in selected if line_by_token[id(token)] == best_line]
    selected.sort(key=lambda token: (token["bbox"][0], token.get("reading_order", 0)))
    value = " ".join(t["text"].strip() for t in selected)
    confidence = sum(t["confidence"] for t in selected) / len(selected)
    return NameRegionResult(
        value,
        tuple(selected),
        box,
        "CROP_SAFE",
        ("STANDARD_ZONE_TOKEN_GEOMETRY", "LABEL_ZONE_EXCLUDED", "NEIGHBOR_TYPES_EXCLUDED"),
        {
            "geometry": 1.0,
            "token_confidence": confidence,
            "label_exclusion": 1.0,
            "neighbor_exclusion": 1.0,
        },
    )
