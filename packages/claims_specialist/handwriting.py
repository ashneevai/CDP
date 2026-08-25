"""Deterministic handwriting uncertainty policy.

The policy does not perform OCR or accept fields.  It decides whether a field
should request a handwriting-capable candidate provider in shadow/production
routing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HandwritingAssessment:
    suspected: bool
    score: float
    reason_codes: tuple[str, ...]
    request_multimodal_candidate: bool


def assess(
    *,
    ocr_confidence: float | None,
    candidate_disagreement: bool,
    region_quality: float | None = None,
    recognized_character_ratio: float | None = None,
    explicit_handwriting_signal: float | None = None,
) -> HandwritingAssessment:
    score = 0.0
    reasons: list[str] = []
    if explicit_handwriting_signal is not None:
        value = max(0.0, min(1.0, explicit_handwriting_signal))
        score += value * 0.55
        if value >= 0.55:
            reasons.append("HANDWRITING_VISUAL_SIGNAL")
    if ocr_confidence is not None and ocr_confidence < 0.70:
        score += 0.20
        reasons.append("LOW_OCR_CONFIDENCE")
    if candidate_disagreement:
        score += 0.15
        reasons.append("OCR_CANDIDATE_DISAGREEMENT")
    if region_quality is not None and region_quality < 0.55:
        score += 0.08
        reasons.append("LOW_REGION_QUALITY")
    if recognized_character_ratio is not None and recognized_character_ratio < 0.65:
        score += 0.12
        reasons.append("LOW_CHARACTER_RECOVERY")
    score = min(1.0, score)
    suspected = score >= 0.45
    return HandwritingAssessment(
        suspected=suspected,
        score=score,
        reason_codes=tuple(reasons or ("NO_HANDWRITING_SIGNAL",)),
        request_multimodal_candidate=score >= 0.60,
    )
