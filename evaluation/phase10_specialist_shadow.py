"""Shadow-only specialist analysis for production-equivalent predictions.

This module attaches deterministic specialist diagnostics to a prediction but
never mutates the canonical field value or field/claim disposition.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from packages.claims_specialist.cms import policy_for
from packages.claims_specialist.fusion import FusionCandidate, fuse
from packages.claims_specialist.handwriting import assess


def _candidate_rows(field: dict[str, Any]) -> list[FusionCandidate]:
    rows: list[FusionCandidate] = []
    raw_candidates = field.get("candidates")
    if isinstance(raw_candidates, list):
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict) or raw.get("value") in (None, ""):
                continue
            rows.append(FusionCandidate(
                value=str(raw.get("value")),
                source=str(raw.get("source") or raw.get("engine") or "candidate"),
                confidence=float(raw.get("confidence") or raw.get("raw_confidence") or 0.0),
                lineage=str(raw.get("lineage") or raw.get("evidence_reference") or f"candidate:{index}"),
                spatial_confidence=(float(raw["spatial_confidence"]) if raw.get("spatial_confidence") is not None else None),
                handwriting=bool(raw.get("handwriting")),
            ))
    if not rows and field.get("value") not in (None, ""):
        decision = field.get("decision") if isinstance(field.get("decision"), dict) else {}
        rows.append(FusionCandidate(
            value=str(field.get("value")),
            source=str(field.get("source") or "canonical_runtime"),
            confidence=float(field.get("confidence") or 0.0),
            lineage=str(decision.get("evidence_reference") or "canonical_runtime"),
            spatial_confidence=(float(field["spatial_confidence"]) if field.get("spatial_confidence") is not None else None),
            handwriting=bool(field.get("handwriting")),
        ))
    return rows


def enrich_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = prediction.get("fields")
    if not isinstance(fields, dict):
        return {"enabled": True, "eligible": False, "reason": "NO_FIELDS", "fields": {}}
    shadow: dict[str, Any] = {}
    for field_name, field in fields.items():
        if not isinstance(field, dict):
            continue
        policy = policy_for(field_name)
        if policy is None:
            continue
        candidates = _candidate_rows(field)
        disagreement = len({policy.normalizer(row.value) for row in candidates}) > 1
        handwriting = assess(
            ocr_confidence=(float(field["confidence"]) if field.get("confidence") is not None else None),
            candidate_disagreement=disagreement,
            region_quality=(float(field["region_quality"]) if field.get("region_quality") is not None else None),
            recognized_character_ratio=(float(field["recognized_character_ratio"]) if field.get("recognized_character_ratio") is not None else None),
            explicit_handwriting_signal=(float(field["handwriting_score"]) if field.get("handwriting_score") is not None else None),
        )
        result = fuse(field_name, candidates)
        current_value = field.get("value")
        normalized_current = policy.normalizer(str(current_value or ""))
        shadow[field_name] = {
            "current_normalized": normalized_current,
            "current_valid": policy.validator(normalized_current),
            "fusion": asdict(result),
            "handwriting": asdict(handwriting),
            "candidate_changed": bool(result.value and result.value != normalized_current),
            "production_mutation_allowed": False,
        }
    return {
        "enabled": True,
        "eligible": bool(shadow),
        "document_id": prediction.get("document_id"),
        "route": prediction.get("route"),
        "fields": shadow,
        "production_mutation_allowed": False,
    }
