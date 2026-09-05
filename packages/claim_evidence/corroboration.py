"""Deterministic, provenance-preserving claim corroboration.

This module deliberately distinguishes validation (E4) and cross-field evidence
(E6) from independent OCR evidence (E2).  Validation never manufactures an OCR
observation and two candidates derived from one crop never become independent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from packages.claim_evidence.independence import observations_are_independent
from packages.evidence.normalization import normalize_agreement_value


@dataclass(frozen=True)
class Corroboration:
    evidence_class: str
    rule_id: str
    supported_fields: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    source_crop_sha256s: tuple[str, ...]


def deterministic_validation_evidence(field: str, row: dict[str, Any]) -> Corroboration | None:
    validation = row.get("deterministic_validation") or {}
    candidates = row.get("candidates") or []
    provenance = [candidate.get("provenance") or {} for candidate in candidates]
    source_ids = tuple(dict.fromkeys(p.get("source_candidate_id") for p in provenance if p.get("source_candidate_id")))
    crops = tuple(dict.fromkeys(p.get("crop_sha256") for p in provenance if p.get("crop_sha256")))
    if not validation.get("passed") or not row.get("final_value") or not source_ids or not crops:
        return None
    return Corroboration("E4", "FROZEN_DETERMINISTIC_VALIDATION", (field,), source_ids, crops)


def independent_ocr_evidence(field: str, row: dict[str, Any]) -> Corroboration | None:
    candidates = row.get("candidates") or []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if not observations_are_independent(left.get("provenance") or {}, right.get("provenance") or {}):
                continue
            if normalize_agreement_value(field, left.get("value")) != normalize_agreement_value(field, right.get("value")):
                continue
            if not left.get("value"):
                continue
            provenances = (left["provenance"], right["provenance"])
            return Corroboration(
                "E2", "DISTINCT_OBSERVATION_AGREEMENT", (field,),
                tuple(p["source_candidate_id"] for p in provenances),
                tuple(p["crop_sha256"] for p in provenances),
            )
    return None


def self_identity_evidence(rows: Iterable[dict[str, Any]]) -> Corroboration | None:
    by_field = {row["field_name"]: row for row in rows}
    patient, insured, relationship = (by_field.get(name) for name in ("patient_name", "insured_name", "relationship"))
    if not patient or not insured or not relationship:
        return None
    if str(relationship.get("final_value") or "").strip().upper() != "SELF":
        return None
    if normalize_agreement_value("patient_name", patient.get("final_value")) != normalize_agreement_value("patient_name", insured.get("final_value")):
        return None
    left = _first_provenance(patient)
    right = _first_provenance(insured)
    if not left or not right or not observations_are_independent(left, right):
        return None
    return Corroboration(
        "E6", "SELF_PATIENT_SUBSCRIBER_IDENTITY", ("patient_name", "insured_name", "relationship"),
        (left["source_candidate_id"], right["source_candidate_id"]),
        (left["crop_sha256"], right["crop_sha256"]),
    )


def _first_provenance(row: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in row.get("candidates") or []:
        provenance = candidate.get("provenance") or {}
        if provenance.get("source_candidate_id") and provenance.get("crop_sha256"):
            return provenance
    return None
