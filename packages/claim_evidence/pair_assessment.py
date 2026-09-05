"""Strict pair classification for independently observed claim evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from packages.claim_evidence.independence import observations_are_independent
from packages.evidence.normalization import normalize_agreement_value

COMPATIBLE_SECTIONS = {
    "member_id": {frozenset(("subscriber", "claim_header"))},
    "provider_npi": {frozenset(("billing_provider", "service_facility"))},
    "federal_tax_no": {frozenset(("billing_provider", "tax_summary"))},
    "service_date": {frozenset(("service_line", "certification"))},
    "patient_name": {frozenset(("patient", "subscriber"))},
}


@dataclass(frozen=True)
class PairAssessment:
    genuinely_independent: bool
    semantically_compatible: bool
    agreeing: bool
    conflicting: bool
    different_page: bool
    different_document: bool
    rejection_reasons: tuple[str, ...]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def assess_evidence_pair(field_name: str, left: dict[str, Any], right: dict[str, Any]) -> PairAssessment:
    left_provenance = left.get("provenance") or {}
    right_provenance = right.get("provenance") or {}
    sections = frozenset((left.get("semantic_section"), right.get("semantic_section")))
    base_independent = observations_are_independent(left_provenance, right_provenance)
    sections_distinct = None not in sections and len(sections) == 2
    genuinely_independent = base_independent and sections_distinct
    compatible = genuinely_independent and sections in COMPATIBLE_SECTIONS.get(field_name, set())
    left_value = normalize_agreement_value(field_name, left.get("value"))
    right_value = normalize_agreement_value(field_name, right.get("value"))
    agreeing = bool(compatible and left_value and left_value == right_value)
    conflicting = bool(compatible and left_value and right_value and left_value != right_value)
    reasons = []
    if left_provenance.get("crop_sha256") == right_provenance.get("crop_sha256"):
        reasons.append("SAME_CROP_SHA256")
    if left_provenance.get("localization_region_id") == right_provenance.get("localization_region_id"):
        reasons.append("SAME_LOCALIZATION_REGION")
    if not sections_distinct:
        reasons.append("SEMANTIC_SECTION_NOT_DISTINCT")
    elif sections not in COMPATIBLE_SECTIONS.get(field_name, set()):
        reasons.append("SEMANTIC_SECTIONS_INCOMPATIBLE")
    if conflicting:
        reasons.append("MATERIAL_VALUE_CONFLICT")
    return PairAssessment(
        genuinely_independent=genuinely_independent,
        semantically_compatible=compatible,
        agreeing=agreeing,
        conflicting=conflicting,
        different_page=left_provenance.get("page_id") != right_provenance.get("page_id"),
        different_document=left_provenance.get("document_id") != right_provenance.get("document_id"),
        rejection_reasons=tuple(reasons),
    )
