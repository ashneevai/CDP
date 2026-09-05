"""Declarative deterministic alternatives to universal E2 requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldPolicyDecision:
    accepted: bool
    evidence_used: tuple[str, ...]
    reason: str
    policy_version: str = "field-evidence-policy-phase9a-v1"


def evaluate_field_policy(field: str, criticality: str, row: dict[str, Any]) -> FieldPolicyDecision:
    location = row.get("localization_evidence") or {}
    validation = row.get("deterministic_validation") or {}
    safe = bool(
        location.get("confirmed")
        and location.get("positive_bounded_roi")
        and location.get("geometry_valid")
        and not row.get("wrong_crop_suspected")
    )
    base = bool(row.get("final_value") and validation.get("passed") and safe)
    evidence = ("E1_PRIMARY_OCR", "E3_FORMAT_VALIDATION", "E5_SAFE_LOCALIZATION")
    if field == "provider_npi" and base and "CHECKSUM_VALID" in validation.get("evidence", []):
        return FieldPolicyDecision(
            True, (*evidence, "E6_PROVIDER_SECTION_NPI_CHECKSUM"), "NPI_SAFE_CHECKSUM_POLICY"
        )
    if (
        field == "federal_tax_no"
        and base
        and "TAX_IDENTIFIER_SYNTAX_VALID" in validation.get("evidence", [])
    ):
        return FieldPolicyDecision(
            True, (*evidence, "E6_BILLING_PROVIDER_TAX_SYNTAX"), "TAX_ID_SAFE_SECTION_POLICY"
        )
    return FieldPolicyDecision(False, evidence, "INSUFFICIENT_FIELD_SPECIFIC_EVIDENCE")
