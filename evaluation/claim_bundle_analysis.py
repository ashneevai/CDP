"""Deterministic claim-blocker and bundle Pareto analysis.

This module is diagnostic only.  Truth is used after decisions are frozen and
never grants runtime acceptance authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

BUNDLES = {
    "IDENTITY": frozenset({
        "member_id", "patient_name", "patient_dob", "insured_name", "relationship",
    }),
    "PROVIDER": frozenset({"provider_name", "provider_npi", "federal_tax_no"}),
    "FINANCIAL": frozenset({
        "service_date", "procedure_code", "cpt_hcpcs", "units", "line_item_charge",
        "total_charge",
    }),
    "CLINICAL": frozenset({
        "diagnosis", "principal_diagnosis", "procedure_code", "cpt_hcpcs",
        "service_date", "type_of_bill",
    }),
}


def failure_category(field: Mapping) -> str:
    """Map persisted extraction/evidence outcomes to the governed taxonomy."""
    layer = str(field.get("failure_layer") or "").upper()
    reasons = {str(item).upper() for item in field.get("reason_codes", ())}
    missing = {str(item).upper() for item in field.get("missing_evidence", ())}
    if any("WRONG_CROP" in item for item in reasons):
        return "WRONG_CROP"
    if not field.get("selected_value") and not field.get("exact"):
        return "MISSING_CROP"
    if "FIELD_LOCALIZATION" in layer or "LOCALIZATION" in layer:
        return "WRONG_CROP"
    if "TABLE" in layer:
        return "TABLE_RECONSTRUCTION_ERROR"
    if "NORMALIZATION" in layer or "PARSER" in layer:
        return "NORMALIZATION_ERROR"
    if "SEGMENT" in layer:
        return "OCR_SEGMENTATION_ERROR"
    if "OCR" in layer:
        return "OCR_CHARACTER_ERROR"
    if any("REFERENCE" in item for item in reasons):
        return "REFERENCE_DATA_MISMATCH"
    if any("CONFLICT" in item or "CONTRADICTION" in item for item in reasons):
        return "CROSS_FIELD_INCONSISTENCY"
    if any("HANDWRIT" in item for item in reasons):
        return "UNSUPPORTED_HANDWRITING"
    if missing:
        return "MISSING_INDEPENDENT_EVIDENCE"
    return "UNCLASSIFIED"


def _bundle_names(blockers: set[str]) -> list[str]:
    return sorted(name for name, fields in BUNDLES.items() if blockers & fields)


def build_claim_blocker_analysis(
    claims: Iterable[Mapping], fields: Iterable[Mapping], *, effort: Mapping[str, int] | None = None,
) -> dict:
    """Return claim matrix and bundle Pareto without estimating false unlocks."""
    effort = effort or {name: index + 1 for index, name in enumerate(BUNDLES)}
    field_index = {
        (str(row["document_id"]), str(row["field_name"])): dict(row) for row in fields
    }
    matrix = []
    combinations: Counter[tuple[str, ...]] = Counter()
    correct_combinations: Counter[tuple[str, ...]] = Counter()
    incorrect_combinations: Counter[tuple[str, ...]] = Counter()
    blocker_distribution: Counter[int] = Counter()
    bundle_claims: Counter[str] = Counter()
    bundle_safe_unlocks: Counter[str] = Counter()

    for claim in claims:
        claim_id = str(claim["claim_id"])
        blockers = tuple(sorted(set(claim.get("blocking_unresolved_fields") or ())))
        blocker_distribution[len(blockers)] += 1
        combinations[blockers] += 1
        details = []
        for name in blockers:
            field = field_index[(claim_id, name)]
            details.append({
                "field_name": name,
                "criticality": field.get("criticality"),
                "critical": bool(field.get("critical")),
                "failure_category": failure_category(field),
                "extracted_value_correct": bool(field.get("exact")),
                "disposition": field.get("disposition"),
                "reason_codes": list(field.get("reason_codes") or ()),
                "missing_evidence": list(field.get("missing_evidence") or ()),
                "resolving_field_unlocks_claim": len(blockers) == 1,
            })
        if details and all(item["extracted_value_correct"] for item in details):
            correct_combinations[blockers] += 1
        if any(not item["extracted_value_correct"] for item in details):
            incorrect_combinations[blockers] += 1
        blocker_set = set(blockers)
        names = _bundle_names(blocker_set)
        for bundle in names:
            bundle_claims[bundle] += 1
            # A bundle can unlock only when it contains every blocker and every
            # persisted value is correct.  This is an opportunity, not authority.
            if blocker_set <= BUNDLES[bundle] and all(
                item["extracted_value_correct"] for item in details
            ):
                bundle_safe_unlocks[bundle] += 1
        matrix.append({
            "claim_id": claim_id,
            "document_family": claim.get("family"),
            "source": claim.get("source"),
            "quality_segment": claim.get("quality_segment", "UNKNOWN_NOT_CAPTURED"),
            "blocker_count": len(blockers),
            "blocker_fields": list(blockers),
            "bundles": names,
            "fields": details,
            "current_disposition": claim.get("disposition"),
            "claim_reason_codes": list(claim.get("reason_codes") or ()),
        })

    def ranked(counter: Counter[tuple[str, ...]]) -> list[dict]:
        return [
            {"fields": list(fields), "claims": count}
            for fields, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    pareto = []
    for name in BUNDLES:
        unlocks = bundle_safe_unlocks[name]
        unit_effort = max(1, int(effort.get(name, 1)))
        pareto.append({
            "bundle": name,
            "claims_containing_bundle_fields": bundle_claims[name],
            "complete_claim_opportunities": unlocks,
            "implementation_effort": unit_effort,
            "safe_claims_per_effort": unlocks / unit_effort,
            "production_authority": False,
        })
    pareto.sort(key=lambda row: (-row["safe_claims_per_effort"], row["bundle"]))
    return {
        "claim_blocker_matrix": matrix,
        "blocker_count_distribution": dict(sorted(blocker_distribution.items())),
        "top_blocker_combinations": ranked(combinations),
        "correct_but_reviewed_combinations": ranked(correct_combinations),
        "incorrect_extraction_combinations": ranked(incorrect_combinations),
        "bundle_pareto": pareto,
    }
