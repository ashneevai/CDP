from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

from packages.reference_enrichment.code_matching import match_code
from packages.reference_enrichment.contracts import (
    ReferenceDecision,
    ReferenceLookupRequest,
    ReferenceRecord,
    ReferenceResolution,
    SourceTier,
)
from packages.reference_enrichment.lineage import lineage_violations
from packages.reference_enrichment.member_matching import _norm, match_member
from packages.reference_enrichment.payer_matching import match_payer
from packages.reference_enrichment.provider_matching import match_provider

_CODE_FIELDS = {
    "principal_diagnosis",
    "diagnosis_code",
    "cpt",
    "cpt_hcpcs",
    "hcpcs",
    "place_of_service",
}
_PAYER_FIELDS = {"payer_id", "payer_name"}


def _timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def normalize_reference(field: str, value: str | None) -> tuple[str | None, list[str]]:
    if value is None or not value.strip() or value.strip().upper() == "SAME":
        return None, ["MISSING_OR_UNAPPROVED_SAME_VALUE"]
    raw = value.strip()
    if field.endswith("_state"):
        normalized = raw.upper()
        return (
            (normalized, []) if re.fullmatch(r"[A-Z]{2}", normalized) else (None, ["INVALID_STATE"])
        )
    if field.endswith("_zip"):
        normalized = re.sub(r"[^0-9-]", "", raw)
        return (
            (normalized, [])
            if re.fullmatch(r"\d{5}(-?\d{4})?", normalized)
            else (None, ["INVALID_ZIP"])
        )
    return _norm(raw), []


def pending(
    request: ReferenceLookupRequest, reason: str, *, decision: str = "PENDING"
) -> ReferenceDecision:
    return ReferenceDecision(
        identity_key=request.identity_key,
        current_candidate=request.current_candidate,
        decision=decision,
        source_tier=SourceTier.UNVERIFIED,
        label_status=reason,
        policy_version=request.policy_version,
        system_decision_id=str(uuid4()),
        decision_reason=reason,
        created_at=datetime.now(UTC),
    )


def decide(
    request: ReferenceLookupRequest, records: list[ReferenceRecord], *, test_only: bool = False
) -> ReferenceDecision:
    if not records:
        return pending(
            request, "AWAITING_AUTHORIZED_REFERENCE_SOURCE", decision="REFERENCE_NOT_FOUND"
        )
    if len(records) != 1:
        return pending(
            request, "COMPETING_REFERENCE_RECORDS", decision="MULTIPLE_REFERENCE_MATCHES"
        )
    record = records[0]
    if not record.provider_authorized:
        return pending(request, "PROVIDER_UNAUTHORIZED", decision="PROVIDER_UNAUTHORIZED")
    violations = lineage_violations(record)
    if violations:
        return pending(request, ";".join(violations), decision="CIRCULAR_LINEAGE_REJECTED")
    matcher = (
        match_code
        if request.field_name in _CODE_FIELDS
        else match_payer
        if request.field_name in _PAYER_FIELDS
        else match_provider
        if "provider" in request.field_name or request.field_name == "npi"
        else match_member
    )
    matched, scores, contradictions, multi = matcher(request, record)
    value = record.field_values.get(request.field_name)
    normalized, validation = normalize_reference(request.field_name, value)
    contradictions.extend(validation)
    kind = record.provider_type.upper()
    downstream = kind in {"DOWNSTREAM", "FINALIZED_CLAIMS"}
    correction = kind in {"CORRECTION", "APPROVED_CORRECTION"}
    attributes = record.reference_attributes
    field_mapping = str(attributes.get("field_mapping_verified", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    claim_revalidated = str(attributes.get("claim_revalidated", "")).lower() in {"1", "true", "yes"}
    primary = attributes.get("primary_approved_by")
    second = attributes.get("second_approved_by")
    critical = request.criticality.upper() == "CRITICAL"
    correction_approved = bool(
        primary and attributes.get("primary_approved_at") and claim_revalidated
    )
    if critical:
        correction_approved = correction_approved and bool(
            second and second != primary and attributes.get("second_approved_at")
        )
    if downstream and (record.record_status.upper() != "FINALIZED" or not field_mapping):
        contradictions.append("DOWNSTREAM_NOT_FINALIZED_OR_MAPPING_UNVERIFIED")
    if correction and not correction_approved:
        contradictions.append("CORRECTION_APPROVAL_POLICY_FAILED")
    base_eligible = (
        multi
        and not contradictions
        and bool(normalized and record.dataset_version)
        and not test_only
    )
    eligible = (
        base_eligible
        and (not downstream or field_mapping)
        and (not correction or correction_approved)
    )
    source_tier = (
        SourceTier.TRAINING_ONLY
        if test_only
        else SourceTier.TIER_B_DOWNSTREAM
        if downstream
        else SourceTier.TIER_A_APPROVED_CORRECTION
        if correction
        else SourceTier.TIER_A_REFERENCE
    )
    verified_decision = (
        "DOWNSTREAM_VERIFIED"
        if downstream
        else "CORRECTION_VERIFIED"
        if correction
        else "REFERENCE_VERIFIED"
    )
    return ReferenceDecision(
        identity_key=request.identity_key,
        current_candidate=request.current_candidate,
        reference_value=value,
        normalized_reference_value=normalized,
        decision=verified_decision
        if eligible
        else "REFERENCE_CONTRADICTION"
        if contradictions
        else "INSUFFICIENT_MATCH_ATTRIBUTES",
        source_tier=source_tier,
        label_status="TEST_ONLY" if test_only else "VERIFIED" if eligible else "NOT_VERIFIED",
        reference_provider=record.provider_name,
        provider_authorized=record.provider_authorized,
        reference_dataset_version=record.dataset_version,
        source_record_id=record.source_record_id,
        snapshot_timestamp=record.snapshot_timestamp,
        snapshot_checksum=record.snapshot_checksum or record.response_hash,
        source_lineage=record.source_lineage,
        independent_truth=record.independent_truth,
        non_circular_lineage=record.non_circular_lineage,
        matching_attributes=matched,
        match_scores=scores,
        multi_attribute_match=multi,
        contradictions=contradictions,
        downstream_finalized=downstream and record.record_status.upper() == "FINALIZED",
        field_mapping_verified=field_mapping,
        approval_method=(
            "APPROVED_CORRECTION"
            if correction
            else "FINALIZED_DOWNSTREAM"
            if downstream
            else "AUTHORIZED_REFERENCE"
        )
        if eligible
        else None,
        second_approval_requirement=(
            "REQUIRED_CRITICAL_CORRECTION"
            if correction and critical
            else "NOT_REQUIRED_AUTH_REFERENCE"
        )
        if eligible
        else None,
        primary_approved_by=str(primary)
        if correction and primary
        else "reference-policy-engine"
        if eligible
        else None,
        primary_approved_at=_timestamp(attributes.get("primary_approved_at")),
        second_approved_by=str(second) if second else None,
        second_approved_at=_timestamp(attributes.get("second_approved_at")),
        claim_revalidated=claim_revalidated,
        evaluation_eligible=eligible,
        policy_version=request.policy_version,
        system_decision_id=str(uuid4()),
        decision_reason="policy passed" if eligible else "policy failed",
        created_at=datetime.now(UTC),
    )


def resolve(
    request: ReferenceLookupRequest,
    decision: ReferenceDecision,
    *,
    raw_value: str | None,
    normalized_value: str | None,
) -> ReferenceResolution:
    verified = decision.evaluation_eligible and decision.normalized_reference_value is not None
    scores = list(decision.match_scores.values())
    confidence = min(scores) if scores and verified else 0.0
    corrected = decision.normalized_reference_value if verified else None
    return ReferenceResolution(
        field_name=request.field_name,
        raw_value=raw_value,
        normalized_value=normalized_value,
        reference_candidate=decision.normalized_reference_value,
        corrected_value=corrected,
        final_value=corrected if verified else normalized_value,
        correction_reason=decision.decision_reason if verified else None,
        reference_source=decision.reference_provider,
        reference_version=decision.reference_dataset_version,
        matching_attributes=decision.matching_attributes,
        conflicting_attributes=decision.contradictions,
        reference_confidence=confidence,
        decision=decision,
    )
