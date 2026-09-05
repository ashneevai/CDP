"""Claim-level qualification for non-authoritative real-source shadow traffic."""

from __future__ import annotations

from collections import Counter
from math import sqrt

from pydantic import ConfigDict

from packages.domain.common import DomainModel
from packages.shadow_evaluation.models import ClaimShadowObservation


class ShadowQualificationPolicy(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_claims: int = 1000
    minimum_accepted_critical_field_decisions: int = 3000
    maximum_claim_hitl: float = 0.08
    maximum_claim_hitl_upper_95: float = 0.10
    maximum_segment_claim_hitl: float = 0.15
    maximum_false_accept_rate: float = 0.001
    maximum_critical_false_accepts: int = 0
    minimum_overall_raw_accuracy: float = 0.95
    minimum_critical_raw_accuracy: float = 0.98
    minimum_accepted_precision: float = 0.995
    minimum_critical_accepted_precision: float = 0.995
    minimum_wrong_crop_recall: float = 0.95
    minimum_ocr_only_processing_rate: float = 0.99
    maximum_llm_escalation_rate: float = 0.01


class ShadowQualificationReport(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str
    promotion_authority: bool = False
    claim_count: int
    source_group_count: int
    claim_hitl: float | None
    claim_hitl_upper_95: float | None
    claim_stp: float | None
    maximum_segment_claim_hitl: float | None
    segment_claim_hitl: dict[str, float]
    evaluated_field_decisions: int
    overall_raw_accuracy: float | None
    critical_accuracy: float | None
    safe_field_coverage: float | None
    accepted_field_decisions: int
    accepted_critical_field_decisions: int
    accepted_precision: float | None
    false_accept_rate: float | None
    critical_false_accepts: int
    critical_accepted_precision: float | None
    wrong_crop_recall: float | None
    p95_latency_ms: float | None
    cost_per_document_usd: float | None
    ocr_only_processing_rate: float | None
    llm_escalation_rate: float | None
    gates: dict[str, bool]
    blocking_reasons: list[str]


def _wilson_upper(successes: int, total: int) -> float | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre + margin) / denominator


def qualify_shadow_claims(
    observations: list[ClaimShadowObservation],
    *,
    policy: ShadowQualificationPolicy | None = None,
    prohibited_source_groups: set[str] | None = None,
) -> ShadowQualificationReport:
    policy = policy or ShadowQualificationPolicy()
    prohibited = prohibited_source_groups or set()
    claim_ids = [row.claim_id for row in observations]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("duplicate claim_id in shadow traffic")
    overlap = {row.source_group_id for row in observations} & prohibited
    if overlap:
        raise ValueError("shadow traffic overlaps training/calibration source groups")
    if any(not row.shadow_only for row in observations):
        raise ValueError("shadow observation claims serving authority")

    claims = len(observations)
    hitl_count = sum(row.shadow_requires_review for row in observations)
    claim_hitl = hitl_count / claims if claims else None
    segments = Counter(row.source_segment for row in observations)
    segment_hitl_counts = Counter(
        row.source_segment for row in observations if row.shadow_requires_review
    )
    segment_hitl = {
        name: segment_hitl_counts[name] / count for name, count in sorted(segments.items())
    }
    maximum_segment = max(segment_hitl.values()) if segment_hitl else None
    accepted = sum(row.accepted_field_decisions for row in observations)
    evaluated = sum(row.evaluated_field_decisions for row in observations)
    correct = sum(row.correct_field_decisions for row in observations)
    evaluated_critical = sum(
        row.evaluated_critical_field_decisions for row in observations
    )
    correct_evaluated_critical = sum(
        row.correct_critical_field_decisions for row in observations
    )
    accepted_critical = sum(row.accepted_critical_field_decisions for row in observations)
    correct_accepted_critical = sum(
        row.correct_accepted_critical_field_decisions for row in observations
    )
    false_accepts = sum(row.false_accepts for row in observations)
    critical_false_accepts = sum(row.critical_false_accepts for row in observations)
    wrong_crops = sum(row.wrong_crops for row in observations)
    wrong_crops_detected = sum(row.wrong_crops_detected for row in observations)
    upper = _wilson_upper(hitl_count, claims)
    false_accept_rate = false_accepts / accepted if accepted else None
    accepted_precision = (
        sum(row.correct_accepted_field_decisions for row in observations) / accepted
        if accepted else None
    )
    critical_precision = (
        correct_accepted_critical / accepted_critical if accepted_critical else None
    )
    wrong_crop_recall = wrong_crops_detected / wrong_crops if wrong_crops else None
    latencies = sorted(row.runtime_latency_ms for row in observations)
    p95_latency = latencies[max(0, int(len(latencies) * .95 + .999999) - 1)] if latencies else None
    cost_per_document = (
        sum(row.cost_usd for row in observations) / claims if claims else None
    )
    llm_count = sum(row.llm_escalated for row in observations)
    llm_escalation_rate = llm_count / claims if claims else None
    ocr_only_processing_rate = (claims - llm_count) / claims if claims else None
    gates = {
        "locked_holdout": bool(observations) and all(row.locked_holdout for row in observations),
        "source_disjoint": not overlap,
        "minimum_claims": claims >= policy.minimum_claims,
        "minimum_accepted_critical_fields": (
            accepted_critical >= policy.minimum_accepted_critical_field_decisions
        ),
        "overall_raw_accuracy": (
            evaluated > 0 and correct / evaluated >= policy.minimum_overall_raw_accuracy
        ),
        "critical_raw_accuracy": (
            evaluated_critical > 0
            and correct_evaluated_critical / evaluated_critical
            >= policy.minimum_critical_raw_accuracy
        ),
        "accepted_precision": (
            accepted_precision is not None
            and accepted_precision >= policy.minimum_accepted_precision
        ),
        "claim_hitl": claim_hitl is not None and claim_hitl <= policy.maximum_claim_hitl,
        "claim_hitl_upper_95": upper is not None and upper < policy.maximum_claim_hitl_upper_95,
        "segment_claim_hitl": (
            maximum_segment is not None
            and maximum_segment <= policy.maximum_segment_claim_hitl
        ),
        "false_accept_rate": (
            false_accept_rate is not None
            and false_accept_rate <= policy.maximum_false_accept_rate
        ),
        "critical_false_accepts": critical_false_accepts <= policy.maximum_critical_false_accepts,
        "critical_accepted_precision": (
            critical_precision is not None
            and critical_precision >= policy.minimum_critical_accepted_precision
        ),
        "wrong_crop_recall": (
            wrong_crop_recall is not None
            and wrong_crop_recall >= policy.minimum_wrong_crop_recall
        ),
        "runtime_parity": bool(observations) and all(
            row.runtime_decision_parity for row in observations
        ),
        "route_governance": bool(observations) and all(
            row.route_governance_passed for row in observations
        ),
        "ocr_only_processing": (
            ocr_only_processing_rate is not None
            and ocr_only_processing_rate >= policy.minimum_ocr_only_processing_rate
        ),
        "llm_escalation": (
            llm_escalation_rate is not None
            and llm_escalation_rate <= policy.maximum_llm_escalation_rate
        ),
        "shadow_only": all(row.shadow_only for row in observations),
    }
    blockers = [name.upper() for name, passed in gates.items() if not passed]
    return ShadowQualificationReport(
        status="QUALIFIED" if not blockers else "NEEDS_MORE_DATA",
        claim_count=claims,
        source_group_count=len({row.source_group_id for row in observations}),
        claim_hitl=claim_hitl,
        claim_hitl_upper_95=upper,
        claim_stp=(1 - claim_hitl) if claim_hitl is not None else None,
        maximum_segment_claim_hitl=maximum_segment,
        segment_claim_hitl=segment_hitl,
        evaluated_field_decisions=evaluated,
        overall_raw_accuracy=correct / evaluated if evaluated else None,
        critical_accuracy=(
            correct_evaluated_critical / evaluated_critical
            if evaluated_critical else None
        ),
        safe_field_coverage=accepted / evaluated if evaluated else None,
        accepted_field_decisions=accepted,
        accepted_critical_field_decisions=accepted_critical,
        accepted_precision=accepted_precision,
        false_accept_rate=false_accept_rate,
        critical_false_accepts=critical_false_accepts,
        critical_accepted_precision=critical_precision,
        wrong_crop_recall=wrong_crop_recall,
        p95_latency_ms=p95_latency,
        cost_per_document_usd=cost_per_document,
        ocr_only_processing_rate=ocr_only_processing_rate,
        llm_escalation_rate=llm_escalation_rate,
        gates=gates,
        blocking_reasons=blockers,
    )
