from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionMetrics:
    accepted_precision: float
    critical_field_precision: float
    critical_false_accepts: int
    field_hitl_rate: float
    claim_hitl_rate: float
    claim_stp_rate: float
    p95_seconds_per_page: float
    cost_usd_per_page: float
    sample_size: int
    overall_field_accuracy: float | None = None
    critical_field_accuracy: float | None = None
    routing_accuracy: float | None = None


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


class PromotionGate:
    """Fail-closed promotion gate for production model/policy changes."""

    def __init__(
        self,
        *,
        min_accepted_precision: float = 0.995,
        min_critical_precision: float = 0.995,
        max_critical_false_accepts: int = 0,
        min_overall_field_accuracy: float = 0.98,
        min_critical_field_accuracy: float = 0.995,
        min_claim_stp_rate: float = 0.80,
        max_field_hitl_rate: float = 0.15,
        max_claim_hitl_rate: float = 0.20,
        max_p95_seconds_per_page: float = 5.0,
        max_cost_usd_per_page: float = 0.03,
        min_sample_size: int = 500,
    ) -> None:
        self.min_accepted_precision = min_accepted_precision
        self.min_critical_precision = min_critical_precision
        self.max_critical_false_accepts = max_critical_false_accepts
        self.min_overall_field_accuracy = min_overall_field_accuracy
        self.min_critical_field_accuracy = min_critical_field_accuracy
        self.min_claim_stp_rate = min_claim_stp_rate
        self.max_field_hitl_rate = max_field_hitl_rate
        self.max_claim_hitl_rate = max_claim_hitl_rate
        self.max_p95_seconds_per_page = max_p95_seconds_per_page
        self.max_cost_usd_per_page = max_cost_usd_per_page
        self.min_sample_size = min_sample_size

    def evaluate(self, metrics: PromotionMetrics) -> PromotionDecision:
        reasons: list[str] = []
        if metrics.sample_size < self.min_sample_size:
            reasons.append("INSUFFICIENT_SAMPLE_SIZE")
        if metrics.accepted_precision < self.min_accepted_precision:
            reasons.append("ACCEPTED_PRECISION_BELOW_GATE")
        if metrics.critical_field_precision < self.min_critical_precision:
            reasons.append("CRITICAL_PRECISION_BELOW_GATE")
        if metrics.critical_false_accepts > self.max_critical_false_accepts:
            reasons.append("CRITICAL_FALSE_ACCEPTS_PRESENT")
        if (
            metrics.overall_field_accuracy is not None
            and metrics.overall_field_accuracy < self.min_overall_field_accuracy
        ):
            reasons.append("OVERALL_FIELD_ACCURACY_BELOW_GATE")
        if (
            metrics.critical_field_accuracy is not None
            and metrics.critical_field_accuracy < self.min_critical_field_accuracy
        ):
            reasons.append("CRITICAL_FIELD_ACCURACY_BELOW_GATE")
        if metrics.claim_stp_rate < self.min_claim_stp_rate:
            reasons.append("CLAIM_STP_BELOW_GATE")
        if metrics.field_hitl_rate > self.max_field_hitl_rate:
            reasons.append("FIELD_HITL_ABOVE_GATE")
        if metrics.claim_hitl_rate > self.max_claim_hitl_rate:
            reasons.append("CLAIM_HITL_ABOVE_GATE")
        if metrics.p95_seconds_per_page > self.max_p95_seconds_per_page:
            reasons.append("P95_LATENCY_ABOVE_GATE")
        if metrics.cost_usd_per_page > self.max_cost_usd_per_page:
            reasons.append("COST_ABOVE_GATE")
        return PromotionDecision(promote=not reasons, reasons=tuple(reasons))
