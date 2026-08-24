"""Fail-closed activation gates for promoting specialist extraction improvements.

Specialist outputs may run in shadow freely, but may affect production only when
measured evidence satisfies the configured safety and performance gates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpecialistMetrics:
    sample_size: int
    baseline_accuracy: float
    specialist_accuracy: float
    accepted_precision: float
    critical_accepted_precision: float
    critical_false_accepts: int
    field_hitl_rate: float
    p95_seconds_per_page: float
    cost_usd_per_page: float


@dataclass(frozen=True, slots=True)
class SpecialistActivationDecision:
    activate: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpecialistActivationGate:
    min_sample_size: int = 50
    min_accuracy_gain: float = 0.01
    min_accepted_precision: float = 0.995
    min_critical_precision: float = 0.995
    max_critical_false_accepts: int = 0
    max_field_hitl_rate: float = 0.15
    max_p95_seconds_per_page: float = 5.0
    max_cost_usd_per_page: float = 0.03

    def evaluate(self, metrics: SpecialistMetrics) -> SpecialistActivationDecision:
        reasons: list[str] = []
        if metrics.sample_size < self.min_sample_size:
            reasons.append("INSUFFICIENT_SAMPLE_SIZE")
        if metrics.specialist_accuracy - metrics.baseline_accuracy < self.min_accuracy_gain:
            reasons.append("INSUFFICIENT_ACCURACY_GAIN")
        if metrics.accepted_precision < self.min_accepted_precision:
            reasons.append("ACCEPTED_PRECISION_BELOW_GATE")
        if metrics.critical_accepted_precision < self.min_critical_precision:
            reasons.append("CRITICAL_PRECISION_BELOW_GATE")
        if metrics.critical_false_accepts > self.max_critical_false_accepts:
            reasons.append("CRITICAL_FALSE_ACCEPT_PRESENT")
        if metrics.field_hitl_rate > self.max_field_hitl_rate:
            reasons.append("FIELD_HITL_ABOVE_GATE")
        if metrics.p95_seconds_per_page > self.max_p95_seconds_per_page:
            reasons.append("LATENCY_ABOVE_GATE")
        if metrics.cost_usd_per_page > self.max_cost_usd_per_page:
            reasons.append("COST_ABOVE_GATE")
        return SpecialistActivationDecision(not reasons, tuple(reasons))
