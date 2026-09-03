from __future__ import annotations

from dataclasses import dataclass

from packages.extraction_providers.contracts import ProviderCapability


@dataclass(frozen=True)
class RouteCandidate:
    provider: str
    capabilities: frozenset[ProviderCapability]
    expected_accuracy: float
    expected_latency_ms: float
    expected_cost_usd: float
    cloud: bool = False


class ExtractionRoutingPolicy:
    """Selects provider order from explicit cost/latency/privacy constraints."""

    def rank(
        self,
        routes: list[RouteCandidate],
        *,
        required: ProviderCapability,
        privacy_allows_cloud: bool,
        latency_budget_ms: float | None,
        cost_budget_usd: float | None,
    ) -> list[RouteCandidate]:
        eligible = []
        for route in routes:
            if required not in route.capabilities:
                continue
            if route.cloud and not privacy_allows_cloud:
                continue
            if latency_budget_ms is not None and route.expected_latency_ms > latency_budget_ms:
                continue
            if cost_budget_usd is not None and route.expected_cost_usd > cost_budget_usd:
                continue
            eligible.append(route)

        def score(route: RouteCandidate) -> float:
            accuracy = max(0.0, min(1.0, route.expected_accuracy))
            latency_penalty = min(1.0, route.expected_latency_ms / max(1.0, latency_budget_ms or 5000.0))
            cost_penalty = min(1.0, route.expected_cost_usd / max(0.000001, cost_budget_usd or 0.05))
            return 0.72 * accuracy - 0.18 * latency_penalty - 0.10 * cost_penalty

        return sorted(eligible, key=score, reverse=True)
