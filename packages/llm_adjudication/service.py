"""Production shadow observer for canonical unresolved field decisions."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from packages.llm_adjudication.azure import (
    AdjudicationCandidate,
    AdjudicationRequest,
    AdjudicationResult,
    AzureLLMPricingConfig,
    LLMAdjudicationConfig,
    LLMCostGovernor,
    LLMRouter,
    build_azure_adjudication_provider,
)


def _optional_float(value: str | None) -> float | None:
    return float(value) if value and value.strip() else None


@dataclass
class AzureShadowAdjudicationService:
    router: LLMRouter
    governor: LLMCostGovernor

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None):
        env = environ or os.environ
        config = LLMAdjudicationConfig.from_env(env)
        if not config.enabled:
            return None
        if config.mode != "SHADOW_ONLY":
            raise ValueError("runtime integration is shadow-only until promotion gates pass")
        pricing = AzureLLMPricingConfig(
            _optional_float(
                env.get("AZURE_OPENAI_INPUT_COST_PER_MILLION")
                or env.get("AZURE_OPENAI_INPUT_COST_PER_MILLION_TOKENS")
            ),
            _optional_float(
                env.get("AZURE_OPENAI_OUTPUT_COST_PER_MILLION")
                or env.get("AZURE_OPENAI_OUTPUT_COST_PER_MILLION_TOKENS")
            ),
            _optional_float(
                env.get("AZURE_OPENAI_CACHED_INPUT_COST_PER_MILLION")
                or env.get("AZURE_OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS")
            ),
        )
        governor = LLMCostGovernor(config, pricing)
        return cls(
            LLMRouter(config, build_azure_adjudication_provider(config, pricing), governor),
            governor,
        )

    def observe(
        self,
        *,
        field_name: str,
        field_type: str,
        candidates: Sequence[str],
        claim_blocking: bool,
        crop_safe: bool,
        localization_confidence: float,
        critical: bool,
        authoritative_conflict: bool,
        page_key: str,
        claim_key: str,
        claim_distance: int,
        evidence: Mapping[str, Any],
    ) -> AdjudicationResult:
        self.governor.register_page()
        unique = tuple(dict.fromkeys(v for v in candidates if v))
        request = AdjudicationRequest(
            field_name=field_name,
            field_type=field_type,
            candidates=tuple(
                AdjudicationCandidate(f"candidate_{i}", v) for i, v in enumerate(unique)
            ),
            claim_blocking=claim_blocking,
            crop_safe=crop_safe,
            localization_confidence=localization_confidence,
            critical=critical,
            authoritative_conflict=authoritative_conflict,
            claim_distance=claim_distance,
            evidence=evidence,
        )
        return self.router.route(request, page_key=page_key, claim_key=claim_key)
