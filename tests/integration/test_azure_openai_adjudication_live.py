"""Explicit opt-in Azure test; never consumes quota by default."""

import os

import pytest

from packages.llm_adjudication import (
    AdjudicationCandidate,
    AdjudicationRequest,
    AzureLLMPricingConfig,
    LLMAdjudicationConfig,
    build_azure_adjudication_provider,
)
from packages.llm_adjudication.azure import RoutingTier


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_OPENAI_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Azure quota use requires explicit opt-in",
)
def test_live_azure_closed_world_smoke():
    config = LLMAdjudicationConfig.from_env()
    assert config.enabled and config.mode == "SHADOW_ONLY"
    result = build_azure_adjudication_provider(config, AzureLLMPricingConfig()).adjudicate(
        AdjudicationRequest(
            field_name="non_phi_test_code",
            field_type="CODE",
            candidates=(
                AdjudicationCandidate("candidate_0", "ABC"),
                AdjudicationCandidate("candidate_1", "ABD"),
            ),
            claim_blocking=True,
            crop_safe=True,
            localization_confidence=0.9,
        ),
        RoutingTier.TEXT,
    )
    assert result.selected_value in {None, "ABC", "ABD"}
