import json

import httpx
import pytest

from packages.llm_adjudication.azure import *


def cfg(**kw):
    v = {
        "enabled": True,
        "endpoint": "https://x.openai.azure.com",
        "deployment": "gpt-4o",
        "api_key": "test",
        "max_retries": 1,
    }
    v.update(kw)
    return LLMAdjudicationConfig(**v)


def req(**kw):
    v = {
        "field_name": "provider_name",
        "field_type": "NAME",
        "candidates": (
            AdjudicationCandidate("candidate_0", "JOHN SM1TH"),
            AdjudicationCandidate("candidate_1", "JOHN SMITH"),
        ),
        "claim_blocking": True,
        "crop_safe": True,
        "localization_confidence": 0.96,
        "evidence": {"provider_section": "RENDERING", "patient_name": "DROP"},
    }
    v.update(kw)
    return AdjudicationRequest(**v)


def body(decision="SELECT_CANDIDATE", candidate_id="candidate_1", **extra):
    x = {"decision": decision, "candidate_id": candidate_id, "reason_code": "BEST"}
    x.update(extra)
    return {
        "model": "gpt-4o-version",
        "choices": [{"message": {"content": json.dumps(x)}}],
        "usage": {"prompt_tokens": 90, "completion_tokens": 12},
    }


def fake(*items):
    calls = []
    pending = list(items)

    def handler(r):
        calls.append(r)
        status, payload = pending.pop(0)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_disabled_defaults_and_missing_auth():
    default = LLMAdjudicationConfig.from_env({})
    assert not default.enabled and default.mode == "SHADOW_ONLY"
    default.validate()
    with pytest.raises(LLMConfigurationError, match="credential"):
        cfg(api_key=None).validate()


def test_closed_world_mock_response_usage_cost_and_minimization():
    client, calls = fake((200, body()))
    p = AzureOpenAIAdjudicationProvider(cfg(), AzureLLMPricingConfig(2, 8), http_client=client)
    r = p.adjudicate(req(), RoutingTier.TEXT)
    assert (
        r.selected_value == "JOHN SMITH"
        and r.deployment == "gpt-4o"
        and r.model == "gpt-4o-version"
        and (r.input_tokens, r.output_tokens) == (90, 12)
    )
    assert r.cost_usd == pytest.approx((90 * 2 + 12 * 8) / 1_000_000)
    raw = calls[0].content.decode()
    assert "patient_name" not in raw and json.loads(raw)["max_tokens"] == 40


@pytest.mark.parametrize(
    "payload",
    [
        body(candidate_id="unknown"),
        body(value="NOVEL"),
        {"choices": [{"message": {"content": "bad"}}]},
        body(decision="BAD", candidate_id=None),
    ],
)
def test_unknown_novel_malformed_and_unsupported_fail_closed(payload):
    client, _ = fake((200, payload))
    r = AzureOpenAIAdjudicationProvider(
        cfg(), AzureLLMPricingConfig(), http_client=client
    ).adjudicate(req(), RoutingTier.TEXT)
    assert r.decision == AdjudicationDecision.HITL and r.selected_value is None


def test_phi_minimization():
    payload, categories = AzureLLMDataMinimizer().minimize(req())
    assert payload["evidence"] == {
        "crop_safe": True,
        "localization_confidence": 0.96,
        "provider_section": "RENDERING",
    }
    assert "claim_id" not in payload and len(categories) == 2


def test_tier0_tier1_tier2_and_safety_firewalls():
    client, calls = fake((200, body()), (200, body()))
    c = cfg()
    price = AzureLLMPricingConfig()
    router = LLMRouter(
        c, AzureOpenAIAdjudicationProvider(c, price, http_client=client), LLMCostGovernor(c, price)
    )
    assert (
        router.route(req(local_resolved=True), page_key="0", claim_key="c").tier
        == RoutingTier.LOCAL_ONLY
    )
    assert (
        router.route(req(authoritative_conflict=True), page_key="0", claim_key="c").reason_code
        == "AUTHORITATIVE_CONFLICT_PROTECTED"
    )
    assert router.route(req(), page_key="1", claim_key="c").tier == RoutingTier.TEXT
    assert (
        router.route(
            req(tier1_failed=True, visual_resolvable=True, crop_bytes=b"png"),
            page_key="2",
            claim_key="c",
        ).tier
        == RoutingTier.FIELD_CROP
    )
    assert "data:image/png;base64" in calls[1].content.decode()
    assert (
        router.route(
            req(tier1_failed=True, visual_resolvable=True, crop_safe=False),
            page_key="3",
            claim_key="c",
        ).reason_code
        == "TIER2_INELIGIBLE"
    )


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_retry_once(status):
    client, calls = fake((status, {}), (status, {}))
    r = AzureOpenAIAdjudicationProvider(
        cfg(), AzureLLMPricingConfig(), http_client=client
    ).adjudicate(req(), RoutingTier.TEXT)
    assert r.decision == AdjudicationDecision.HITL and r.retries == 1 and len(calls) == 2


def test_timeout_cache_pricing_governor_shadow_and_promotion():
    timeout_calls = []

    def timeout(r):
        timeout_calls.append(r)
        raise httpx.ReadTimeout("slow", request=r)

    r = AzureOpenAIAdjudicationProvider(
        cfg(),
        AzureLLMPricingConfig(),
        http_client=httpx.Client(transport=httpx.MockTransport(timeout)),
    ).adjudicate(req(), RoutingTier.TEXT)
    assert r.decision == AdjudicationDecision.HITL and len(timeout_calls) == 2
    client, calls = fake((200, body()))
    p = AzureOpenAIAdjudicationProvider(cfg(), AzureLLMPricingConfig(1, 1), http_client=client)
    assert not p.adjudicate(req(), RoutingTier.TEXT).cache_hit
    assert p.adjudicate(req(), RoutingTier.TEXT).cache_hit and len(calls) == 1
    authority = cfg(mode="AUTHORITY")
    missing = AzureLLMPricingConfig()
    g = LLMCostGovernor(authority, missing)
    g.register_page()
    assert (
        g.authorize("p", "c", RoutingTier.TEXT, authority=True)[1] == "PRICING_NOT_CONFIGURED"
        and g.mean_cost_per_page is None
    )
    expensive = LLMCostGovernor(authority, AzureLLMPricingConfig(1000, 1000))
    expensive.register_page()
    assert expensive.authorize("p", "c", RoutingTier.TEXT, authority=True)[1] == "CLAIM_COST_LIMIT"
    metrics = {
        "trusted_evaluation": True,
        "llm_candidate_selection_precision": 0.995,
        "overall_accepted_precision": 0.996,
        "critical_false_accepts": 0,
        "mean_paid_ai_cost_per_page_usd": 0.001,
        "authoritative_conflicts_overridden": 0,
        "novel_values_accepted": 0,
    }
    assert (
        promotion_gate(metrics, AzureLLMPricingConfig(1, 1)) == (True, "PASS")
        and promotion_gate(metrics, missing)[1] == "PRICING_NOT_CONFIGURED"
    )


def test_shadow_runtime_factory_is_disabled_by_default_and_rejects_authority():
    from packages.llm_adjudication.service import AzureShadowAdjudicationService

    assert AzureShadowAdjudicationService.from_env({}) is None
    with pytest.raises(ValueError, match="shadow-only"):
        AzureShadowAdjudicationService.from_env(
            {
                "LLM_ENABLED": "true",
                "LLM_MODE": "AUTHORITY",
                "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
                "AZURE_AI_EVALUATION_DEPLOYMENT": "gpt-4o",
                "AZURE_OPENAI_API_KEY": "test",
            }
        )
