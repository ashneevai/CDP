import hashlib

import pytest

from packages.ai_gateway import (
    AdaptiveResolutionService,
    AIGateway,
    FieldResolutionRequest,
    FieldResolutionResponse,
    SelectiveResolutionCoordinator,
    SelectiveResolutionError,
    TenantAIPolicy,
)
from packages.policy_engine import AdaptivePolicyEngine, DecisionContext, PolicyAction


class Provider:
    provider_name = "vertex_ai_gemini"
    model_name = "gemini-2.5-flash-lite"
    model_version = "2.5"
    region = "us-central1"

    def __init__(self, value="A123", confidence=.96):
        self.value = value
        self.confidence = confidence
        self.calls = 0

    async def resolve(self, request):
        self.calls += 1
        return FieldResolutionResponse(
            value=self.value,
            confidence=self.confidence,
            insufficient_evidence=False,
            provider=self.provider_name,
            model=self.model_name,
            model_version=self.model_version,
            actual_cost_usd=.001,
        )


def _request(**changes):
    crop = b"only-the-field-crop"
    values = dict(
        request_id="r1", tenant_id="t1", document_id="d1", field_name="member_id",
        expected_type="code", crop_bytes=crop, crop_sha256=hashlib.sha256(crop).hexdigest(),
        allowed_pattern=r"[A-Z]\d{3}",
    )
    values.update(changes)
    return FieldResolutionRequest(**values)


def _coordinator(provider, **changes):
    policy = TenantAIPolicy(
        tenant_id="t1", enabled=True, phi_external_processing_approved=True,
        approved_regions={"us-central1"}, allowed_models={provider.model_name},
        daily_budget_usd=1, max_requests_per_minute=10,
    )
    return SelectiveResolutionCoordinator(
        AIGateway({"t1": policy}), {provider.model_name: provider}, **changes
    )


@pytest.mark.asyncio
async def test_cloud_result_is_auxiliary_and_requires_reconciliation():
    result = await _coordinator(Provider()).resolve(
        PolicyAction.GEMINI_CHEAP, _request(), estimated_cost_usd=.002
    )
    assert result.candidate.value == "A123"
    assert result.candidate.acceptance_authority is False
    assert result.requires_reconciliation is True
    assert result.candidate.validation_results == ("allowed_pattern_passed",)


@pytest.mark.asyncio
async def test_pattern_failure_marks_candidate_insufficient_not_accepted():
    result = await _coordinator(Provider(value="wrong value")).resolve(
        PolicyAction.GEMINI_CHEAP, _request(), estimated_cost_usd=.002
    )
    assert result.candidate.insufficient_evidence is True
    assert "allowed_pattern_failed" in result.candidate.validation_results


@pytest.mark.asyncio
async def test_npi_cannot_be_sent_to_gemini():
    provider = Provider()
    with pytest.raises(SelectiveResolutionError, match="NPI"):
        await _coordinator(provider).resolve(
            PolicyAction.GEMINI_CHEAP, _request(field_name="provider_npi"),
            estimated_cost_usd=.002,
        )
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_per_field_cloud_attempt_limit_is_fail_closed():
    provider = Provider()
    coordinator = _coordinator(provider, max_cloud_attempts_per_field=1)
    await coordinator.resolve(PolicyAction.GEMINI_CHEAP, _request(), estimated_cost_usd=.002)
    with pytest.raises(SelectiveResolutionError, match="limit"):
        await coordinator.resolve(
            PolicyAction.GEMINI_CHEAP, _request(request_id="r2"), estimated_cost_usd=.002
        )
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_action_must_have_exact_configured_provider():
    with pytest.raises(SelectiveResolutionError, match="not configured"):
        await _coordinator(Provider()).resolve(
            PolicyAction.TEXTRACT, _request(), estimated_cost_usd=.003
        )


@pytest.mark.asyncio
async def test_adaptive_service_executes_only_the_selected_cloud_step():
    provider = Provider()
    service = AdaptiveResolutionService(
        AdaptivePolicyEngine.load(), _coordinator(provider)
    )
    context = DecisionContext(
        document_type="CMS1500", field_name="member_id", criticality="critical",
        previous_attempts={PolicyAction.RAPIDOCR, PolicyAction.RETRY_PREPROCESSING,
                           PolicyAction.PADDLEOCR, PolicyAction.TESSERACT,
                           PolicyAction.REFERENCE_LOOKUP},
        cloud_processing_allowed=True,
    )
    step = await service.execute_next(context, _request())
    assert step.decision.action is PolicyAction.GEMINI_CHEAP
    assert step.resolution is not None
    assert step.resolution.requires_reconciliation is True


@pytest.mark.asyncio
async def test_adaptive_service_does_not_call_cloud_for_local_step():
    provider = Provider()
    service = AdaptiveResolutionService(
        AdaptivePolicyEngine.load(), _coordinator(provider)
    )
    step = await service.execute_next(
        DecisionContext(document_type="CMS1500", field_name="member_id", criticality="critical"),
        _request(),
    )
    assert step.decision.action is PolicyAction.RAPIDOCR
    assert step.resolution is None
    assert provider.calls == 0
