"""Closed-world Azure OpenAI adjudication with fail-closed cost controls.

This module is intentionally separate from generative field extraction.  The
provider may choose only an already-observed candidate or abstain; provider
text can never become a claim value.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol

import httpx


class LLMConfigurationError(RuntimeError):
    """Enabled Azure adjudication is missing safe runtime configuration."""


class AdjudicationDecision(StrEnum):
    SELECT_CANDIDATE = "SELECT_CANDIDATE"
    HITL = "HITL"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RoutingTier(StrEnum):
    LOCAL_ONLY = "TIER0"
    TEXT = "TIER1"
    FIELD_CROP = "TIER2"


@dataclass(frozen=True)
class AzureLLMPricingConfig:
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    cached_input_cost_per_million_tokens: float | None = None

    @property
    def configured(self) -> bool:
        return (
            self.input_cost_per_million_tokens is not None
            and self.output_cost_per_million_tokens is not None
        )

    @property
    def cost_status(self) -> str:
        return "CONFIGURED" if self.configured else "PRICING_NOT_CONFIGURED"

    def calculate(
        self, input_tokens: int, output_tokens: int, *, cached: bool = False
    ) -> float | None:
        if not self.configured:
            return None
        input_rate = self.input_cost_per_million_tokens
        if cached and self.cached_input_cost_per_million_tokens is not None:
            input_rate = self.cached_input_cost_per_million_tokens
        return (
            input_tokens * float(input_rate)
            + output_tokens * float(self.output_cost_per_million_tokens)
        ) / 1_000_000


@dataclass(frozen=True)
class LLMAdjudicationConfig:
    enabled: bool = False
    mode: str = "SHADOW_ONLY"
    provider: str = "azure_openai"
    endpoint: str | None = None
    deployment: str | None = None
    api_version: str = "2024-10-21"
    api_key: str | None = field(default=None, repr=False)
    target_avg_cost_per_page_usd: float = 0.0005
    max_avg_cost_per_page_usd: float = 0.001
    max_cost_per_claim_usd: float = 0.005
    max_tier1_calls_per_page: int = 1
    max_tier2_calls_per_page: int = 1
    tier1_max_input_tokens: int = 500
    tier1_max_output_tokens: int = 40
    timeout_seconds: float = 15.0
    max_retries: int = 1
    prompt_version: str = "azure-closed-world-v1"
    policy_version: str = "llm-cost-policy-v1"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMAdjudicationConfig:
        env = environ or os.environ

        def flag(name: str, default: bool = False) -> bool:
            return env.get(name, str(default)).strip().casefold() in {"1", "true", "yes", "on"}

        return cls(
            enabled=flag("LLM_ENABLED"),
            mode=env.get("LLM_MODE", "SHADOW_ONLY").upper(),
            provider=env.get("LLM_PROVIDER", "azure_openai").lower(),
            endpoint=env.get("AZURE_OPENAI_ENDPOINT") or None,
            deployment=env.get("AZURE_AI_EVALUATION_DEPLOYMENT") or None,
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            api_key=env.get("AZURE_OPENAI_API_KEY") or None,
            target_avg_cost_per_page_usd=float(
                env.get("LLM_TARGET_AVG_COST_PER_PAGE_USD", "0.0005")
            ),
            max_avg_cost_per_page_usd=float(env.get("LLM_MAX_AVG_COST_PER_PAGE_USD", "0.001")),
            max_cost_per_claim_usd=float(env.get("LLM_MAX_COST_PER_CLAIM_USD", "0.005")),
            max_tier1_calls_per_page=int(env.get("LLM_MAX_TIER1_CALLS_PER_PAGE", "1")),
            max_tier2_calls_per_page=int(env.get("LLM_MAX_TIER2_CALLS_PER_PAGE", "1")),
            tier1_max_input_tokens=int(env.get("LLM_TIER1_MAX_INPUT_TOKENS", "500")),
            tier1_max_output_tokens=int(env.get("LLM_TIER1_MAX_OUTPUT_TOKENS", "40")),
            timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "15")),
            max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
        )

    def validate(self, *, identity_available: bool = False) -> None:
        if not self.enabled:
            return
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_ENDPOINT", self.endpoint),
                ("AZURE_AI_EVALUATION_DEPLOYMENT", self.deployment),
                ("AZURE_OPENAI_API_VERSION", self.api_version),
            )
            if not value
        ]
        if self.provider != "azure_openai":
            missing.append("LLM_PROVIDER=azure_openai")
        if not self.api_key and not identity_available:
            missing.append("Azure credential (managed identity or AZURE_OPENAI_API_KEY)")
        if missing:
            raise LLMConfigurationError(
                "LLM_ENABLED=true but configuration is unavailable: " + ", ".join(missing)
            )
        if self.mode not in {"SHADOW_ONLY", "AUTHORITY"}:
            raise LLMConfigurationError("LLM_MODE must be SHADOW_ONLY or AUTHORITY")


@dataclass(frozen=True)
class AdjudicationCandidate:
    candidate_id: str
    value: str


@dataclass(frozen=True)
class AdjudicationRequest:
    field_name: str
    field_type: str
    candidates: tuple[AdjudicationCandidate, ...]
    claim_blocking: bool
    crop_safe: bool
    localization_confidence: float
    critical: bool = False
    authoritative_conflict: bool = False
    local_resolved: bool = False
    tier1_failed: bool = False
    visual_resolvable: bool = False
    claim_distance: int = 1
    evidence: Mapping[str, Any] = field(default_factory=dict)
    crop_bytes: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AdjudicationResult:
    decision: AdjudicationDecision
    candidate_id: str | None
    selected_value: str | None
    reason_code: str
    tier: RoutingTier
    authoritative: bool = False
    cache_hit: bool = False
    provider: str = "AZURE_OPENAI"
    deployment: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    cost_status: str = "PRICING_NOT_CONFIGURED"
    retries: int = 0
    data_categories_sent: tuple[str, ...] = ()
    request_id: str | None = None


class AzureLLMDataMinimizer:
    """Construct a target-field-only payload without internal identifiers."""

    _ALLOWED_EVIDENCE: ClassVar[set[str]] = {
        "crop_safe",
        "localization_confidence",
        "npi_valid",
        "provider_section",
        "conflict",
        "validation_codes",
        "semantic_section",
        "field_business_valid",
    }

    def minimize(self, request: AdjudicationRequest) -> tuple[dict[str, Any], tuple[str, ...]]:
        evidence = {
            key: request.evidence[key]
            for key in sorted(self._ALLOWED_EVIDENCE & request.evidence.keys())
        }
        evidence["crop_safe"] = request.crop_safe
        evidence["localization_confidence"] = round(request.localization_confidence, 6)
        payload = {
            "field_name": request.field_name,
            "field_type": request.field_type,
            "critical": request.critical,
            "candidates": [
                {"id": candidate.candidate_id, "value": candidate.value}
                for candidate in request.candidates
            ],
            "evidence": evidence,
        }
        categories = ("TARGET_FIELD_CANDIDATES", "TARGET_FIELD_EVIDENCE")
        return payload, categories


class LLMCostGovernor:
    def __init__(self, config: LLMAdjudicationConfig, pricing: AzureLLMPricingConfig) -> None:
        self.config = config
        self.pricing = pricing
        self.total_pages = 0
        self.total_cost_usd = 0.0
        self.page_costs: dict[str, float] = {}
        self.claim_costs: dict[str, float] = {}
        self.page_calls: dict[tuple[str, RoutingTier], int] = {}

    def register_page(self) -> None:
        self.total_pages += 1

    def authorize(
        self, page_key: str, claim_key: str, tier: RoutingTier, *, authority: bool
    ) -> tuple[bool, str]:
        if authority and not self.pricing.configured:
            return False, "PRICING_NOT_CONFIGURED"
        limit = (
            self.config.max_tier1_calls_per_page
            if tier == RoutingTier.TEXT
            else self.config.max_tier2_calls_per_page
        )
        if self.page_calls.get((page_key, tier), 0) >= limit:
            return False, "PAGE_CALL_LIMIT"
        if self.pricing.configured:
            estimate = (
                self.pricing.calculate(
                    self.config.tier1_max_input_tokens, self.config.tier1_max_output_tokens
                )
                or 0.0
            )
            if self.claim_costs.get(claim_key, 0.0) + estimate > self.config.max_cost_per_claim_usd:
                return False, "CLAIM_COST_LIMIT"
            denominator = max(1, self.total_pages)
            if (
                self.total_cost_usd + estimate
            ) / denominator > self.config.max_avg_cost_per_page_usd:
                return False, "MEAN_COST_LIMIT"
        return True, "AUTHORIZED"

    def record(
        self, page_key: str, claim_key: str, tier: RoutingTier, cost_usd: float | None
    ) -> None:
        self.page_calls[(page_key, tier)] = self.page_calls.get((page_key, tier), 0) + 1
        if cost_usd is not None:
            self.total_cost_usd += cost_usd
            self.page_costs[page_key] = self.page_costs.get(page_key, 0.0) + cost_usd
            self.claim_costs[claim_key] = self.claim_costs.get(claim_key, 0.0) + cost_usd

    @property
    def mean_cost_per_page(self) -> float | None:
        if not self.pricing.configured or not self.total_pages:
            return None
        return self.total_cost_usd / self.total_pages


class ResponseClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


class AzureOpenAIAdjudicationProvider:
    _SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [item.value for item in AdjudicationDecision]},
            "candidate_id": {"type": ["string", "null"]},
            "reason_code": {"type": "string", "maxLength": 120},
        },
        "required": ["decision", "candidate_id", "reason_code"],
    }

    def __init__(
        self,
        config: LLMAdjudicationConfig,
        pricing: AzureLLMPricingConfig,
        *,
        http_client: ResponseClient | None = None,
        token_provider: Callable[[], str] | None = None,
        minimizer: AzureLLMDataMinimizer | None = None,
    ) -> None:
        config.validate(identity_available=token_provider is not None)
        self.config = config
        self.pricing = pricing
        self.client = http_client or httpx.Client(timeout=config.timeout_seconds)
        self.token_provider = token_provider
        self.minimizer = minimizer or AzureLLMDataMinimizer()
        self.cache: dict[str, AdjudicationResult] = {}
        self.cache_lookups = 0
        self.cache_hits = 0

    def _cache_key(self, request: AdjudicationRequest, tier: RoutingTier) -> str:
        minimized, _ = self.minimizer.minimize(request)
        identity = {
            "deployment": self.config.deployment,
            "api_version": self.config.api_version,
            "prompt_version": self.config.prompt_version,
            "policy_version": self.config.policy_version,
            "tier": tier.value,
            "candidate_hashes": [
                hashlib.sha256(c.value.encode()).hexdigest() for c in request.candidates
            ],
            "evidence_hash": hashlib.sha256(
                json.dumps(minimized["evidence"], sort_keys=True).encode()
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def adjudicate(self, request: AdjudicationRequest, tier: RoutingTier) -> AdjudicationResult:
        if not self.config.enabled:
            return self._fallback("LLM_DISABLED", tier)
        self.cache_lookups += 1
        key = self._cache_key(request, tier)
        if key in self.cache:
            cached = self.cache[key]
            return AdjudicationResult(**{**cached.__dict__, "cache_hit": True, "cost_usd": 0.0})
        minimized, categories = self.minimizer.minimize(request)
        minimized_text = json.dumps(minimized, separators=(",", ":"), sort_keys=True)
        if (len(minimized_text) + 1) // 2 > self.config.tier1_max_input_tokens:
            return self._fallback("INPUT_TOKEN_CAP_PREFLIGHT", tier)
        content: list[dict[str, Any]] = [{"type": "text", "text": minimized_text}]
        if tier == RoutingTier.FIELD_CROP:
            if not request.crop_safe or not request.crop_bytes:
                return self._fallback("UNSAFE_OR_MISSING_CROP", tier)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(request.crop_bytes).decode("ascii"),
                        "detail": "high",
                    },
                }
            )
            categories += ("TARGET_FIELD_CROP",)
        payload = {
            "temperature": 0,
            "max_tokens": self.config.tier1_max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "Choose only a supplied candidate id or abstain. Never create or rewrite a value.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "closed_world_adjudication",
                    "strict": True,
                    "schema": self._SCHEMA,
                },
            },
        }
        url = f"{str(self.config.endpoint).rstrip('/')}/openai/deployments/{self.config.deployment}/chat/completions?api-version={self.config.api_version}"
        try:
            headers = (
                {"api-key": self.config.api_key}
                if self.config.api_key
                else {
                    "Authorization": f"Bearer {self.token_provider() if self.token_provider else ''}"
                }
            )
        except Exception:  # noqa: BLE001 - credential chain errors vary
            return self._fallback("AUTHENTICATION_FAILURE", tier)
        started = time.perf_counter()
        retries = 0
        try:
            while True:
                try:
                    response = self.client.post(
                        url, json=payload, headers=headers, timeout=self.config.timeout_seconds
                    )
                    if (
                        response.status_code in {429, 500, 502, 503, 504}
                        and retries < self.config.max_retries
                    ):
                        retries += 1
                        continue
                    response.raise_for_status()
                    break
                except (httpx.TimeoutException, httpx.TransportError):
                    if retries >= self.config.max_retries:
                        raise
                    retries += 1
            body = response.json()
            parsed = json.loads(body["choices"][0]["message"]["content"])
            if set(parsed) != {"decision", "candidate_id", "reason_code"}:
                raise ValueError("schema keys do not match")
            decision = AdjudicationDecision(parsed["decision"])
            candidate_id = parsed["candidate_id"]
            candidates = {
                candidate.candidate_id: candidate.value for candidate in request.candidates
            }
            if decision == AdjudicationDecision.SELECT_CANDIDATE:
                if candidate_id not in candidates:
                    raise ValueError("unknown candidate")
                selected = candidates[candidate_id]
            else:
                if candidate_id is not None:
                    raise ValueError("abstention must not select a candidate")
                selected = None
            usage = body.get("usage", {})
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            if (
                input_tokens > self.config.tier1_max_input_tokens
                or output_tokens > self.config.tier1_max_output_tokens
            ):
                raise ValueError("token cap exceeded")
            result = AdjudicationResult(
                decision=decision,
                candidate_id=candidate_id,
                selected_value=selected,
                reason_code=str(parsed["reason_code"])[:120],
                tier=tier,
                deployment=self.config.deployment,
                model=body.get("model"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
                cost_usd=self.pricing.calculate(input_tokens, output_tokens),
                cost_status=self.pricing.cost_status,
                retries=retries,
                data_categories_sent=categories,
                request_id=response.headers.get("x-request-id")
                or response.headers.get("apim-request-id"),
            )
            self.cache[key] = result
            return result
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, httpx.HTTPError):
            return self._fallback("PROVIDER_ERROR_OR_INVALID_RESPONSE", tier, retries=retries)

    def _fallback(self, reason: str, tier: RoutingTier, *, retries: int = 0) -> AdjudicationResult:
        return AdjudicationResult(
            decision=AdjudicationDecision.HITL,
            candidate_id=None,
            selected_value=None,
            reason_code=reason,
            tier=tier,
            deployment=self.config.deployment,
            cost_status=self.pricing.cost_status,
            retries=retries,
        )


class LLMRouter:
    """Deterministic Tier 0/1/2 eligibility and shadow-authority firewall."""

    def __init__(
        self,
        config: LLMAdjudicationConfig,
        provider: AzureOpenAIAdjudicationProvider,
        governor: LLMCostGovernor,
    ) -> None:
        self.config = config
        self.provider = provider
        self.governor = governor

    @staticmethod
    def priority(request: AdjudicationRequest) -> tuple[int, int, float]:
        return (
            max(1, request.claim_distance),
            0 if request.critical else 1,
            -request.localization_confidence,
        )

    def route(
        self, request: AdjudicationRequest, *, page_key: str, claim_key: str
    ) -> AdjudicationResult:
        if not self.config.enabled or request.local_resolved or not request.claim_blocking:
            return self.provider._fallback("TIER0_LOCAL_RESULT", RoutingTier.LOCAL_ONLY)
        if request.authoritative_conflict:
            return self.provider._fallback(
                "AUTHORITATIVE_CONFLICT_PROTECTED", RoutingTier.LOCAL_ONLY
            )
        tier = RoutingTier.TEXT
        if request.tier1_failed:
            if not request.crop_safe or not request.visual_resolvable:
                return self.provider._fallback("TIER2_INELIGIBLE", RoutingTier.LOCAL_ONLY)
            tier = RoutingTier.FIELD_CROP
        authority = self.config.mode == "AUTHORITY"
        allowed, reason = self.governor.authorize(page_key, claim_key, tier, authority=authority)
        if not allowed:
            return self.provider._fallback(reason, tier)
        result = self.provider.adjudicate(request, tier)
        if not result.cache_hit:
            self.governor.record(page_key, claim_key, tier, result.cost_usd)
        if self.config.mode == "SHADOW_ONLY":
            return AdjudicationResult(**{**result.__dict__, "authoritative": False})
        return AdjudicationResult(
            **{
                **result.__dict__,
                "authoritative": result.decision == AdjudicationDecision.SELECT_CANDIDATE,
            }
        )


def promotion_gate(metrics: Mapping[str, Any], pricing: AzureLLMPricingConfig) -> tuple[bool, str]:
    if not pricing.configured:
        return False, "PRICING_NOT_CONFIGURED"
    required = {
        "trusted_evaluation": True,
        "llm_candidate_selection_precision": 0.995,
        "overall_accepted_precision": 0.995,
        "critical_false_accepts": 0,
        "mean_paid_ai_cost_per_page_usd": 0.001,
        "authoritative_conflicts_overridden": 0,
        "novel_values_accepted": 0,
    }
    passed = (
        metrics.get("trusted_evaluation") is required["trusted_evaluation"]
        and (metrics.get("llm_candidate_selection_precision") or 0)
        >= required["llm_candidate_selection_precision"]
        and (metrics.get("overall_accepted_precision") or 0)
        >= required["overall_accepted_precision"]
        and metrics.get("critical_false_accepts") == 0
        and (metrics.get("mean_paid_ai_cost_per_page_usd") or float("inf"))
        <= required["mean_paid_ai_cost_per_page_usd"]
        and metrics.get("authoritative_conflicts_overridden") == 0
        and metrics.get("novel_values_accepted") == 0
    )
    return passed, "PASS" if passed else "PROMOTION_GATES_NOT_MET"


def build_azure_adjudication_provider(
    config: LLMAdjudicationConfig,
    pricing: AzureLLMPricingConfig,
    *,
    http_client: ResponseClient | None = None,
) -> AzureOpenAIAdjudicationProvider:
    """Use an environment key when supplied, otherwise Azure managed identity."""
    token_provider: Callable[[], str] | None = None
    if config.enabled and not config.api_key:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise LLMConfigurationError(
                "Azure managed identity requested but azure-identity is unavailable"
            ) from exc
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
    return AzureOpenAIAdjudicationProvider(
        config, pricing, http_client=http_client, token_provider=token_provider
    )
