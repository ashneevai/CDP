from packages.candidate_fusion.service import CandidateFusionService, CandidateObservation
from packages.classification_v5.contracts import ClassificationSignal
from packages.classification_v5.service import ClassificationV5Service
from packages.extraction_providers.contracts import ProviderCapability
from packages.extraction_providers.routing import ExtractionRoutingPolicy, RouteCandidate
from packages.promotion_gates import PromotionGate, PromotionMetrics


def test_classification_v5_fails_closed_when_margin_is_low():
    service = ClassificationV5Service(min_confidence=0.80, min_margin=0.15)
    result = service.classify(
        "p1",
        [
            ClassificationSignal(source="layout", label="CMS_1500", confidence=0.91, lineage_id="pixels"),
            ClassificationSignal(source="text", label="UB_04", confidence=0.88, lineage_id="ocr"),
        ],
    )
    assert result.requires_review is True
    assert result.route == "SAFE_GENERIC"


def test_candidate_fusion_deduplicates_same_lineage():
    service = CandidateFusionService()
    result = service.fuse(
        [
            CandidateObservation("123", "rapid", "same_pixels", 0.90),
            CandidateObservation("123", "paddle", "same_pixels", 0.95),
            CandidateObservation("123", "reference", "registry", 0.99, reference_support=1.0),
        ]
    )
    assert result[0].lineages == ("registry", "same_pixels")
    assert "INDEPENDENT_LINEAGE_AGREEMENT" in result[0].reason_codes


def test_provider_routing_respects_privacy_and_budget():
    ranked = ExtractionRoutingPolicy().rank(
        [
            RouteCandidate("local", frozenset({ProviderCapability.OCR}), 0.94, 500, 0.001, False),
            RouteCandidate("cloud", frozenset({ProviderCapability.OCR}), 0.99, 700, 0.02, True),
        ],
        required=ProviderCapability.OCR,
        privacy_allows_cloud=False,
        latency_budget_ms=1000,
        cost_budget_usd=0.03,
    )
    assert [item.provider for item in ranked] == ["local"]


def test_promotion_gate_blocks_critical_false_accept():
    decision = PromotionGate().evaluate(
        PromotionMetrics(
            accepted_precision=1.0,
            critical_field_precision=1.0,
            critical_false_accepts=1,
            field_hitl_rate=0.10,
            claim_hitl_rate=0.20,
            claim_stp_rate=0.80,
            p95_seconds_per_page=2.0,
            cost_usd_per_page=0.01,
            sample_size=1000,
        )
    )
    assert decision.promote is False
    assert "CRITICAL_FALSE_ACCEPTS_PRESENT" in decision.reasons


def test_new_foundations_do_not_expose_field_decision_methods():
    assert not hasattr(CandidateFusionService(), "decide")
    assert not hasattr(ClassificationV5Service(), "decide")
