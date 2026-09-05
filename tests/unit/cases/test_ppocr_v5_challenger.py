from __future__ import annotations

import asyncio
from dataclasses import replace

from PIL import Image

from packages.domain.common import BoundingBox
from packages.domain.enums import ClaimFormType
from packages.ocr.adjudication import adjudicate_candidates
from packages.ocr.contracts import OCRCandidate, OCRRequest
from packages.ocr.execution import OCRExecutionService
from packages.ocr.ppocr_v5_provider import (
    PPOCRv5Provider,
    UnsafeChallengerScopeError,
)
from packages.ocr.provenance import EvidenceProvenance
from packages.ocr.source_b_routing import ChallengerBudget

BOX = BoundingBox(x0=10, y0=20, x1=110, y1=60, image_width=1000, image_height=1000)


def request(scope="FIELD_CROP"):
    return OCRRequest(
        document_id="doc",
        page_number=1,
        field_name="billing_provider_npi",
        field_type="code",
        form_type=ClaimFormType.CMS1500,
        image=Image.new("RGB", (100, 40), "white"),
        bounding_box=BOX,
        scope=scope,
    )


def candidate(value, engine, invocation, *, crop="same-crop", region="same-region"):
    return OCRCandidate(
        value=value,
        raw_value=value or "",
        engine=engine,
        model_name=engine,
        model_version="1",
        preprocessing_variant="SOURCE_CROP",
        raw_confidence=0.99,
        calibrated_confidence=None,
        bounding_box=BOX,
        latency_ms=1,
        provenance=EvidenceProvenance(
            crop_sha256=crop,
            localization_region_id=region,
            invocation_id=invocation,
            engine_name=engine,
            source_candidate_id=invocation,
            bbox=BOX,
        ),
    )


def test_small_blocker_bundle_gets_one_challenge():
    budget = ChallengerBudget(1)
    assert budget.limit == 1
    assert budget.claim()
    assert not budget.claim()


def test_provider_preserves_tokens_and_execution_provenance():
    def backend(_image):
        return [
            {
                "res": {
                    "rec_texts": ["1234567893"],
                    "rec_scores": [0.93],
                    "rec_polys": [[[0, 0], [99, 0], [99, 39], [0, 39]]],
                }
            }
        ]

    result = asyncio.run(
        OCRExecutionService(benchmark_mode=True).execute(
            PPOCRv5Provider(backend=backend), request()
        )
    )
    assert result.candidates[0].tokens[0].text == "1234567893"
    assert result.candidates[0].tokens[0].confidence == 0.93
    assert result.candidates[0].provenance is not None


def test_provider_rejects_non_field_crop_before_backend_call():
    called = False

    def backend(_image):
        nonlocal called
        called = True
        return []

    try:
        asyncio.run(PPOCRv5Provider(backend=backend).extract(request("FULL_PAGE")))
    except UnsafeChallengerScopeError:
        pass
    else:
        raise AssertionError("unsafe challenger scope was not rejected")
    assert not called


def test_same_crop_engine_diversity_cannot_replace_primary_as_independent_evidence():
    result = adjudicate_candidates(
        field_name="billing_provider_npi",
        primary=candidate("123", "rapidocr", "primary"),
        challenger=candidate("1234567893", "ppocr-v5", "challenger"),
        crop_safety_status="CROP_SAFE",
    )
    assert result.action == "HITL"
    assert result.reason == "CHALLENGER_ACCEPTANCE_GATES_FAILED"
    assert not result.evidence_independent


def test_distinct_source_observation_can_replace_invalid_primary():
    result = adjudicate_candidates(
        field_name="billing_provider_npi",
        primary=candidate("123", "rapidocr", "primary", crop="crop-a", region="region-a"),
        challenger=candidate(
            "1234567893", "ppocr-v5", "challenger", crop="crop-b", region="region-b"
        ),
        crop_safety_status="CROP_SAFE",
    )
    assert result.action == "USE_CHALLENGER"
    assert result.evidence_independent


def test_materially_disagreeing_valid_candidates_fail_closed_to_hitl():
    result = adjudicate_candidates(
        field_name="member_id",
        primary=candidate("ABCDE1", "rapidocr", "primary"),
        challenger=candidate("ABCDE2", "ppocr-v5", "challenger"),
        crop_safety_status="CROP_SAFE",
    )
    assert result.action == "HITL"
    assert result.agreement_status == "DISAGREE"


def test_confidence_cannot_override_unsafe_localization():
    high_confidence = replace(candidate("1234567893", "ppocr-v5", "challenger"), raw_confidence=1)
    result = adjudicate_candidates(
        field_name="billing_provider_npi",
        primary=candidate("123", "rapidocr", "primary"),
        challenger=high_confidence,
        crop_safety_status="UNCERTAIN",
    )
    assert result.action == "HITL"
    assert result.reason == "LOCALIZATION_NOT_CROP_SAFE"
