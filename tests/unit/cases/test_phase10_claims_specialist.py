from evaluation.phase10_specialist_shadow import enrich_prediction
from packages.claims_specialist.cms import (
    normalize_date,
    normalize_npi,
    validate_diagnosis,
    validate_npi,
    validate_procedure,
)
from packages.claims_specialist.fusion import FusionCandidate, fuse
from packages.claims_specialist.handwriting import assess


def test_cms_field_validators_cover_high_value_fields():
    assert normalize_date("03/13/2016") == "2016-03-13"
    assert validate_diagnosis("F90.2")
    assert validate_procedure("99214")
    assert normalize_npi("123-456-7890") == "1234567890"
    assert not validate_npi("1234567890")


def test_fusion_prefers_valid_independently_corroborated_candidate():
    result = fuse(
        "diagnosis",
        [
            FusionCandidate("F90.2", "rapidocr", 0.91, "full-page-rapid"),
            FusionCandidate("F902", "regional_ocr", 0.94, "roi-crop-1"),
            FusionCandidate("F90.2", "paddleocr", 0.88, "full-page-paddle"),
        ],
    )
    assert result.value == "F90.2"
    assert result.valid is True
    assert result.independent_lineage_count == 2
    assert "INDEPENDENT_LINEAGE_CORROBORATION" in result.reason_codes
    assert not hasattr(result, "disposition")
    assert not hasattr(result, "accepted")


def test_handwriting_policy_requests_fallback_only_when_uncertainty_is_high():
    assessment = assess(
        ocr_confidence=0.42,
        candidate_disagreement=True,
        region_quality=0.45,
        recognized_character_ratio=0.50,
        explicit_handwriting_signal=0.85,
    )
    assert assessment.suspected
    assert assessment.request_multimodal_candidate
    assert not hasattr(assessment, "disposition")


def test_shadow_enrichment_never_allows_production_mutation():
    prediction = {
        "document_id": "doc-1",
        "route": "CMS1500",
        "fields": {
            "diagnosis": {
                "value": "F90.2",
                "confidence": 0.91,
                "decision": {"disposition": "HITL_REQUIRED"},
            }
        },
    }
    shadow = enrich_prediction(prediction)
    assert shadow["production_mutation_allowed"] is False
    assert shadow["fields"]["diagnosis"]["production_mutation_allowed"] is False
    assert prediction["fields"]["diagnosis"]["decision"]["disposition"] == "HITL_REQUIRED"
