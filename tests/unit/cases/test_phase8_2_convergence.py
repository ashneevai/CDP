from pathlib import Path

from PIL import Image

from packages.document_taxonomy.taxonomy import DocumentClass
from packages.deterministic_evidence import DeterministicEvidenceService
from packages.evidence_decision import FieldDecision, FieldDisposition, NextAction
from packages.extraction_geometry import FormIdentityDecision, FormIdentityStatus
from packages.hitl_optimization import CanonicalHITLAuthority
from packages.page_observation import PageObservationService, line_clustered_reading_order
from packages.local_evidence_cascade import decide_local_candidate
from packages.templates import TemplateRegistry
from workers.page_detection.text_extraction import TextLine
from workers.standard_form_extraction import (
    StandardFormExtractionService,
    StandardFormProcessingService,
)
from workers.validation.consumer import extraction_geometry_evidence


ROOT = Path(__file__).resolve().parents[3]


class FixtureOCR:
    model_version = "fixture-v1"

    def __init__(self, lines):
        self.lines = lines
        self.full_page_calls = 0
        self.region_calls = 0

    def extract(self, _image):
        self.full_page_calls += 1
        return self.lines

    def extract_region(self, _image, x0, y0, x1, y1):
        self.region_calls += 1
        return [line for line in self.lines if x0 <= (line.x0+line.x1)/2 <= x1
                and y0 <= (line.y0+line.y1)/2 <= y1]


def test_line_clustering_orders_same_line_left_to_right_despite_y_jitter():
    values = [
        TextLine("SMITH", 150, 185, 230, 210, .99),
        TextLine("MARIA", 70, 186, 145, 211, .99),
        TextLine("JOHN", 70, 240, 130, 265, .99),
        TextLine("PATEL", 135, 239, 205, 264, .99),
    ]

    assert [item.text for item in line_clustered_reading_order(values)] == [
        "MARIA", "SMITH", "JOHN", "PATEL",
    ]


def test_runtime_and_evaluation_use_same_canonical_candidate_service():
    lines = [
        TextLine("PATIENT NAME", 75, 145, 250, 170, .99),
        TextLine("MARIA", 78, 190, 145, 208, .99),
        TextLine("SMITH", 150, 189, 225, 207, .99),
    ]
    ocr = FixtureOCR(lines)
    observations = PageObservationService(ocr, preprocessing_version="fixture-v1")
    extraction = StandardFormExtractionService(ocr)
    service = StandardFormProcessingService(observations, extraction)
    image = Image.new("RGB", (1600, 1800), "white")
    identity = FormIdentityDecision(
        family=DocumentClass.CMS1500, status=FormIdentityStatus.VERIFIED, score=.99,
    )
    template = TemplateRegistry.load_from_directory().get("cms1500", "02-12")

    runtime = service.process(image, template, 1, identity, page_id="runtime")
    evaluation = service.process(
        image, template, 1, identity, page_id="evaluation", observation=runtime.observation,
    )

    assert [
        (field.field_name, field.raw_value, field.normalized_value, field.bounding_box,
         field.validation_status, field.validation_reasons)
        for field in runtime.fields
    ] == [
        (field.field_name, field.raw_value, field.normalized_value, field.bounding_box,
         field.validation_status, field.validation_reasons)
        for field in evaluation.fields
    ]
    assert runtime.roi_results == evaluation.roi_results
    assert ocr.full_page_calls == 1


def test_dynamic_geometry_evidence_is_not_treated_as_missing_registration():
    confidence, source = extraction_geometry_evidence({
        "page_number": 2,
        "extraction_geometry": {
            "mode": "ANCHOR_RELATIVE", "structural_confidence": .91,
        },
    }, 2)

    assert confidence == .91
    assert source == "DYNAMIC_GEOMETRY:ANCHOR_RELATIVE"


def test_legacy_hitl_projection_cannot_override_canonical_field_decision():
    decision = FieldDecision(
        field_name="member_id", disposition=FieldDisposition.ESCALATE,
        calibrated_probability=.99, next_action=NextAction.HUMAN_REVIEW,
        policy_version="fixture", reason_codes=["INSUFFICIENT_EVIDENCE"],
    )

    summary = CanonicalHITLAuthority.summarize(decision)

    assert summary.review_required is True
    assert summary.disposition == FieldDisposition.ESCALATE.value


def test_production_workers_do_not_import_legacy_hitl_authority():
    offenders = []
    for path in (ROOT / "workers").rglob("*.py"):
        if "packages.hitl_optimization" in path.read_text("utf-8"):
            offenders.append(path)
    assert offenders == []


def test_runtime_worker_calls_canonical_phase8_processing_service():
    source = (ROOT / "workers/standard_form_extraction/consumer.py").read_text("utf-8")
    assert "self._processing_service.process" in source
    assert "StandardFormProcessingService" in source


def test_registered_field_labels_fail_hard_validation_as_values():
    service = DeterministicEvidenceService()

    assert service.evaluate("insured_name", "RENDERING PROVIDER").passed is False
    assert service.evaluate("relationship", "NPI").passed is False


def test_merged_provider_tokens_require_secondary_evidence():
    assert decide_local_candidate("MARIAPATEL MD", "PERSON_OR_ORGANIZATION").accepted is False
    assert decide_local_candidate("JOHN SMITHMD", "PERSON_OR_ORGANIZATION").accepted is False
    assert decide_local_candidate("RIVER VALLEYHOSPITAL", "PERSON_OR_ORGANIZATION").accepted is False
    assert decide_local_candidate("MARIA PATEL MD", "PERSON_OR_ORGANIZATION").accepted is True


def test_npi_does_not_repeat_same_engine_secondary_ocr():
    source = (ROOT / "workers/standard_form_extraction/extractor.py").read_text("utf-8")
    assert "secondary_eligible = False" in source
    assert "same-engine regional retries as NO_CHANGE" in source


def test_ub_semantic_headers_supply_skew_resilient_structure_confidence():
    from packages.forms.ub04.structural_map import UB04StructuralMapDetector
    from packages.page_observation import (
        ImageQualityEvidence, ObservationToken, PageObservation,
    )

    tokens = [
        ObservationToken(
            token_id=str(x), text=text,
            bbox=(x, 500 + x * .02, x + 50, 525 + x * .02),
            confidence=.99, line_index=0, reading_order=x,
        )
        for x, text in enumerate(
            ("REV", "DESCRIPTION", "HCPCS", "SERVICE DATE", "UNITS", "CHARGE"),
            start=100,
        )
    ]
    observation = PageObservation(
        page_id="skew", page_sha256="0" * 64, width=1400, height=1800,
        aspect_ratio=1400 / 1800,
        image_quality=ImageQualityEvidence(
            blur_score=1, contrast_score=1, foreground_ratio=.1,
            quality_bucket="GOOD",
        ),
        ocr_tokens=tokens, text_lines=(), word_boxes=(), horizontal_lines=(),
        vertical_lines=(), connected_components=(), checkbox_candidates=(),
        table_regions=(), anchor_candidates=(), structural_regions=(),
        ocr_model_version="test", preprocessing_version="test", full_page_ocr_calls=1,
    )

    structure = UB04StructuralMapDetector().detect(observation)

    assert structure.confidence >= .8
    assert "SERVICE_HEADERS_OBSERVED_6" in structure.reason_codes
