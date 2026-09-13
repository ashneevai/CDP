from __future__ import annotations

import pytest

from packages.criticality import CriticalityLevel
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.domain.common import BoundingBox
from packages.domain.enums import ExtractionMethod
from packages.domain.extraction import ExtractedField, FieldEvidence
from packages.evidence import (
    EvidenceClass,
    EvidencePolicy,
    StructuralLocalizationEvidence,
    StructuralLocalizationType,
    build_evidence_bundle,
)
from packages.evidence_decision.adapters import ocr_candidates_from_field
from packages.evidence_decision import DecisionContext, EvidenceDecisionService
from packages.evidence_dependency import DependencyRelation, EvidenceDependencyService
from packages.extraction_geometry import (
    ExtractionGeometryDecision,
    ExtractionGeometryMode,
    FormIdentityDecision,
    FormIdentityStatus,
)
from packages.field_localization import DynamicROIResolver
from packages.ocr.contracts import OCRCandidate
from packages.ocr.provenance import EvidenceProvenance
from packages.domain.registration import RegistrationEvidence
from packages.template_compatibility import (
    TemplateCompatibilityEvidence,
    TemplateCompatibilityStatus,
)
from workers.validation.consumer import qualified_structural_localization

BOX = BoundingBox(x0=10, y0=10, x1=110, y1=40, image_width=1000, image_height=1200)


def provenance(tag: str, *, shared: bool = False) -> EvidenceProvenance:
    suffix = "shared" if shared else tag
    return EvidenceProvenance(
        page_sha256="page",
        source_representation_id=f"representation-{suffix}",
        observation_id=f"observation-{suffix}",
        crop_sha256=f"crop-{suffix}",
        localization_id=f"localization-{suffix}",
        localization_method=f"method-{suffix}",
        preprocessing_profile=f"profile-{suffix}",
        preprocessing_version="1",
        engine_family=f"{tag.upper()}_FAMILY",
        engine_name=tag,
        model_family=f"{tag.upper()}_MODEL",
        model_name=tag,
        model_version="1",
        source_candidate_id=f"candidate-{tag}",
        bbox=BOX if shared else BoundingBox(
            x0=10 if tag == "rapid" else 300,
            y0=10,
            x1=110 if tag == "rapid" else 400,
            y1=40,
            image_width=1000,
            image_height=1200,
        ),
    )


def candidate(engine: str, value: str, *, shared: bool) -> OCRCandidate:
    return OCRCandidate(
        value=value,
        raw_value=value,
        engine=engine,
        model_name=engine,
        model_version="1",
        preprocessing_variant="same" if shared else engine,
        raw_confidence=.99,
        calibrated_confidence=None,
        bounding_box=BOX,
        latency_ms=1,
        provenance=provenance(engine, shared=shared),
    )


def structure(field: str) -> StructuralLocalizationEvidence:
    return StructuralLocalizationEvidence(
        evidence_type=StructuralLocalizationType.ANCHOR_RELATIVE_LOCALIZATION_CONFIRMED,
        confidence=.99,
        confirmed=True,
        reason_codes=("BOUNDED_ALIAS_MATCH", "OBSERVED_VALUE_SPAN_GEOMETRY"),
        source="test",
        field_name=field,
        field_bbox=(10, 10, 110, 40),
        localization_mode="ANCHOR_RELATIVE",
        positive_bounded_roi=True,
        geometry_valid=True,
    )


def bundle(field: str, value: str, fact: str, *, shared: bool):
    return build_evidence_bundle(
        field_name=field,
        candidates=[candidate("rapid", value, shared=shared), candidate("tesseract", value, shared=shared)],
        registration_confidence=.99,
        wrong_crop_suspected=False,
        deterministic_evidence={fact},
        hard_validation_passed=True,
        structural_localization=structure(field),
    )


def test_engine_difference_alone_is_not_independence_and_missing_is_unknown():
    service = EvidenceDependencyService()
    assert service.classify(None, provenance("rapid")).relation is DependencyRelation.UNKNOWN
    same = service.classify(provenance("rapid", shared=True), provenance("paddle", shared=True))
    assert same.relation is DependencyRelation.CORRELATED
    assert same.dependency_dimensions["same_crop_hash"] is True


def test_distinct_representation_crop_preprocessing_and_engine_can_be_independent():
    result = EvidenceDependencyService().classify(provenance("rapid"), provenance("tesseract"))
    assert result.relation is DependencyRelation.INDEPENDENT


@pytest.mark.parametrize(
    ("field", "value", "weak_fact"),
    [
        ("member_id", "AB798842", "FORMAT_VALID"),
        ("total_charge", "1234.50", "FORMAT_VALID"),
        ("patient_name", "JOAN SMITH", "NAME_TOKEN_BOUNDARIES_VALID"),
        ("provider_npi", "1992999991", "CHECKSUM_VALID"),
    ],
)
def test_correlated_valid_looking_false_agreement_cannot_satisfy_critical_policy(
    field, value, weak_fact
):
    evidence = bundle(field, value, weak_fact, shared=True)
    agreement = next(item for item in evidence.items if item.evidence_class is EvidenceClass.E2)
    assert agreement.evidence_type == "OCR_AGREEMENT_CORRELATED"
    satisfied, _, _, _ = EvidencePolicy.load().evaluate(
        field, CriticalityLevel.C2, evidence, "CMS1500"
    )
    assert not satisfied


def test_genuinely_independent_npi_agreement_with_checksum_and_field_e3_is_eligible():
    evidence = bundle("provider_npi", "1992999991", "CHECKSUM_VALID", shared=False)
    agreement = next(item for item in evidence.items if item.evidence_class is EvidenceClass.E2)
    assert agreement.evidence_type == "OCR_AGREEMENT_INDEPENDENT"
    satisfied, available, _, _ = EvidencePolicy.load().evaluate(
        "provider_npi", CriticalityLevel.C2, evidence, "CMS1500"
    )
    assert satisfied
    assert set(available) >= {"E2", "E3", "E4"}


def test_old_field_evidence_loads_and_runtime_preserves_candidate_confidences_and_provenance():
    old = FieldEvidence.model_validate({
        "source": "REGIONAL_RAPIDOCR", "raw_text": "A", "confidence": .31
    })
    assert old.provenance is None
    field = ExtractedField(
        field_name="member_id", raw_value="B", normalized_value="B", confidence=.91,
        page_number=1, bounding_box=BOX, extraction_method=ExtractionMethod.REGIONAL_RAPIDOCR,
        candidates=[
            FieldEvidence(source=ExtractionMethod.REGIONAL_RAPIDOCR, raw_text="A", confidence=.31,
                          provenance=provenance("rapid")),
            FieldEvidence(source=ExtractionMethod.ALTERNATE_PREPROCESS_OCR, raw_text="B", confidence=.87,
                          provenance=provenance("tesseract")),
        ],
    )
    rebuilt = ocr_candidates_from_field(field)
    assert [item.raw_confidence for item in rebuilt] == [.31, .87]
    assert rebuilt[0].preprocessing_variant == "profile-rapid"
    assert rebuilt[1].provenance.crop_sha256 == "crop-tesseract"


def test_runtime_candidate_preserves_selected_normalized_span_and_raw_audit_text():
    field = ExtractedField(
        field_name="diagnosis", raw_value="Z30.0", normalized_value="Z30.0",
        confidence=.98, page_number=1, bounding_box=BOX,
        extraction_method=ExtractionMethod.REGIONAL_RAPIDOCR,
        candidates=[FieldEvidence(
            source=ExtractionMethod.REGIONAL_RAPIDOCR,
            raw_text="Z30.0 .50", confidence=.98, provenance=provenance("rapid"),
        )],
    )
    candidate = ocr_candidates_from_field(field)[0]
    assert candidate.value == "Z30.0"
    assert candidate.raw_value == "Z30.0 .50"


def test_runtime_candidate_rebuilds_persisted_token_geometry():
    field = ExtractedField(
        field_name="patient_name",
        raw_value="DOE, JANE",
        normalized_value="DOE, JANE",
        confidence=.98,
        page_number=1,
        bounding_box=BOX,
        extraction_method=ExtractionMethod.REGIONAL_RAPIDOCR,
        candidates=[FieldEvidence(
            source=ExtractionMethod.REGIONAL_RAPIDOCR,
            raw_text="DOE, JANE",
            confidence=.98,
            bounding_box=BOX,
            provenance=provenance("rapid"),
            tokens=({
                "text": "DOE, JANE",
                "confidence": .98,
                "bounding_box": BOX.model_dump(mode="json"),
            },),
        )],
    )
    persisted = ExtractedField.model_validate_json(field.model_dump_json())
    candidate = ocr_candidates_from_field(persisted)[0]
    assert candidate.tokens[0].text == "DOE, JANE"
    assert candidate.tokens[0].bounding_box.x0 == BOX.x0
    assert candidate.tokens[0].bounding_box.image_width == BOX.image_width


def _fixed_geometry(status=FormIdentityStatus.VERIFIED):
    identity = FormIdentityDecision(family=DocumentClass.CMS1500, status=status, score=.99)
    compatibility = TemplateCompatibilityEvidence(
        family="CMS1500", family_compatibility=1, aspect_ratio_similarity=1,
        line_structure_similarity=1, edge_projection_similarity=1, anchor_visibility=1,
        normalized_layout_similarity=1, form_fingerprint_similarity=1,
        compatibility_score=1, status=TemplateCompatibilityStatus.COMPATIBLE,
    )
    registration = RegistrationEvidence(
        algorithm="SIFT", accepted=True, corner_validity=True, alignment_confidence=.95
    )
    return ExtractionGeometryDecision(
        mode=ExtractionGeometryMode.REGISTERED_FIXED, form_identity=identity,
        template_id="cms1500", template_version="1", compatibility=compatibility,
        registration=registration, transformed_geometry_valid=True,
    )


def test_registered_template_is_third_priority_and_requires_valid_registration():
    result = DynamicROIResolver().resolve(
        "member_id", anchor=None, structural=None, geometry=_fixed_geometry(),
        registered_template_bbox=(10, 10, 100, 30),
    )
    assert result.mode.value == "FIXED_REGISTERED"
    assert result.reason_codes == ("DYNAMIC_PRIORITY_3_TEMPLATE_FAST_PATH",)
    assert result.field_structural_confidence == .95

    with pytest.raises(ValueError, match="VERIFIED_FORM_IDENTITY"):
        _fixed_geometry(FormIdentityStatus.AMBIGUOUS)


def test_field_e3_uses_field_confidence_not_high_page_average():
    payload = {
        "page_number": 1,
        "extraction_geometry": {
            "mode": "ANCHOR_RELATIVE",
            "structural_confidence": .99,
            "form_identity": {"status": "VERIFIED"},
        },
        "roi_resolution": {
            "patient_name": {
                "bbox": [10, 10, 100, 30],
                "field_structural_confidence": .20,
                "reason_codes": [
                    "DYNAMIC_PRIORITY_1_ANCHOR", "BOUNDED_ALIAS_MATCH",
                    "OBSERVED_VALUE_SPAN_GEOMETRY",
                ],
            }
        },
    }
    evidence = qualified_structural_localization(payload, 1, "patient_name")
    assert evidence.confidence == .20
    assert not evidence.confirmed


def test_persisted_runtime_and_direct_evaluation_have_identical_disposition():
    original = [
        candidate("rapid", "1992999991", shared=False),
        candidate("tesseract", "1992999991", shared=False),
    ]
    persisted = ExtractedField(
        field_name="provider_npi", raw_value="1992999991", normalized_value="1992999991",
        confidence=.99, page_number=1, bounding_box=BOX,
        extraction_method=ExtractionMethod.REGIONAL_RAPIDOCR,
        candidates=[
            FieldEvidence(
                source=(ExtractionMethod.REGIONAL_RAPIDOCR if item.engine == "rapid"
                        else ExtractionMethod.ALTERNATE_PREPROCESS_OCR), raw_text=item.raw_value,
                confidence=item.raw_confidence, bounding_box=BOX, model_name=item.model_name,
                model_version=item.model_version, provenance=item.provenance,
            )
            for item in original
        ],
    )
    common = dict(
        field_name="provider_npi", document_family="CMS1500",
        criticality=CriticalityLevel.C2, hard_validation_passed=True,
        deterministic_evidence={"CHECKSUM_VALID"}, structural_localization=structure("provider_npi"),
    )
    service = EvidenceDecisionService()
    direct = service.decide(DecisionContext(candidates=original, **common))
    runtime = service.decide(
        DecisionContext(candidates=ocr_candidates_from_field(persisted), **common)
    )
    assert runtime.disposition == direct.disposition
    assert runtime.selected_value == direct.selected_value


def test_cross_field_plausibility_is_not_field_identity_confirmation():
    weak = build_evidence_bundle(
        field_name="patient_name",
        candidates=[candidate("rapid", "WRONG NAME", shared=False)],
        registration_confidence=.99,
        wrong_crop_suspected=False,
        deterministic_evidence={"FORMAT_VALID"},
        hard_validation_passed=True,
        structural_localization=structure("patient_name"),
        cross_field_evidence={"MEMBER_RELATIONSHIP_CONFIRMED"},
    )
    strong = build_evidence_bundle(
        field_name="patient_name",
        candidates=[candidate("rapid", "RIGHT NAME", shared=False)],
        registration_confidence=.99,
        wrong_crop_suspected=False,
        deterministic_evidence={"FORMAT_VALID"},
        hard_validation_passed=True,
        structural_localization=structure("patient_name"),
        cross_field_evidence={"MULTI_ATTRIBUTE_IDENTITY_CONFIRMED"},
    )
    policy = EvidencePolicy.load()
    assert not policy.evaluate("patient_name", CriticalityLevel.C2, weak, "CMS1500")[0]
    assert policy.evaluate("patient_name", CriticalityLevel.C2, strong, "CMS1500")[0]
