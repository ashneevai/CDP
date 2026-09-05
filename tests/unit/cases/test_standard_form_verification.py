import pytest

from packages.document_routing.decision_service import DocumentRoutingDecisionService
from packages.document_taxonomy.contracts import DocumentClassification
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.processing_routes.contracts import ProcessingRoute
from packages.standard_form_verification.contracts import (
    StandardFormStatus,
    StandardFormVerification,
)
from packages.standard_form_verification.evidence import (
    AnchorRegionEvidence,
    StandardFormEvidence,
)


def _classification(family: DocumentClass, *, structured=True):
    return DocumentClassification(
        document_id="d",
        page_id="p",
        top_level_class=DocumentClass.CLAIM,
        document_family=DocumentClass.STANDARD_CLAIM,
        document_subtype=family,
        structured=structured,
        claim_related=True,
        standard_candidate=True,
        confidence=0.99,
        supporting_evidence=("NOMINATION",),
        classifier_version="test",
    )


def _anchor(family, anchor, region, anchor_class="HIGH_DISCRIMINATION"):
    return AnchorRegionEvidence(
        family=family,
        anchor=anchor,
        region_id=region,
        anchor_class=anchor_class,
        match_type="EXACT",
        ocr_confidence=0.95,
        geometry_score=0.9,
        phrase_score=1.0,
    )


def _cms_evidence():
    family = DocumentClass.CMS1500
    observations = (
        _anchor(family, "health insurance claim form", "identity_header"),
        _anchor(family, "insured id number", "insured_identity"),
        _anchor(family, "diagnosis or nature of illness", "diagnosis"),
        _anchor(family, "federal tax id", "provider_billing"),
    )
    return StandardFormEvidence(
        candidate_family=family,
        page_geometry_score=0.9,
        region_layout_scores={item.region_id: item.geometry_score for item in observations},
        service_grid_score=0.9,
        structure_score=0.9,
        standard_score=0.9,
        family_margin=0.5,
        high_value_anchor_score=0.9,
        high_value_anchor_count=4,
        independent_region_count=4,
        spatial_relationship_score=0.9,
        canonical_identity_confirmed=True,
        identity_status="CONFIRMED",
        identity_family=family,
        authorization_path="EXPLICIT_IDENTITY",
        matched_identity_anchors=("cms 1500",),
        per_anchor_evidence=observations,
    )


def _ub_evidence():
    family = DocumentClass.UB04
    observations = (
        _anchor(family, "type of bill", "institutional_header"),
        _anchor(family, "patient control", "patient_control"),
        _anchor(family, "statement covers", "statement_period"),
        _anchor(family, "revenue code", "revenue_service"),
    )
    return StandardFormEvidence(
        candidate_family=family,
        page_geometry_score=0.9,
        region_layout_scores={item.region_id: item.geometry_score for item in observations},
        service_grid_score=0.9,
        structure_score=0.9,
        standard_score=0.9,
        family_margin=0.5,
        high_value_anchor_score=0.9,
        high_value_anchor_count=4,
        independent_region_count=4,
        spatial_relationship_score=0.9,
        repeating_row_score=0.9,
        canonical_identity_confirmed=True,
        identity_status="CONFIRMED",
        identity_family=family,
        authorization_path="EXPLICIT_IDENTITY",
        matched_identity_anchors=("ub 04",),
        per_anchor_evidence=observations,
    )


@pytest.mark.parametrize(
    ("family", "evidence", "route"),
    [
        (DocumentClass.CMS1500, _cms_evidence(), ProcessingRoute.CMS_STANDARD_EXTRACTOR),
        (DocumentClass.UB04, _ub_evidence(), ProcessingRoute.UB_STANDARD_EXTRACTOR),
    ],
)
def test_fixed_extractor_requires_family_specific_verification(family, evidence, route):
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(family), evidence
    )
    assert decision.standard_verification.status == StandardFormStatus.VERIFIED
    assert decision.standard_verification.form_identity.family == family
    assert decision.processing_route == route


@pytest.mark.parametrize("status", [StandardFormStatus.NOT_VERIFIED, StandardFormStatus.AMBIGUOUS])
def test_non_verified_contract_cannot_claim_fixed_extractor_eligibility(status):
    with pytest.raises(ValueError):
        StandardFormVerification(
            candidate_family=DocumentClass.CMS1500,
            status=status,
            verification_score=0.5,
            eligible_for_fixed_extractor=True,
        )


def test_verified_contract_without_identity_cannot_claim_eligibility():
    with pytest.raises(ValueError):
        StandardFormVerification(
            candidate_family=DocumentClass.CMS1500,
            status=StandardFormStatus.VERIFIED,
            verification_score=1.0,
            eligible_for_fixed_extractor=True,
        )


def test_missing_verification_fails_closed_and_preserves_structure():
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(DocumentClass.CMS1500)
    )
    assert decision.standard_verification.status == StandardFormStatus.NOT_VERIFIED
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_visual_probability_alone_cannot_verify_standard_form():
    evidence = StandardFormEvidence(candidate_family=DocumentClass.UB04, visual_probability=1.0)
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(DocumentClass.UB04), evidence
    )
    assert decision.standard_verification.status == StandardFormStatus.NOT_VERIFIED
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_cms_ub_contradiction_blocks_unsafe_verification():
    evidence = _cms_evidence().model_copy(
        update={"contradiction_codes": ("UB_INSTITUTIONAL_GRID",)}
    )
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(DocumentClass.CMS1500), evidence
    )
    assert decision.standard_verification.status == StandardFormStatus.NOT_VERIFIED
    assert "CMS_UB_CONTRADICTION" in decision.standard_verification.reason_codes
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_ub_hard_negative_does_not_trade_precision_for_recall():
    evidence = _ub_evidence().model_copy(
        update={"contradiction_codes": ("CMS_FIELD_CONSTELLATION",)}
    )
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(DocumentClass.UB04), evidence
    )
    assert decision.standard_verification.status == StandardFormStatus.NOT_VERIFIED
    assert "UB_CMS_CONTRADICTION" in decision.standard_verification.reason_codes
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_runtime_and_evaluation_use_identical_policy_results():
    service = DocumentRoutingDecisionService()
    classification = _classification(DocumentClass.UB04)
    runtime = service.decide_classification(classification, _ub_evidence(), evaluation_only=False)
    evaluation = service.decide_classification(classification, _ub_evidence(), evaluation_only=True)
    assert runtime.classification == evaluation.classification
    assert runtime.standard_verification == evaluation.standard_verification
    assert runtime.processing_route == evaluation.processing_route


def test_geometry_and_layout_without_canonical_identity_cannot_verify():
    evidence = _ub_evidence().model_copy(
        update={
            "canonical_identity_confirmed": False,
            "identity_status": "UNKNOWN",
            "identity_family": None,
        }
    )
    decision = DocumentRoutingDecisionService().decide_classification(
        _classification(DocumentClass.UB04), evidence
    )
    assert decision.standard_verification.status == StandardFormStatus.AMBIGUOUS
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_verified_family_mismatch_is_blocked_at_route_resolver():
    verification = DocumentRoutingDecisionService().verification_service.verify(_ub_evidence())
    route = DocumentRoutingDecisionService().route_resolver.resolve(
        _classification(DocumentClass.CMS1500), verification
    )
    assert route.route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR
    assert route.reason_codes == ("STANDARD_IDENTITY_CLASSIFICATION_MISMATCH",)


def test_classification_contradiction_blocks_otherwise_verified_identity():
    classification = _classification(DocumentClass.CMS1500).model_copy(
        update={"contradicting_evidence": ("NONCANONICAL_REFERENCE",)}
    )
    decision = DocumentRoutingDecisionService().decide_classification(
        classification, _cms_evidence()
    )
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR


def test_nonstandard_claim_nomination_uses_generic_structured_route():
    decision = DocumentRoutingDecisionService().decide_nomination(
        document_id="d",
        page_id="p",
        nominated_family=None,
        structured=True,
        claim_related=True,
        non_claim=False,
        confidence=0.8,
        supporting_evidence=("CLAIM_FORM_NONCANONICAL",),
    )
    assert decision.classification.document_subtype == DocumentClass.OTHER_CLAIM_FORM
    assert not decision.classification.standard_candidate
    assert decision.processing_route == ProcessingRoute.LAYOUT_STRUCTURED_EXTRACTOR
