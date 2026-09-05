import pytest
from PIL import Image, ImageDraw

from packages.document_routing import DocumentRoutingDecisionService, MultiSignalRouter
from packages.extraction_routing import ExtractionTarget, extraction_target
from packages.processing_routes.contracts import ProcessingRoute
from workers.page_detection.text_extraction import TextLine

FIXED_ROUTES = {
    ProcessingRoute.CMS_STANDARD_EXTRACTOR,
    ProcessingRoute.UB_STANDARD_EXTRACTOR,
}


def _image():
    image = Image.new("L", (1000, 1300), 255)
    draw = ImageDraw.Draw(image)
    for y in range(150, 1100, 100):
        draw.line((40, y, 960, y), fill=0, width=2)
    for x in (40, 250, 500, 750, 960):
        draw.line((x, 150, x, 1100), fill=0, width=2)
    return image


def _line(text, x0, y0, x1=None, confidence=0.95):
    return TextLine(text, x0, y0, x1 or min(980, x0 + 420), y0 + 28, confidence)


def _cms_lines(identity="CMS-1500"):
    return [
        _line(identity, 20, 30, 500),
        _line("HEALTH INSURANCE CLAIM FORM", 20, 100, 500),
        _line("PATIENTS NAME", 40, 260, 300),
        _line("INSURED ID NUMBER", 600, 220, 920),
        _line("DIAGNOSIS OR NATURE OF ILLNESS", 40, 700, 520),
        _line("FEDERAL TAX ID", 600, 1050, 920),
    ]


def _ub_lines(identity="UB-04"):
    return [
        _line(identity, 20, 30, 400),
        _line("TYPE OF BILL", 760, 80, 960),
        _line("PATIENT CONTROL", 600, 100, 790),
        _line("STATEMENT COVERS", 650, 250, 900),
        _line("REVENUE CODE", 40, 420, 220),
        _line("HCPCS", 300, 430, 390),
        _line("UNITS", 650, 440, 720),
        _line("TOTAL CHARGES", 760, 450, 950),
        _line("PRINCIPAL DIAGNOSIS", 50, 900, 300),
    ]


def _full_chain(lines):
    routing = MultiSignalRouter.load().route(_image(), lines)
    decision = DocumentRoutingDecisionService().decide("document", "page", routing)
    target = (
        extraction_target(decision.processing_route)
        if decision.processing_route != ProcessingRoute.SAFE_UNKNOWN
        else None
    )
    return routing, decision, target


@pytest.mark.parametrize(
    "lines",
    [
        _cms_lines("NOT A CMS-1500 FORM"),
        _cms_lines("INSTRUCTIONS FOR COMPLETING CMS-1500"),
        _cms_lines("ATTACH TO CMS-1500"),
        _cms_lines("SUBMIT WITH CMS-1500"),
        [_line("EXPLANATION OF BENEFITS - ATTACH TO CMS-1500", 20, 30, 700), *_cms_lines()[1:]],
        [_line("COVER LETTER - SUBMIT WITH CMS-1500", 20, 30, 700), *_cms_lines()[1:]],
        _ub_lines("INSTRUCTIONS FOR COMPLETING UB-04"),
        [_line("PAYER SPECIFIC FORM CMS-1500", 20, 30, 600), *_cms_lines()[1:]],
        [_line("REIMBURSEMENT REQUEST CMS-1500", 20, 30, 600), *_cms_lines()[1:]],
        [_line("PROPRIETARY CLAIM FORM CMS-1500", 20, 30, 650), *_cms_lines()[1:]],
        [_line("LEGACY CLAIM FORM CMS-1500", 20, 30, 600), *_cms_lines()[1:]],
        [_line("CMS-1500 UB-04", 20, 30, 500), *_cms_lines()[1:], *_ub_lines()[1:]],
        [_line("CMS-1500", 20, 650, 500), *_cms_lines()[1:4]],
        [_line("CMS", 20, 30, 100), _line("1500", 800, 30, 900), _cms_lines()[2]],
        [_line("CMS-1500", 20, 30, 300), _cms_lines()[2]],
        [_line("PATIENT PROVIDER DIAGNOSIS CLAIM", 20, 200, 700)],
        [_line("INSTRUCTIONS FOR COMPLETING CMS 1500", 20, 30, 700), *_cms_lines()[1:]],
    ],
)
def test_adversarial_identity_mentions_never_reach_fixed_dispatch(lines):
    routing, decision, target = _full_chain(lines)
    assert decision.processing_route not in FIXED_ROUTES
    assert not routing.localization_allowed
    assert target not in {ExtractionTarget.CMS1500_STANDARD, ExtractionTarget.UB04_STANDARD}


@pytest.mark.parametrize(
    ("lines", "route", "target"),
    [
        (_cms_lines(), ProcessingRoute.CMS_STANDARD_EXTRACTOR, ExtractionTarget.CMS1500_STANDARD),
        (_ub_lines(), ProcessingRoute.UB_STANDARD_EXTRACTOR, ExtractionTarget.UB04_STANDARD),
        (
            _cms_lines("CMS 1500"),
            ProcessingRoute.CMS_STANDARD_EXTRACTOR,
            ExtractionTarget.CMS1500_STANDARD,
        ),
    ],
)
def test_genuine_canonical_forms_are_family_consistent_through_dispatch(lines, route, target):
    routing, decision, dispatched = _full_chain(lines)
    verification = decision.standard_verification
    assert decision.processing_route == route
    assert dispatched == target
    assert routing.localization_allowed
    assert verification is not None
    assert verification.eligible_for_fixed_extractor
    assert verification.candidate_family == decision.classification.document_subtype
    assert verification.form_identity.family == verification.candidate_family
    assert not verification.form_identity.contradiction_codes


def test_complete_topology_without_header_can_authorize_only_with_all_regions():
    routing, decision, target = _full_chain(_ub_lines()[1:])
    assert routing.family_eligibility["UB04"]["authorization_path"] == "COMPLETE_TOPOLOGY"
    assert decision.processing_route == ProcessingRoute.UB_STANDARD_EXTRACTOR
    assert target == ExtractionTarget.UB04_STANDARD
