import pytest
from PIL import Image, ImageDraw

from packages.document_routing import MultiSignalRoute, MultiSignalRouter
from packages.ub04 import build_ub04_fingerprint
from workers.page_detection.text_extraction import TextLine


def _image(grid=True):
    image = Image.new("L", (1000, 1300), 255)
    if grid:
        draw = ImageDraw.Draw(image)
        for y in range(150, 1100, 100):
            draw.line((40, y, 960, y), fill=0, width=2)
        for x in (40, 250, 500, 750, 960):
            draw.line((x, 150, x, 1100), fill=0, width=2)
    return image


def _lines(*values):
    return [
        TextLine(value, 10, index * 30, 500, index * 30 + 20, 0.95)
        for index, value in enumerate(values)
    ]


def test_ub_fingerprint_requires_multiple_independent_signals():
    decision = MultiSignalRouter.load().route(
        _image(),
        [
            TextLine("UB-04", 20, 30, 400, 58, 0.95),
            TextLine("TYPE OF BILL", 760, 80, 960, 108, 0.95),
            TextLine("PATIENT CONTROL", 600, 100, 790, 128, 0.95),
            TextLine("STATEMENT COVERS", 650, 250, 900, 278, 0.95),
            TextLine("REVENUE CODE", 40, 420, 220, 448, 0.95),
            TextLine("HCPCS", 300, 430, 390, 458, 0.95),
            TextLine("UNITS", 650, 440, 720, 468, 0.95),
            TextLine("TOTAL CHARGES", 760, 450, 950, 478, 0.95),
            TextLine("PRINCIPAL DIAGNOSIS", 50, 900, 300, 928, 0.95),
        ],
    )
    assert decision.route is MultiSignalRoute.UB04
    evidence = build_ub04_fingerprint(decision, width=1000, height=1300)
    assert evidence.identity_anchor_present
    assert evidence.service_line_anchor_count >= 4
    assert evidence.type_of_bill_evidence


def test_healthcare_vocabulary_without_standard_identity_routes_custom():
    decision = MultiSignalRouter.load().route(
        _image(),
        _lines(
            "PATIENT MEMBER PROVIDER",
            "DIAGNOSIS PROCEDURE SERVICE DATE",
            "NPI CHARGE CLAIM",
        ),
    )
    assert decision.route is MultiSignalRoute.OTHER_CLAIM_FORM
    assert not decision.localization_allowed


def test_multiple_negative_anchors_and_low_healthcare_density_stop_nonclaim():
    decision = MultiSignalRouter.load().route(
        _image(False),
        _lines(
            "DOCUMENT COVER SHEET",
            "CORRESPONDENCE MEMORANDUM",
        ),
    )
    assert decision.route is MultiSignalRoute.NON_CLAIM


def test_close_standard_scores_fail_closed_to_unknown():
    decision = MultiSignalRouter.load().route(
        _image(),
        _lines(
            "CMS 1500 UB 04",
            "PATIENT CONTROL",
            "HEALTH INSURANCE CLAIM FORM",
        ),
    )
    assert decision.route not in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}


def test_explicit_ub_identity_and_one_anchor_stays_fail_closed():
    decision = MultiSignalRouter.load().route(
        _image(),
        _lines(
            "UB-04",
            "TYPE OF BILL",
            "CLAIM",
        ),
    )
    assert decision.route is not MultiSignalRoute.UB04
    assert not decision.eligibility["UB04"]
    assert not decision.localization_allowed


def test_identity_without_family_specific_anchor_stays_fail_closed():
    decision = MultiSignalRouter.load().route(_image(False), _lines("UB-04", "CLAIM"))
    assert decision.route is not MultiSignalRoute.UB04


def test_ocr_safe_normalization_is_bounded_to_multi_token_labels():
    decision = MultiSignalRouter.load().route(
        _image(),
        [
            TextLine("UB-04", 20, 30, 400, 58, 0.95),
            TextLine("TYPE OF BIIL", 760, 80, 960, 108, 0.95),
            TextLine("PATLENT CONTROL", 600, 100, 790, 128, 0.95),
            TextLine("STATEMENT COVERS", 650, 250, 900, 278, 0.95),
            TextLine("REVENUE CODE", 40, 420, 220, 448, 0.95),
            TextLine("HCPCS", 300, 430, 390, 458, 0.95),
            TextLine("UNITS", 650, 440, 720, 468, 0.95),
            TextLine("TOTAL CHARGES", 760, 450, 950, 478, 0.95),
            TextLine("PRINCIPAL DIAGNOS1S", 50, 900, 300, 928, 0.95),
        ],
    )
    assert decision.route is MultiSignalRoute.UB04
    assert decision.normalized_anchor_count >= 2
    generic = MultiSignalRouter.load().route(_image(), _lines("UN1TS", "NAME", "DATE"))
    assert generic.route not in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}


def test_complete_field_topology_can_survive_missing_identity_header():
    lines = [
        TextLine("TYPE OF BILL", 760, 70, 960, 100, 0.95),
        TextLine("PATIENT CONTROL", 600, 80, 790, 110, 0.95),
        TextLine("STATEMENT COVERS", 650, 150, 900, 180, 0.95),
        TextLine("REVENUE CODE", 40, 420, 220, 450, 0.95),
        TextLine("HCPCS", 300, 430, 390, 460, 0.95),
        TextLine("UNITS", 650, 440, 720, 470, 0.95),
        TextLine("TOTAL CHARGES", 760, 450, 950, 480, 0.95),
        TextLine("PRINCIPAL DIAGNOSIS", 50, 900, 300, 930, 0.95),
    ]
    decision = MultiSignalRouter.load().route(_image(), lines)
    assert not decision.matched_anchors["UB04_IDENTITY"]
    assert decision.eligibility["UB04"]
    assert decision.route is MultiSignalRoute.UB04
    assert "UB04_TOPOLOGY_CONFIRMED" in decision.reason_codes
    assert decision.identity_state["UB04"] == "CONFIRMED"
    assert decision.localization_allowed


# Synthetic regression fixtures derived from observed false-positive patterns; no PHI.
@pytest.mark.parametrize(
    "values",
    [
        ("REIMBURSEMENT REQUEST", "PATIENT PROVIDER CLAIM", "TYPE OF BILL", "TOTAL CHARGES"),
        ("PROPRIETARY CLAIM FORM", "PATIENT CONTROL", "REVENUE CODE", "HCPCS UNITS TOTAL CHARGES"),
        ("LEGACY CLAIM FORM", "STATEMENT COVERS", "MEDICAL RECORD", "PRINCIPAL DIAGNOSIS"),
    ],
)
def test_noncanonical_grid_claims_never_reach_ub_localization(values):
    decision = MultiSignalRouter.load().route(_image(), _lines(*values))
    assert decision.route is MultiSignalRoute.OTHER_CLAIM_FORM
    assert decision.identity_state["UB04"] == "REJECTED"
    assert not decision.localization_allowed
    assert "CLAIM_FORM_NONCANONICAL" in decision.reason_codes


def test_clean_cms_identity_reaches_canonical_route():
    decision = MultiSignalRouter.load().route(
        _image(),
        [
            TextLine("CMS-1500", 20, 30, 500, 58, 0.95),
            TextLine("HEALTH INSURANCE CLAIM FORM", 20, 100, 500, 128, 0.95),
            TextLine("PATIENTS NAME", 40, 260, 300, 288, 0.95),
            TextLine("INSURED ID NUMBER", 600, 220, 920, 248, 0.95),
            TextLine("DIAGNOSIS OR NATURE OF ILLNESS", 40, 700, 520, 728, 0.95),
            TextLine("FEDERAL TAX ID", 600, 1050, 920, 1078, 0.95),
        ],
    )
    assert decision.route is MultiSignalRoute.CMS1500
    assert decision.identity_state["CMS1500"] == "CONFIRMED"
    assert decision.localization_allowed


def test_low_confidence_identity_noise_cannot_veto_official_title_identity():
    decision = MultiSignalRouter.load().route(
        _image(),
        [
            # Some real scans split/drop the word HEALTH but retain this
            # official-title fragment with strong OCR confidence.
            TextLine("INSURANCE CLAIM FORM", 20, 100, 500, 128, 0.96),
            TextLine("PATIENTS NAME", 40, 260, 300, 288, 0.95),
            TextLine("INSURED ID NUMBER", 600, 220, 920, 248, 0.95),
            TextLine("DIAGNOSIS OR NATURE OF ILLNESS", 40, 700, 520, 728, 0.95),
            TextLine("FEDERAL TAX ID", 600, 1050, 920, 1078, 0.95),
            # A detector artifact elsewhere on the page has no authority.
            TextLine("CMS 1500", 400, 700, 600, 728, 0.0),
        ],
    )
    assert decision.route is MultiSignalRoute.CMS1500
    assert decision.identity_state["CMS1500"] == "CONFIRMED"
    assert not decision.conflicting_anchors["CMS1500"]
    assert decision.localization_allowed


def test_low_confidence_identity_text_cannot_authorize_a_standard_form():
    lines = [
        TextLine("CMS 1500", 20, 30, 500, 58, 0.79),
        TextLine("HEALTH INSURANCE CLAIM FORM", 20, 100, 500, 128, 0.95),
        TextLine("PATIENTS NAME", 40, 260, 300, 288, 0.95),
        TextLine("INSURED ID NUMBER", 600, 220, 920, 248, 0.95),
        TextLine("DIAGNOSIS OR NATURE OF ILLNESS", 40, 700, 520, 728, 0.95),
        TextLine("FEDERAL TAX ID", 600, 1050, 920, 1078, 0.95),
    ]
    decision = MultiSignalRouter.load().route(_image(), lines)
    # The official title remains a separate valid identity path; verify the
    # below-floor CMS token itself was excluded from the evidence ledger.
    assert all(
        not (item.canonical_anchor == "cms 1500" and item.ocr_confidence == 0.79)
        for item in decision.identity_anchor_evidence
    )


def test_exact_cms_identity_survives_adjacent_non_authorizing_fuzzy_shadow():
    # Word-level OCR can combine the exact ``1500`` token with the adjacent
    # title and produce a second fuzzy observation for the longer identity
    # alias.  That fuzzy shadow is evidence, but it is neither authority nor a
    # contradiction of the exact identity on the same canonical header.
    words = [
        ("CMS-1500", 21, 36, 151, 55),
        ("HEALTH", 23, 106, 118, 125),
        ("INSURANCE", 131, 106, 278, 125),
        ("CLAIM", 289, 106, 367, 125),
        ("FORM", 380, 106, 451, 125),
        ("INSURED", 603, 226, 715, 245),
        ("ID.", 728, 226, 752, 245),
        ("NUMBER", 764, 226, 874, 245),
        ("PATIENTS", 45, 266, 160, 285),
        ("NAME", 173, 266, 245, 285),
        ("DIAGNOSIS OR NATURE OF ILLNESS", 45, 706, 502, 725),
        ("FEDERAL TAX ID", 603, 1056, 809, 1075),
    ]
    decision = MultiSignalRouter.load().route(
        _image(),
        [TextLine(text, x0, y0, x1, y1, 0.95) for text, x0, y0, x1, y1 in words],
    )
    cms_evidence = [item for item in decision.identity_anchor_evidence if item.family == "CMS1500"]
    assert any(item.match_type == "FUZZY" for item in cms_evidence)
    assert decision.route is MultiSignalRoute.CMS1500
    assert decision.identity_state["CMS1500"] == "CONFIRMED"
    assert not decision.conflicting_anchors["CMS1500"]
    assert decision.localization_allowed


def test_conflicting_canonical_headers_fail_closed():
    decision = MultiSignalRouter.load().route(
        _image(),
        _lines(
            "CMS-1500 UB-04",
            "HEALTH INSURANCE CLAIM FORM",
            "TYPE OF BILL",
            "PATIENT CONTROL",
        ),
    )
    assert decision.route not in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}
    assert not decision.localization_allowed
