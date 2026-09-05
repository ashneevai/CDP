from evaluation.benchmark_synthetic_claims import (
    _extract_crop,
    _retry_variant,
    _select_reference_ids,
    _should_register,
    _should_retry_field,
)
from evaluation.generate_public_synthetic_claims import (
    _fields,
    _intersects,
    _render,
    _template,
    _valid_npi,
    _value_layout,
)
from packages.validation_rules.npi import is_valid_npi


def test_generated_npi_is_checksum_valid_but_fixture_is_marked_synthetic():
    assert is_valid_npi(_valid_npi(42))
    assert _fields("CMS1500", 42)["insured_id_number"].startswith("SYN")


def test_generated_families_have_truth_and_template_sized_images():
    cms, cms_truth, cms_crops = _render("CMS1500", 1, "clean_scan")
    ub, _, ub_crops = _render("UB04", 2, "fax")
    assert cms.size == (1712, 2214)
    assert ub.size == (1711, 2216)
    assert set(cms_crops) <= set(cms_truth)
    assert {"provider_npi", "type_of_bill", "principal_diagnosis"} <= set(ub_crops)


def test_registration_reference_selection_avoids_geometrically_damaged_pages():
    manifest = {
        "cms-cropped": {"form_type": "CMS1500", "condition": "cropped_edges"},
        "cms-clean": {"form_type": "CMS1500", "condition": "clean_scan"},
        "ub-rotated": {"form_type": "UB04", "condition": "rotation"},
        "ub-fax": {"form_type": "UB04", "condition": "fax"},
    }
    assert _select_reference_ids(manifest) == {
        "CMS1500": "cms-clean",
        "UB04": "ub-fax",
    }


def test_adaptive_registration_preserves_mild_affine_skew_localization():
    assert _should_register(0.0)
    assert not _should_register(0.464)
    assert _should_register(-1.2)


def test_field_retry_is_bounded_by_type_and_confidence():
    assert _should_retry_field("npi", 0.99)
    assert _should_retry_field("date", 0.50)
    assert not _should_retry_field("date", 0.95)
    assert _should_retry_field("code", 0.50)
    assert not _should_retry_field("code", 0.95)
    assert not _should_retry_field("text", 0.10)


def test_code_retry_uses_stronger_bounded_upscale_than_date_retry():
    crop = _render("UB04", 2, "handwriting")[0].crop((0, 0, 40, 20))
    assert _retry_variant("code", crop).size == (120, 60)
    assert _retry_variant("date", crop).size == (80, 40)


def test_synthetic_labels_never_overlap_populated_data_regions():
    assert _intersects((0, 0, 10, 10), (9, 9, 20, 20))
    assert not _intersects((0, 0, 10, 10), (10, 10, 20, 20))
    cms, _, crops = _render("CMS1500", 1, "clean_scan")
    # Regression check for the previously contaminated member-ID crop: the
    # red insured_name label must no longer appear inside this populated ROI.
    member = cms.crop(crops["insured_id_number"])
    interior = member.crop((2, 2, member.width - 2, member.height - 2))
    red_pixels = sum(
        count
        for count, (r, g, b) in interior.getcolors(maxcolors=interior.width * interior.height)
        if r > g + 40 and r > b + 40
    )
    assert red_pixels == 0


def test_tight_synthetic_value_is_sized_inside_its_roi_without_changing_normal_fields():
    regions = {item["field_name"]: item for item in _template("UB04")["field_regions"]}
    tight = regions["federal_tax_no"]
    font, (_, y) = _value_layout("990000002", tight, False)
    _, top, _, bottom = font.getbbox("990000002")
    assert y + top >= tight["y0"]
    assert y + bottom <= tight["y1"]
    normal = regions["patient_name"]
    normal_font, normal_position = _value_layout("DEMO CHARLIE", normal, False)
    assert normal_font.size == 24
    assert normal_position == (normal["x0"] + 7, normal["y0"] + 7)


def test_benchmark_can_route_region_only_engines_without_using_full_page_ocr():
    calls = []

    class RegionOnly:
        def extract_region(self, image, x0, y0, x1, y1):
            calls.append((x0, y0, x1, y1))
            return []

    image, _, _ = _render("CMS1500", 1, "clean_scan")
    assert _extract_crop(RegionOnly(), image) == []
    assert calls == [(0, 0, image.width, image.height)]
