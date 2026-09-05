from PIL import Image, ImageDraw

from packages.domain.registration import RegistrationEvidence
from packages.templates.models import FieldRegion
from workers.page_detection.crop_safety import (
    CropSafetyOutcome,
    expanded_crop_boxes,
    validate_field_crop,
)


def _structured_page() -> Image.Image:
    image = Image.new("L", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 60, 160, 100), outline="black", width=2)
    draw.text((45, 65), "PATIENT NAME", fill="black")
    return image


def _region() -> FieldRegion:
    return FieldRegion(field_name="patient_name", x0=40, y0=60, x1=160, y1=100)


def _registration(confidence: float = 0.7) -> RegistrationEvidence:
    return RegistrationEvidence(
        algorithm="test",
        alignment_confidence=confidence,
        accepted=True,
        corner_validity=True,
    )


def test_expansion_is_bounded_to_three_variants_in_mid_confidence_band() -> None:
    boxes = expanded_crop_boxes(_region(), (200, 200), 0.7)
    assert len(boxes) == 3
    assert boxes[0] == (40, 60, 160, 100)
    assert boxes[-1] == (28, 56, 172, 104)


def test_expansion_is_disabled_outside_mid_confidence_band() -> None:
    assert len(expanded_crop_boxes(_region(), (200, 200), 0.59)) == 1
    assert len(expanded_crop_boxes(_region(), (200, 200), 0.80)) == 1


def test_matching_critical_crop_passes_geometry_checks() -> None:
    page = _structured_page()
    result = validate_field_crop(page, page.copy(), _region(), _registration(), critical=True)
    assert result.accepted
    assert result.local_alignment_accepted
    assert result.outcome is CropSafetyOutcome.CROP_SAFE


def test_critical_crop_fails_closed_without_registration() -> None:
    page = _structured_page()
    result = validate_field_crop(page, page.copy(), _region(), None, critical=True)
    assert not result.accepted
    assert "LOW_REGISTRATION_CONFIDENCE" in result.reason_codes


def test_critical_crop_rejects_neighbor_structure_mismatch() -> None:
    reference = _structured_page()
    candidate = Image.new("L", reference.size, "white")
    result = validate_field_crop(candidate, reference, _region(), _registration(), critical=True)
    assert not result.accepted
    assert "WRONG_CROP_SUSPECTED" in result.reason_codes
    assert result.outcome in {
        CropSafetyOutcome.WRONG_CROP_SUSPECTED, CropSafetyOutcome.EMPTY_CROP,
    }
