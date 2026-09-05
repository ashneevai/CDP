from packages.field_localization.name_region import resolve_name_region


def test_geometry_selects_name_and_excludes_neighbor_npi():
    observation = {
        "width": 1000,
        "height": 1000,
        "ocr_tokens": [
            {
                "text": "JOHN SMITH",
                "confidence": 0.9,
                "bbox": [50, 240, 250, 260],
                "reading_order": 2,
            },
            {
                "text": "1234567893",
                "confidence": 0.99,
                "bbox": [500, 240, 650, 260],
                "reading_order": 1,
            },
        ],
    }
    result = resolve_name_region("CMS1500", "provider_name", observation)
    assert result.value == "JOHN SMITH" and result.crop_safety_outcome == "CROP_SAFE"


def test_empty_and_unknown_regions_fail_closed():
    observation = {"width": 1000, "height": 1000, "ocr_tokens": []}
    assert (
        resolve_name_region("CMS1500", "provider_name", observation).crop_safety_outcome
        == "EMPTY_CROP"
    )
    assert (
        resolve_name_region("UB04", "insured_name", observation).crop_safety_outcome
        == "LOCALIZATION_UNCERTAIN"
    )


def test_geometry_chooses_nearest_value_line_and_preserves_reading_order():
    observation = {
        "width": 1000,
        "height": 1000,
        "ocr_tokens": [
            {
                "text": "NEIGHBOR NAME",
                "confidence": 0.99,
                "bbox": [50, 220, 250, 232],
                "reading_order": 1,
            },
            {
                "text": "GROUP",
                "confidence": 0.91,
                "bbox": [160, 248, 220, 260],
                "reading_order": 3,
            },
            {
                "text": "SUMMIT",
                "confidence": 0.93,
                "bbox": [50, 248, 150, 260],
                "reading_order": 2,
            },
        ],
    }

    result = resolve_name_region("CMS1500", "provider_name", observation)

    assert result.value == "SUMMIT GROUP"
    assert [token["text"] for token in result.selected_tokens] == ["SUMMIT", "GROUP"]
