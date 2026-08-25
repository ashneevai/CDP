from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from packages.domain.enums import ExtractionMethod
from packages.page_observation import PageObservationCache, PageObservationService
from packages.page_observation.contracts import ObservationToken
from workers.cascade.isolated_ocr import IsolatedTextExtractor, OCRTimeoutError
from workers.standard_form_extraction.spatial_assignment import (
    TokenOwnership,
    assign_tokens,
    spatial_features,
)


class CountingExtractor:
    engine_name = "fixture"
    model_name = "fixture-model"
    model_version = "fixture-v1"

    def __init__(self):
        self.calls = 0

    def extract(self, _image):
        self.calls += 1
        return []


def test_page_observation_cache_isolated_by_document_identity():
    extractor = CountingExtractor()
    service = PageObservationService(
        extractor, preprocessing_version="fixture-v1", cache=PageObservationCache()
    )
    image = Image.new("RGB", (32, 32), "white")
    first = service.observe("document-a", image)
    second = service.observe("document-b", image)

    assert extractor.calls == 2
    assert first.page_sha256 == second.page_sha256
    assert first.page_id != second.page_id


def test_observation_is_immutable_and_records_vendor_neutral_provenance():
    service = PageObservationService(CountingExtractor(), preprocessing_version="fixture-v1")
    observation = service.observe("document-a", Image.new("RGB", (32, 32), "white"))

    assert observation.ocr_provenance["engine"] == "fixture"
    with pytest.raises(ValidationError):
        observation.width = 99


def test_stalled_ocr_is_terminated_and_worker_restarts():
    worker = IsolatedTextExtractor(
        "tests.unit.cases.isolated_ocr_fakes",
        "WidthTriggeredOCR",
        timeout_seconds=2,
        engine_name="fixture",
    )
    try:
        with pytest.raises(OCRTimeoutError, match="OCR_TIMEOUT"):
            worker.extract(Image.new("RGB", (1, 1), "white"))
        assert worker.stats.timeouts == 1
        assert worker.stats.terminations == 1

        assert worker.extract(Image.new("RGB", (2, 2), "white")) == []
        assert worker.stats.starts == 2
        assert worker.stats.completed == 1
    finally:
        worker.close()


def test_phase9_runner_uses_canonical_processing_not_per_field_default():
    source = Path("evaluation/run_production_holdout_v2.py").read_text("utf-8")
    assert "standard_processing.process(" in source
    assert "rapid.extract_fields(" not in source


def test_page_observation_has_distinct_extraction_method():
    assert ExtractionMethod.PAGE_OBSERVATION.value == "PAGE_OBSERVATION"
    assert ExtractionMethod.SPATIAL_TOKEN_MAPPING.value == "SPATIAL_TOKEN_MAPPING"


def _token(token_id: str, text: str, bbox: tuple[float, float, float, float]):
    return ObservationToken(
        token_id=token_id,
        text=text,
        bbox=bbox,
        confidence=.9,
        line_index=0,
        reading_order=0,
    )


def test_spatial_assignment_uses_overlap_when_centroid_is_outside():
    features = spatial_features((8, 2, 14, 8), (0, 0, 10, 10), ())

    assert not features.centroid_contained
    assert features.token_overlap == 1 / 3
    result = assign_tokens((_token("t1", "ABC", (8, 2, 14, 8)),), {"member_id": (0, 0, 10, 10)})
    assert result[0].field_name == "member_id"
    assert result[0].ownership == TokenOwnership.VALUE


def test_spatial_assignment_never_duplicates_ambiguous_token():
    result = assign_tokens(
        (_token("t1", "123", (8, 2, 12, 8)),),
        {"left": (0, 0, 11, 10), "right": (9, 0, 20, 10)},
    )

    assert len(result) == 1
    assert result[0].ownership == TokenOwnership.AMBIGUOUS
    assert result[0].field_name is None


def test_spatial_assignment_firewalls_known_label():
    result = assign_tokens(
        (_token("t1", "Member ID", (1, 1, 9, 9)),),
        {"member_id": (0, 0, 10, 10)},
        labels_by_field={"member_id": {"Member ID"}},
    )

    assert result[0].ownership == TokenOwnership.LABEL


def test_phase9_runner_skips_paddle_for_terminal_fieldless_routes():
    source = Path("evaluation/run_production_holdout_v2.py").read_text("utf-8")
    assert "paddle_escalation_not_required" in source
    assert "duplicate_ocr_avoided" in source
