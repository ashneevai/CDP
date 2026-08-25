from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from packages.domain.enums import ExtractionMethod
from packages.page_observation import PageObservationCache, PageObservationService
from workers.cascade.isolated_ocr import IsolatedTextExtractor, OCRTimeoutError


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
