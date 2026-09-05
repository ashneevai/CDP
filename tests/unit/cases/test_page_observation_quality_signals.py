from PIL import Image, ImageDraw

from packages.page_observation import PageObservationService


class NoopOCR:
    model_version = "noop-v1"

    def extract(self, image):
        return []


def test_new_observation_records_quality_and_source_signals(tmp_path, monkeypatch):
    monkeypatch.setenv("PAGE_OBSERVATION_CACHE_DIR", str(tmp_path / "cache"))
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(10):
        draw.text((30, 40 + offset * 35), "PRINTED CLAIM 12345", fill="black")
    result = PageObservationService(
        NoopOCR(), preprocessing_version="quality-signals-v1", benchmark_mode=True,
    ).observe("page-1", image, source_channel="FAX", resolution_dpi=200)
    quality = result.image_quality
    assert quality.source_channel == "FAX"
    assert quality.resolution_dpi == 200
    assert quality.resolution_width == 600
    assert quality.resolution_height == 800
    assert -15 <= quality.skew_degrees <= 15
    assert 0 <= quality.noise_estimate <= 1
    assert quality.writing_type in {"PRINTED", "MIXED", "HANDWRITTEN", "UNKNOWN"}
    assert quality.quality_bucket in {"HIGH", "MEDIUM", "LOW", "UNREADABLE"}
    assert 0 <= quality.dynamic_range <= 255
    assert 0 <= quality.background_uniformity <= 1
    assert 0 <= quality.binarization_quality <= 1
    assert quality.text_density == quality.foreground_ratio
    assert 0 <= quality.edge_density <= 1
    assert result.observation_version == "page-observation-v2-quality-bands"


def test_old_quality_payload_remains_loadable():
    from packages.page_observation import ImageQualityEvidence

    quality = ImageQualityEvidence(
        blur_score=10, contrast_score=20, foreground_ratio=.1, quality_bucket="degraded",
    )
    assert quality.source_channel == "UNKNOWN"
    assert quality.resolution_dpi is None
