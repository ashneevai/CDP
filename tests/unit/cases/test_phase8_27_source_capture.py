import pytest

from evaluation.phase8_27_source_capture import run
from packages.claim_evidence.source_bundle_manifest import AvailabilityStatus, CapturedAsset


def test_unavailable_asset_cannot_have_fabricated_identity():
    with pytest.raises(ValueError, match="UNAVAILABLE_ASSET_MUST_NOT_HAVE_FABRICATED_IDENTITY"):
        CapturedAsset(status=AvailabilityStatus.UNAVAILABLE, asset_uri="invented.pdf")


def test_capture_manifest_backfills_only_verifiable_normalized_lineage(tmp_path):
    report = run(tmp_path)
    assert report["metrics"]["claims_manifested"] == 20
    assert report["metrics"]["normalized_pages_manifested"] == 20
    assert report["metrics"]["raw_bundles_available"] == 0
    assert report["metrics"]["pages_with_source_document_lineage"] == 0
    assert report["acceptance_gates"]["no_raw_asset_identity_fabricated"]
    assert report["verdict"] == "NEEDS_MORE_DATA"


def test_phase8_27_writes_capture_and_readiness_artifacts(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "source_bundle_capture_manifest.json", "acquisition_request.json",
        "capture_gap_metrics.json", "comparative_report.json",
    }
