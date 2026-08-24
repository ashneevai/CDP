import json
from pathlib import Path
import zipfile

import pytest

from evaluation.reference_corpus import assert_frozen_corpus, summarize_claim_zip

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "evaluation" / "corpora" / "hackathon_1000_claims_v1.json"


def _fixture_zip(path: Path) -> Path:
    tiff = b"II*\x00" + b"test"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Group A/CLAIM1.001", tiff)
        archive.writestr("Group A/CLAIM1.002", tiff)
        archive.writestr("Group B/CLAIM2.001", tiff)
    return path


def test_reference_corpus_summarizer_counts_pages_packages_and_groups(tmp_path):
    summary = summarize_claim_zip(_fixture_zip(tmp_path / "claims.zip"))
    assert summary.pages == 3
    assert summary.packages == 2
    assert summary.groups == {"Group A": 2, "Group B": 1}
    assert summary.package_counts == {"Group A": 1, "Group B": 1}
    assert summary.all_tiff is True


def test_reference_corpus_validation_fails_closed_on_fingerprint_mismatch(tmp_path):
    path = _fixture_zip(tmp_path / "claims.zip")
    summary = summarize_claim_zip(path)
    expected = {
        "sha256": "wrong",
        "pages": summary.pages,
        "packages": summary.packages,
        "groups": summary.groups,
        "package_counts": summary.package_counts,
        "all_tiff": True,
    }
    with pytest.raises(ValueError, match="REFERENCE_CORPUS_MISMATCH"):
        assert_frozen_corpus(path, expected)


def test_hackathon_manifest_is_non_phi_aggregate_only():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["corpus_id"] == "hackathon-1000-claims-v1"
    assert manifest["pages"] == 1000
    assert manifest["packages"] == 110
    assert sum(manifest["groups"].values()) == 1000
    assert sum(manifest["package_counts"].values()) == 110
    assert manifest["privacy"]["raw_data_committed"] is False
    assert manifest["privacy"]["allow_only_manifest_and_aggregate_results"] is True
