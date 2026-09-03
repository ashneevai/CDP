import json
from pathlib import Path
import zipfile

import pytest

from evaluation.prediction_freeze import freeze_predictions


def _zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Group A/CLAIM1.001", b"II*\x00test")
    return path


def _runtime():
    return {
        "pipeline_version": "test",
        "ocr_runtime": "rapidocr@1",
        "preprocessing_version": "prep@1",
        "router_version": "router@1",
        "taxonomy_version": "taxonomy@1",
        "cms_template_version": "cms@1",
        "cms_field_graph_version": "cms-graph@1",
        "ub_template_version": "ub@1",
        "ub_field_registry_version": "ub-fields@1",
        "localization_version": "loc@1",
        "candidate_scoring_version": "candidate@1",
        "normalization_version": "norm@1",
        "evidence_policy_version": "evidence@1",
        "claim_evidence_version": "claim-evidence@1",
        "claim_decision_version": "claim-decision@1",
        "route_registry_version": "routes@1",
    }


def test_prediction_freeze_binds_corpus_runtime_and_predictions(tmp_path):
    corpus = _zip(tmp_path / "claims.zip")
    from evaluation.reference_corpus import summarize_claim_zip
    summary = summarize_claim_zip(corpus)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({
        "corpus_id": "external-v1",
        "sha256": summary.sha256,
        "pages": summary.pages,
        "packages": summary.packages,
        "groups": summary.groups,
        "package_counts": summary.package_counts,
        "all_tiff": summary.all_tiff,
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"page":"CLAIM1.001","route":"CMS1500"}\n', encoding="utf-8")

    freeze = freeze_predictions(
        corpus_zip=corpus,
        corpus_manifest=manifest,
        runtime_manifest=_runtime(),
        predictions_jsonl=predictions,
        output=tmp_path / "freeze.json",
    )
    assert freeze.corpus_id == "external-v1"
    assert freeze.prediction_records == 1
    assert freeze.truth_present is False
    assert freeze.scoring_allowed is False


def test_prediction_freeze_is_immutable(tmp_path):
    corpus = _zip(tmp_path / "claims.zip")
    from evaluation.reference_corpus import summarize_claim_zip
    summary = summarize_claim_zip(corpus)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({
        "corpus_id": "external-v1",
        "sha256": summary.sha256,
        "pages": 1,
        "packages": 1,
        "groups": {"Group A": 1},
        "package_counts": {"Group A": 1},
        "all_tiff": True,
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"page":"CLAIM1.001"}\n', encoding="utf-8")
    output = tmp_path / "freeze.json"
    freeze_predictions(corpus_zip=corpus, corpus_manifest=manifest, runtime_manifest=_runtime(), predictions_jsonl=predictions, output=output)
    with pytest.raises(FileExistsError, match="PREDICTION_FREEZE_ALREADY_EXISTS"):
        freeze_predictions(corpus_zip=corpus, corpus_manifest=manifest, runtime_manifest=_runtime(), predictions_jsonl=predictions, output=output)
