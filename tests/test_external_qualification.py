from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from evaluation.external_qualification import (
    _load_manifest,
    score_frozen,
    verify_freeze,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, count: int = 2) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for index in range(count):
        image = tmp_path / f"page-{index}.png"
        image.write_bytes(f"page-{index}".encode())
        rows.append(
            {
                "document_id": f"DOC-{index}",
                "path": str(image),
                "sha256": _sha(image),
                "group": "A",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_manifest_requires_exact_page_count(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=2)
    with pytest.raises(ValueError, match="CORPUS_PAGE_COUNT_MISMATCH"):
        _load_manifest(manifest, expected_pages=1000)


def test_manifest_rejects_changed_source_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=1)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    Path(row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="CORPUS_FILE_HASH_MISMATCH"):
        _load_manifest(manifest, expected_pages=1)


def test_verify_freeze_rejects_prediction_tampering(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"document_id":"DOC-1"}\n', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "prediction_sha256": _sha(predictions),
                "truth_present": False,
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"document_id":"DOC-1","changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="PREDICTION_HASH_MISMATCH"):
        verify_freeze(predictions_jsonl=predictions, freeze_json=freeze)


def test_score_requires_independent_truth(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"document_id":"DOC-1"}\n', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "prediction_sha256": _sha(predictions),
                "truth_present": False,
                "runtime_manifest_id": "runtime-1",
                "corpus_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="INDEPENDENT_TRUTH_REQUIRED"):
        score_frozen(
            predictions_jsonl=predictions,
            freeze_json=freeze,
            truth_jsonl=tmp_path / "missing-truth.jsonl",
            output_json=tmp_path / "report.json",
        )
