from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from evaluation.promotion_scorer import score
from evaluation.truth_contract import iter_truth


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_truth_contract_rejects_duplicate_documents(tmp_path: Path):
    truth = tmp_path / "truth.jsonl"
    row = {"document_id": "p1", "package_id": "c1", "fields": {}}
    _write_jsonl(truth, [row, row])

    with pytest.raises(ValueError, match="DUPLICATE_TRUTH_DOCUMENT"):
        list(iter_truth(truth))


def test_promotion_scorer_requires_frozen_prediction_identity(
    tmp_path: Path,
):
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "document_id": "p1",
                "package_id": "c1",
                "group": "A",
                "schema": "CMS_1500",
                "wall_seconds": 1.0,
                "fields": {},
            }
        ],
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "corpus_id": "corpus",
                "corpus_sha256": "abc",
                "runtime_manifest_id": "runtime",
                "prediction_sha256": "wrong",
                "truth_present": False,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.jsonl"
    _write_jsonl(
        truth,
        [{"document_id": "p1", "package_id": "c1", "fields": {}}],
    )

    with pytest.raises(ValueError, match="PREDICTION_HASH_MISMATCH"):
        score(
            predictions_jsonl=predictions,
            prediction_freeze_json=freeze,
            truth_jsonl=truth,
            output_json=tmp_path / "report.json",
        )


def test_promotion_scorer_computes_quality_latency_and_cost(
    tmp_path: Path,
):
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "document_id": "p1",
                "package_id": "c1",
                "group": "A",
                "schema": "CMS_1500",
                "wall_seconds": 1.0,
                "cloud_cost_usd": 0.01,
                "fields": {
                    "member_id": {
                        "value": "M-1",
                        "decision": {"disposition": "AUTO_ACCEPTED"},
                    },
                    "npi": {
                        "value": "123",
                        "decision": {"disposition": "REVIEW_REQUIRED"},
                    },
                },
            },
            {
                "document_id": "p2",
                "package_id": "c2",
                "group": "B",
                "schema": "UB_04",
                "wall_seconds": 3.0,
                "cloud_cost_usd": 0.01,
                "fields": {
                    "member_id": {
                        "value": "BAD",
                        "decision": {"disposition": "AUTO_ACCEPTED"},
                    }
                },
            },
        ],
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "corpus_id": "corpus",
                "corpus_sha256": "abc",
                "runtime_manifest_id": "runtime",
                "prediction_sha256": _sha(predictions),
                "truth_present": False,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.jsonl"
    _write_jsonl(
        truth,
        [
            {
                "document_id": "p1",
                "package_id": "c1",
                "document_type": "CMS_1500",
                "fields": {
                    "member_id": {"value": "M-1", "critical": True},
                    "npi": {"value": "123", "critical": True},
                },
            },
            {
                "document_id": "p2",
                "package_id": "c2",
                "document_type": "UB_04",
                "fields": {
                    "member_id": {"value": "M-2", "critical": True}
                },
            },
        ],
    )

    report = score(
        predictions_jsonl=predictions,
        prediction_freeze_json=freeze,
        truth_jsonl=truth,
        output_json=tmp_path / "report.json",
        fully_loaded_cost_usd=0.06,
    )

    assert report["quality"]["overall_field_accuracy"] == pytest.approx(2 / 3)
    assert report["quality"]["accepted_precision"] == pytest.approx(0.5)
    assert report["quality"]["critical_false_accepts"] == 1
    assert report["automation"]["field_hitl_rate"] == pytest.approx(1 / 3)
    assert report["automation"]["claim_stp_rate"] == pytest.approx(0.5)
    assert report["latency_seconds"]["p95"] == 3.0
    assert report["cost"]["cost_usd_per_page"] == pytest.approx(0.03)
    assert report["promotion"]["promote"] is False
