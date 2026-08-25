import json
from hashlib import sha256
from pathlib import Path

import pytest

from evaluation.phase9e_accuracy import freeze_existing_predictions, score, validate_truth


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")


def _fixture(tmp_path: Path):
    image = tmp_path / "page.tif"
    image.write_bytes(b"private page")
    digest = sha256(image.read_bytes()).hexdigest()
    selection = tmp_path / "selection.jsonl"
    _write_jsonl(selection, [{"document_id": "d1", "package_id": "p1", "path": str(image), "sha256": digest}])
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [{"document_id": "d1", "route": "CMS1500", "fields": {"member_id": {"value": "A-1", "decision": {"disposition": "AUTO_ACCEPTED"}}}}])
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"ocr":"frozen"}', "utf-8")
    return selection, predictions, runtime


def test_freeze_is_order_bound_and_immutable(tmp_path):
    selection, predictions, runtime = _fixture(tmp_path)
    output = tmp_path / "prediction_freeze.json"
    frozen = freeze_existing_predictions(selection, predictions, runtime, output, code_sha="92eb506", expected_pages=1)
    assert frozen["truth_present"] is False
    with pytest.raises(FileExistsError):
        freeze_existing_predictions(selection, predictions, runtime, output, code_sha="92eb506", expected_pages=1)


def test_critical_truth_requires_independent_dual_annotation_and_adjudication(tmp_path):
    selection, _, _ = _fixture(tmp_path)
    truth = tmp_path / "truth.jsonl"
    row = {"document_id": "d1", "package_id": "p1", "document_type": "CMS_1500", "fields": {
        "member_id": {"status": "APPLICABLE", "annotator_a": {"annotator_id": "a", "value": "A1"},
                      "annotator_b": {"annotator_id": "b", "value": "A2"}, "final_value": "A1"}}}
    _write_jsonl(truth, [row])
    report = validate_truth(selection, truth, {"member_id"}, expected_pages=1)
    assert not report["complete"]
    assert any("ADJUDICATION" in error for error in report["errors"])


def test_scoring_true_stp_requires_correct_route_fields_and_no_hitl(tmp_path):
    selection, predictions, runtime = _fixture(tmp_path)
    freeze = tmp_path / "freeze.json"
    freeze_existing_predictions(selection, predictions, runtime, freeze, code_sha="92eb506", expected_pages=1)
    truth = tmp_path / "truth.jsonl"
    _write_jsonl(truth, [{"document_id": "d1", "package_id": "p1", "document_type": "CMS_1500",
                         "package_assembly_correct": True, "fields": {"member_id": {
                             "status": "APPLICABLE", "annotator_a": {"annotator_id": "a", "value": "A-1"},
                             "annotator_b": {"annotator_id": "b", "value": "A-1"}, "final_value": "A-1"}}}])
    report = score(selection, predictions, freeze, truth, {"member_id"}, expected_pages=1)
    assert report["critical_false_accepts"] == 0
    assert report["true_claim_stp"] == 1
