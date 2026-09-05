import json

import pytest

from packages.retraining import (
    JsonlCorrectionSink,
    assert_partition_disjoint,
    correction_example,
    export_correction_dataset,
)


def test_export_is_source_disjoint_and_training_only(tmp_path):
    source = tmp_path / "corrections.jsonl"
    sink = JsonlCorrectionSink(source)
    for index in range(60):
        group = f"source-{index // 2}"
        sink.append(correction_example(
            f"doc-{index}", "patient_name", "JANE D0E", "JANE DOE", None, "reviewer-a",
            source_group_id=group, model_provenance={"ocr": "rapidocr-v1"},
        ))

    output = tmp_path / "weekly"
    manifest = export_correction_dataset(source, output, seed="locked-seed")
    assert sum(manifest.record_counts.values()) == 60
    split_groups = {}
    for split in ("train", "calibration", "holdout"):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        split_groups[split] = {row["source_group_id"] for row in rows}
        assert all(row["usage_authority"] == "TRAINING_ONLY" for row in rows)
        assert all(row["runtime_acceptance_authority"] is False for row in rows)
    assert split_groups["train"].isdisjoint(split_groups["calibration"])
    assert split_groups["train"].isdisjoint(split_groups["holdout"])
    assert split_groups["calibration"].isdisjoint(split_groups["holdout"])


def test_export_rejects_any_raw_correction_claiming_runtime_authority(tmp_path):
    source = tmp_path / "corrections.jsonl"
    source.write_text(json.dumps({
        "document_id": "d1", "field_name": "npi", "corrected_value": "123",
        "runtime_acceptance_authority": True,
    }) + "\n")
    with pytest.raises(ValueError, match="improperly claims runtime authority"):
        export_correction_dataset(source, tmp_path / "out")


def test_linked_entities_are_kept_in_one_partition(tmp_path):
    source = tmp_path / "corrections.jsonl"
    sink = JsonlCorrectionSink(source)
    sink.append(correction_example(
        "doc-a", "member_id", "A1", "A2", None, "reviewer-a",
        source_group_id="source-a", patient_id="patient-shared",
    ))
    sink.append(correction_example(
        "doc-b", "member_id", "B1", "B2", None, "reviewer-b",
        source_group_id="source-b", patient_id="patient-shared",
    ))
    output = tmp_path / "partitions"
    export_correction_dataset(source, output, seed="entity-safe")
    containing = []
    for split in ("train", "calibration", "holdout"):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        if rows:
            containing.append((split, {row["document_id"] for row in rows}))
    assert containing == [(containing[0][0], {"doc-a", "doc-b"})]


def test_locked_holdout_correction_cannot_move_to_learning_split(tmp_path):
    source = tmp_path / "corrections.jsonl"
    row = correction_example(
        "doc-h", "provider_npi", "1", "2", None, "reviewer-a",
        source_group_id="source-h",
    )
    payload = {**row.__dict__, "dataset_split": "holdout"}
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "partitions"
    export_correction_dataset(source, output, seed="would-otherwise-reassign")
    assert not (output / "train.jsonl").read_text().strip()
    assert not (output / "calibration.jsonl").read_text().strip()
    assert json.loads((output / "holdout.jsonl").read_text())["document_id"] == "doc-h"


def test_partition_audit_rejects_derived_identity_overlap():
    with pytest.raises(ValueError, match="patient_id"):
        assert_partition_disjoint({
            "train": [{"document_id": "d1", "patient_id": "p1"}],
            "holdout": [{"document_id": "d2", "patient_id": "p1"}],
        })
