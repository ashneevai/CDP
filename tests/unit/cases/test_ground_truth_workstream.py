import json

from evaluation.ground_truth_workstream import validate_annotations


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")


def test_critical_disagreement_requires_adjudication(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    _jsonl(manifest, [{"document_id": "d1", "package_id": "p1"}])
    row = {"document_id": "d1", "package_id": "p1", "document_type": "CMS1500",
           "fields": {"member_id": {"critical": True,
                                      "annotations": {"annotator_a": "A", "annotator_b": "B"}}}}
    _jsonl(annotations, [row])
    assert not validate_annotations(manifest=manifest, annotations=annotations)["valid"]
    row["fields"]["member_id"].update({"adjudicated_value": "A", "adjudicator_id": "sme-1"})
    _jsonl(annotations, [row])
    assert validate_annotations(manifest=manifest, annotations=annotations)["valid"]
