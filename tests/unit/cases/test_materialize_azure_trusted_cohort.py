import json

from evaluation.materialize_azure_trusted_cohort import materialize


def lines(path, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_materializer_requires_review_truth_and_exact_execution_lineage(tmp_path):
    run = tmp_path / "run"
    governed = tmp_path / "governed"
    governed.mkdir()
    (governed / "review_cohort_candidates.json").write_text(json.dumps({"selected_pages": 1}))
    lines(
        run / "page_reviews.jsonl",
        {
            "review_id": "r",
            "reviewer_id": "page-reviewer",
            "timestamp": "t",
            "action": "CONFIRM",
            "reviewed_class": "UB04",
            "reviewed_quality_band": "MEDIUM",
            "boundary_action": "CONFIRM_DOCUMENT_START",
            "source_page_id": "page",
            "source_asset_id": "asset",
            "package_id": "pkg",
        },
    )
    base = {
        "timestamp": "t",
        "package_id": "pkg",
        "source_page_id": "page",
        "field_name": "NPI",
        "critical": True,
        "state": "VALUE",
        "value": "1234567893",
        "value_sha256": "x",
        "source_region_sha256": "a" * 64,
        "authority": "HUMAN_SINGLE_REVIEW",
        "prediction_visible": False,
    }
    lines(
        run / "annotations.jsonl",
        base | {"annotation_id": "a", "annotator_id": "one", "annotator_role": "ANNOTATOR_A"},
        base | {"annotation_id": "b", "annotator_id": "two", "annotator_role": "ANNOTATOR_B"},
    )
    binding = {
        "exact": True,
        "rendered_page_sha256": "b" * 64,
        "source_representation_id": "rep",
        "pipeline_execution_id": "exec",
        "page_observation_id": "obs",
    }
    lines(
        run / "cdp_executions.jsonl",
        {
            "source_page_id": "page",
            "cdp_page_id": "cdp",
            "execution_id": "exec",
            "binding": binding,
            "stages": ["SOURCE_PRESENT", "DISCOVERED"],
            "fields": [
                {
                    "field_id": "field",
                    "field_name": "NPI",
                    "field_type": "NPI",
                    "local_candidates": ["123456789B", "1234567893"],
                    "local_decision": "HUMAN_REVIEW_REQUIRED",
                    "local_hitl": True,
                    "claim_blocking": True,
                    "crop_safe": True,
                    "localization_confidence": 0.9,
                }
            ],
        },
    )
    output = tmp_path / "trusted.json"
    result = materialize(run, output, governed)
    cohort = json.loads(output.read_text())["records"]
    assert (
        result["trusted_cohort_records"] == 1
        and cohort[0]["ground_truth_authority"] == "HUMAN_ADJUDICATED"
    )
    assert result["binding"]["binding_rate"] == 1 and result["annotation"]["agreements"] == 1
    assert json.loads((governed / "pipeline_coverage.json").read_text())["drops"]["INGESTED"] == 1


def test_materializer_does_not_promote_single_critical_review(tmp_path):
    run = tmp_path / "run"
    governed = tmp_path / "gov"
    governed.mkdir()
    (governed / "review_cohort_candidates.json").write_text('{"selected_pages":1}')
    lines(
        run / "page_reviews.jsonl",
        {
            "action": "CONFIRM",
            "reviewed_class": "UB04",
            "reviewed_quality_band": "HIGH",
            "source_page_id": "p",
            "source_asset_id": "a",
            "package_id": "pkg",
        },
    )
    lines(
        run / "annotations.jsonl",
        {
            "annotation_id": "a",
            "annotator_id": "one",
            "annotator_role": "ANNOTATOR_A",
            "source_page_id": "p",
            "field_name": "NPI",
            "state": "VALUE",
            "value": "1",
        },
    )
    lines(run / "cdp_executions.jsonl", {"source_page_id": "p", "fields": [{"field_name": "NPI"}]})
    result = materialize(run, tmp_path / "out.json", governed)
    assert result["trusted_cohort_records"] == 0 and result["annotation"]["trusted_labels"] == 0
