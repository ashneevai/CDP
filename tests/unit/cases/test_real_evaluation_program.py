from __future__ import annotations

import json
from pathlib import Path

from evaluation.real_evaluation_program import (
    adjudication_state,
    blind_annotation_view,
    build_real_evaluation_program,
    critical_regression,
    deterministic_bindings,
    experiment_verdict,
    is_trusted_label,
    package_split,
    pipeline_coverage,
)

ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_real_program_emits_complete_fail_closed_contract(tmp_path: Path) -> None:
    report = build_real_evaluation_program(output_dir=tmp_path)
    required = {
        "source_inventory.json",
        "page_classification_candidates.json",
        "document_boundary_candidates.json",
        "review_progress.json",
        "attachment_semantics.json",
        "package_identity.json",
        "package_mapping_request.json",
        "annotation_manifest.json",
        "annotation_agreement.json",
        "adjudication_log.json",
        "evaluation_cohort.json",
        "dataset_split_manifest.json",
        "source_to_cdp_binding.json",
        "pipeline_coverage.json",
        "development_baseline.json",
        "holdout_baseline.json",
        "performance_baseline.json",
        "real_error_cohorts.json",
        "optimization_experiments.json",
        "regression_analysis.json",
        "authoritative_data_request.json",
        "claim_closure_board.json",
        "readiness_gates.json",
        "final_real_evaluation_report.json",
    }
    assert {p.name for p in tmp_path.glob("*.json")} == required
    assert report["status"] == "NOT_EVALUATION_READY"
    assert report["pages_processed"] == 2173
    assert report["development_metrics"]["raw_accuracy"] is None
    assert report["holdout_metrics"]["claim_stp"] is None
    assert report["safety"]["synthetic_truth_used"] is False


def test_classification_candidates_are_not_promoted_to_truth(tmp_path: Path) -> None:
    build_real_evaluation_program(output_dir=tmp_path)
    artifact = json.loads((tmp_path / "page_classification_candidates.json").read_text())
    assert artifact["candidate_is_ground_truth"] is False
    assert sum(artifact["counts"].values()) == 2173
    assert artifact["counts"] == {"CMS1500": 1, "NON_CLAIM": 7, "UB04": 1, "UNKNOWN": 2164}
    assert all(row["review_state"] == "REVIEW_REQUIRED" for row in artifact["records"])


def test_blind_mode_dual_annotation_and_authority_rules() -> None:
    view = blind_annotation_view(
        {"field": "member_id", "prediction": "secret", "ocr_text": "secret", "crop_id": "c1"}
    )
    assert view == {"field": "member_id", "crop_id": "c1"}
    assert (
        adjudication_state({"normalized_value": "A"}, {"normalized_value": "A"})
        == "AGREED_PENDING_FINALIZATION"
    )
    assert (
        adjudication_state({"normalized_value": "A"}, {"normalized_value": "B"})
        == "ADJUDICATION_REQUIRED"
    )
    assert adjudication_state({"normalized_value": "A"}, None) == "INCOMPLETE"
    assert is_trusted_label(
        {
            "authority": "HUMAN_ADJUDICATED",
            "final_value": "A",
            "annotation_a_id": "a",
            "annotation_b_id": "b",
            "adjudication_id": "j",
            "finalized": True,
        }
    )
    assert is_trusted_label(
        {
            "authority": "SOURCE_SYSTEM_GROUND_TRUTH",
            "final_value": "A",
            "source_system": "core",
            "source_record_id": "r",
            "source_hash": "h",
        }
    )
    assert not is_trusted_label({"authority": "HUMAN_SINGLE_REVIEW", "final_value": "A"})
    assert not is_trusted_label({"authority": "MODEL_GENERATED", "final_value": "A"})


def test_package_split_is_deterministic_and_has_zero_leakage() -> None:
    one = package_split([f"package-{i}" for i in range(100)])
    two = package_split(reversed([f"package-{i}" for i in range(100)]))
    assert one == two
    assert one["package_leakage_count"] == 0
    assert set(one["development_package_ids"]).isdisjoint(one["holdout_package_ids"])
    assert len(one["development_package_ids"]) + len(one["holdout_package_ids"]) == 100


def test_source_to_cdp_binding_requires_unique_exact_evidence() -> None:
    source = [
        {"page_id": "s1", "rendered_sha256": "h1"},
        {"page_id": "s2", "rendered_sha256": "h2"},
        {"page_id": "s3", "source_representation_id": "r3"},
    ]
    cdp = [
        {"cdp_page_id": "c1", "rendered_sha256": "h1"},
        {"cdp_page_id": "c2", "rendered_sha256": "h2"},
        {"cdp_page_id": "c3", "rendered_sha256": "h2"},
    ]
    result = deterministic_bindings(source, cdp)
    assert result["bound_page_count"] == 1
    assert result["bindings"][0]["state"] == "MATCHED_EXACT"
    assert {r["state"] for r in result["unbound_pages"]} == {"AMBIGUOUS", "NOT_FOUND"}
    assert result["fuzzy_binding_allowed"] is False


def test_pipeline_coverage_reports_first_gap_per_selected_page() -> None:
    result = pipeline_coverage(
        ["p1", "p2"],
        [
            {"page_id": "p1", "stage": "SOURCE_PRESENT"},
            {"page_id": "p1", "stage": "DISCOVERED"},
            {"page_id": "p2", "stage": "SOURCE_PRESENT"},
        ],
    )
    assert result["stage_counts"]["SOURCE_PRESENT"] == 2
    assert result["stage_counts"]["DISCOVERED"] == 1
    assert {r["page_id"]: r["first_missing_stage"] for r in result["gaps"]} == {
        "p1": "INGESTED",
        "p2": "DISCOVERED",
    }


def test_optimization_gate_keeps_only_safe_development_improvements() -> None:
    safe = {
        "cohort": "DEVELOPMENT",
        "before": {
            "raw_accuracy": 0.90,
            "critical_accuracy": 0.99,
            "accepted_precision": 1.0,
            "critical_false_accepts": 0,
            "field_hitl": 0.2,
        },
        "after": {
            "raw_accuracy": 0.91,
            "critical_accuracy": 0.99,
            "accepted_precision": 1.0,
            "critical_false_accepts": 0,
            "field_hitl": 0.19,
        },
    }
    assert experiment_verdict(safe) == "KEEP"
    assert experiment_verdict({**safe, "cohort": "HOLDOUT"}) == "REVERT"
    unsafe = {**safe, "after": {**safe["after"], "critical_false_accepts": 1}}
    assert experiment_verdict(unsafe) == "REVERT"
    assert critical_regression(safe["before"], unsafe["after"])


def test_governed_inputs_enable_binding_but_not_claim_metrics_without_claim_identity(
    tmp_path: Path,
) -> None:
    pages = json.loads((ROOT / "evaluation_results/closure/page_classification.json").read_text())[
        "pages"
    ]
    inputs = tmp_path / "inputs"
    _write(
        inputs / "cdp_pages.json",
        {"records": [{"cdp_page_id": "cdp-1", "rendered_sha256": pages[0]["rendered_sha256"]}]},
    )
    _write(
        inputs / "evaluation_results.json",
        {
            "development": [
                {
                    "prediction": "x",
                    "label": "x",
                    "label_authority": "HUMAN_ADJUDICATED",
                    "critical": True,
                    "decision": "ACCEPTED",
                    "annotation_a_id": "a",
                    "annotation_b_id": "b",
                    "adjudication_id": "j",
                    "finalized": True,
                }
            ],
            "holdout": [],
        },
    )
    output = tmp_path / "out"
    report = build_real_evaluation_program(output_dir=output, inputs_dir=inputs)
    assert report["source_to_cdp_binding"] == 1 / 2173
    assert report["development_metrics"]["raw_accuracy"] == 1.0
    assert report["development_metrics"]["claim_stp"] is None


def test_full_runtime_candidates_require_complete_valid_manifest(tmp_path: Path) -> None:
    pages = json.loads((ROOT / "evaluation_results/closure/page_classification.json").read_text())[
        "pages"
    ]
    records = [
        {
            "package_id": page["package_id"],
            "source_asset_id": page["asset_id"],
            "source_page_id": page["page_id"],
            "candidate_class": "NON_CLAIM",
            "classification_confidence": 0.8,
            "confidence_band": "REVIEW_REQUIRED",
            "reason_codes": ["DETERMINISTIC_RUNTIME"],
            "evidence_counts": {"anchors": 1},
            "text_evidence": ["must not persist"],
        }
        for page in pages
    ]
    inputs = tmp_path / "inputs"
    _write(inputs / "page_classification_candidates.json", {"records": records})
    pages_by_package: dict[str, list[str]] = {}
    for page in pages:
        pages_by_package.setdefault(page["package_id"], []).append(page["page_id"])
    boundaries = [
        {
            "candidate_document_id": f"document-{index}",
            "package_id": package_id,
            "source_page_ids": page_ids,
            "boundary_state": "CANDIDATE",
            "notes": "must not persist",
        }
        for index, (package_id, page_ids) in enumerate(sorted(pages_by_package.items()))
    ]
    _write(
        inputs / "document_boundary_candidates.json",
        {"complete": True, "records": boundaries},
    )
    output = tmp_path / "out"
    report = build_real_evaluation_program(output_dir=output, inputs_dir=inputs)
    artifact = json.loads((output / "page_classification_candidates.json").read_text())
    boundary_artifact = json.loads((output / "document_boundary_candidates.json").read_text())
    assert artifact["runtime_manifest_valid"] is True
    assert artifact["source"] == "FULL_CLASSIFIER_RUNTIME"
    assert artifact["counts"] == {"NON_CLAIM": 2173}
    assert "text_evidence" not in artifact["records"][0]
    assert boundary_artifact["runtime_manifest_valid"] is True
    assert boundary_artifact["source"] == "FULL_CLASSIFIER_RUNTIME"
    assert all(not row["asserted_document"] for row in boundary_artifact["candidates"])
    assert "notes" not in boundary_artifact["candidates"][0]
    assert report["candidate_source"] == "FULL_CLASSIFIER_RUNTIME"
    assert report["runtime_manifest_valid"] is True
    assert report["boundary_candidate_source"] == "FULL_CLASSIFIER_RUNTIME"

    boundaries[0]["source_page_ids"] = ["missing-page"]
    _write(
        inputs / "document_boundary_candidates.json",
        {"complete": True, "records": boundaries},
    )
    rejected_output = tmp_path / "rejected"
    rejected = build_real_evaluation_program(output_dir=rejected_output, inputs_dir=inputs)
    assert rejected["runtime_manifest_valid"] is True
    assert rejected["boundary_runtime_manifest_valid"] is False
    assert rejected["boundary_candidate_source"] == "CLOSURE_INVENTORY_SEED"


def test_annotations_are_sanitized_and_unproven_authority_is_rejected(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(
        inputs / "annotations.json",
        {
            "records": [
                {
                    "annotation_id": "ann-1",
                    "field_instance_id": "f-1",
                    "page_id": "p-1",
                    "authority": "HUMAN_ADJUDICATED",
                    "final_value": "PHI",
                    "normalized_value": "PHI",
                    "notes": "PHI",
                    "ocr_text": "PHI",
                    "finalized": True,
                }
            ]
        },
    )
    output = tmp_path / "out"
    build_real_evaluation_program(output_dir=output, inputs_dir=inputs)
    artifact = json.loads((output / "annotation_manifest.json").read_text())
    assert artifact["trusted_labels"] == 0
    serialized = json.dumps(artifact)
    assert "PHI" not in serialized
    assert "final_value" not in artifact["records"][0]
    assert "record_sha256" in artifact["records"][0]


def test_claim_metrics_aggregate_actual_hitl_decisions(tmp_path: Path) -> None:
    provenance = {
        "label_authority": "SOURCE_SYSTEM_GROUND_TRUTH",
        "source_system": "core",
        "source_record_id": "record",
        "source_hash": "hash",
    }
    rows = [
        {**provenance, "claim_id": "c1", "prediction": "a", "label": "a", "decision": "ACCEPTED"},
        {**provenance, "claim_id": "c1", "prediction": "b", "label": "b", "decision": "HITL"},
        {**provenance, "claim_id": "c2", "prediction": "c", "label": "c", "decision": "ACCEPTED"},
    ]
    inputs = tmp_path / "inputs"
    _write(inputs / "evaluation_results.json", {"development": rows, "holdout": []})
    output = tmp_path / "out"
    report = build_real_evaluation_program(output_dir=output, inputs_dir=inputs)
    assert report["development_metrics"]["claim_hitl"] == 0.5
    assert report["development_metrics"]["claim_stp"] == 0.5
    assert report["development_metrics"]["evaluated_claims"] == 2


def test_dataset_ready_requires_explicit_complete_cohort_and_stable_lineage(
    tmp_path: Path,
) -> None:
    page = json.loads((ROOT / "evaluation_results/closure/page_classification.json").read_text())[
        "pages"
    ][0]
    inputs = tmp_path / "inputs"
    _write(
        inputs / "classification_reviews.json",
        {
            "records": [
                {
                    "page_id": page["page_id"],
                    "class": "CMS1500",
                    "review_state": "CONFIRMED",
                    "stable_lineage": True,
                }
            ]
        },
    )
    _write(
        inputs / "package_identities.json",
        {"records": [{"package_id": page["package_id"], "state": "CONFIRMED", "claim_id": "c1"}]},
    )
    annotation = {
        "annotation_id": "truth-1",
        "field_instance_id": "f1",
        "page_id": page["page_id"],
        "package_id": page["package_id"],
        "field_name": "member_id",
        "authority": "SOURCE_SYSTEM_GROUND_TRUTH",
        "final_value": "private",
        "source_system": "core",
        "source_record_id": "r1",
        "source_hash": "h1",
        "finalized": True,
    }
    _write(
        inputs / "annotations.json",
        {"selected_cohort_complete": True, "records": [annotation]},
    )
    _write(inputs / "evaluation_cohort.json", {"selected_page_ids": [page["page_id"]]})
    report = build_real_evaluation_program(output_dir=tmp_path / "complete", inputs_dir=inputs)
    assert report["status"] == "EVALUATION_READY"

    _write(
        inputs / "annotations.json",
        {"selected_cohort_complete": False, "records": [annotation]},
    )
    incomplete = build_real_evaluation_program(
        output_dir=tmp_path / "incomplete", inputs_dir=inputs
    )
    assert incomplete["status"] == "NOT_EVALUATION_READY"
