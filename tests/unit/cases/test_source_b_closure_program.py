from __future__ import annotations

import json
from pathlib import Path

from evaluation.source_b_closure_program import build_closure_program

ROOT = Path(__file__).resolve().parents[3]


def test_closure_program_is_fail_closed_and_phi_safe(tmp_path: Path) -> None:
    report = build_closure_program(output_dir=tmp_path)

    assert report["status"] == "NOT_EVALUATION_READY"
    assert report["real_accuracy"]["raw_field_accuracy"] is None
    assert report["claims"]["achieved_stp"] is None
    assert report["safety"] == {
        "new_acceptance_decisions": 0,
        "ocr_executed": False,
        "phi_written": False,
        "labels_fabricated": False,
    }


def test_closure_program_emits_required_artifacts(tmp_path: Path) -> None:
    build_closure_program(output_dir=tmp_path)
    required = {
        "source_inventory.json",
        "package_identity.json",
        "document_boundaries.json",
        "page_classification.json",
        "attachment_semantics.json",
        "annotation_manifest.json",
        "dataset_split_manifest.json",
        "source_to_cdp_lineage.json",
        "pipeline_coverage_matrix.json",
        "source_gap_analysis.json",
        "error_cohort_analysis.json",
        "extraction_experiment_log.json",
        "development_metrics.json",
        "holdout_metrics.json",
        "regression_analysis.json",
        "evidence_policy_analysis.json",
        "authoritative_data_request.json",
        "authoritative_snapshot_inventory.json",
        "claim_closure_board.json",
        "claim_unlock_distance.json",
        "stp_potential_analysis.json",
        "performance_metrics.json",
        "final_closure_report.json",
    }
    assert {path.name for path in tmp_path.glob("*.json")} == required


def test_real_archive_identity_and_classification_are_not_overstated(tmp_path: Path) -> None:
    build_closure_program(output_dir=tmp_path)
    packages = json.loads((tmp_path / "package_identity.json").read_text())
    pages = json.loads((tmp_path / "page_classification.json").read_text())
    boundaries = json.loads((tmp_path / "document_boundaries.json").read_text())

    assert len(packages["packages"]) == 110
    assert all(
        row["claim_identity_state"] == "IDENTITY_UNAVAILABLE" for row in packages["packages"]
    )
    assert sum(pages["counts"].values()) == 2173
    assert pages["counts"]["UNKNOWN"] > 2100
    assert boundaries["documents_confirmed"] == 0
    assert all(not row["asserted_document"] for row in boundaries["candidates"])


def test_annotation_and_package_split_remain_blocked(tmp_path: Path) -> None:
    build_closure_program(output_dir=tmp_path)
    annotation = json.loads((tmp_path / "annotation_manifest.json").read_text())
    split = json.loads((tmp_path / "dataset_split_manifest.json").read_text())

    assert annotation["assignments"] == []
    assert annotation["trusted_labels"] == 0
    assert split["development_package_ids"] == []
    assert split["frozen_holdout_package_ids"] == []
    assert split["package_leakage_count"] == 0


def test_source_lineage_is_distinct_from_cdp_binding(tmp_path: Path) -> None:
    build_closure_program(output_dir=tmp_path)
    lineage = json.loads((tmp_path / "source_to_cdp_lineage.json").read_text())
    performance = json.loads((tmp_path / "performance_metrics.json").read_text())

    assert lineage["source_internal_lineage_coverage"] == 1.0
    assert lineage["source_to_cdp_coverage"] == 0.0
    assert lineage["cdp_bound_pages"] == 0
    assert performance["inventory_only"] is True
    assert performance["p95_latency_ms"] is None
