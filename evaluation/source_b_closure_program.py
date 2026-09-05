"""Build the PHI-safe Source-B closure control-plane artifacts.

This module intentionally performs no OCR and creates no labels, claim identities, or
acceptance decisions.  It projects already-committed audit facts into a single,
deterministic closure program until governed annotation and lineage are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "evaluation_results/closure1000"
DEFAULT_PHASE9D = ROOT / "evaluation_results/phase9d"
DEFAULT_PHASE9E = ROOT / "evaluation_results/phase9e"
DEFAULT_OUTPUT = ROOT / "evaluation_results/closure"
STATUS = "NOT_EVALUATION_READY"
IDENTITY_STATES = [
    "IDENTITY_CONFIRMED",
    "IDENTITY_CANDIDATE",
    "IDENTITY_AMBIGUOUS",
    "IDENTITY_UNAVAILABLE",
]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(directory: Path, name: str, value: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _stable_id(kind: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode()
    return f"{kind}_{hashlib.sha256(raw).hexdigest()[:24]}"


def build_closure_program(
    audit_dir: Path = DEFAULT_AUDIT,
    phase9d_dir: Path = DEFAULT_PHASE9D,
    phase9e_dir: Path = DEFAULT_PHASE9E,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    manifest = _read(audit_dir / "source_zip_manifest.json")
    inventory = _read(audit_dir / "source_inventory.json")
    structure = _read(audit_dir / "dataset_structure.json")
    classifications = _read(audit_dir / "document_classification.json")
    lineage = _read(audit_dir / "page_lineage.json")
    attachments = _read(audit_dir / "attachment_inventory.json")
    audit_final = _read(audit_dir / "final_closure_report.json")
    audit_perf = _read(audit_dir / "performance_metrics.json")
    audit_replay = _read(audit_dir / "source_audit_replay.json")
    audit_regression = _read(audit_dir / "regression_analysis.json")
    e2 = _read(audit_dir / "e2_opportunity_analysis.json")
    p9d = _read(phase9d_dir / "comparative_report.json")
    ceilings = _read(phase9d_dir / "stp_ceiling_analysis.json")
    blocker_data = _read(phase9d_dir / "blocker_classification.json")
    p9e = _read(phase9e_dir / "comparative_report.json")
    snapshots = _read(phase9e_dir / "authoritative_snapshot_inventory.json")

    archive_id = f"archive_sha256_{manifest['sha256']}"
    package_rows = []
    package_ids: dict[str, str] = {}
    for row in structure["packages"]:
        package_id = _stable_id("package", archive_id, row["bundle_id"])
        package_ids[row["bundle_id"]] = package_id
        package_rows.append(
            {
                "package_id": package_id,
                "source_package_key": row["bundle_id"],
                "archive_id": archive_id,
                "identity_state": "IDENTITY_CONFIRMED",
                "identity_scope": "STRUCTURAL_PACKAGE_ONLY",
                "claim_identity_state": "IDENTITY_UNAVAILABLE",
                "asset_count": row["asset_count"],
                "page_count": row["rendered_page_count"],
                "boundary_evidence": row["boundary_evidence"],
            }
        )

    asset_by_key = {row["asset_id"]: row for row in lineage["assets"]}
    asset_rows = []
    asset_ids: dict[str, str] = {}
    for asset in lineage["assets"]:
        asset_id = _stable_id("asset", archive_id, asset["sha256"])
        asset_ids[asset["asset_id"]] = asset_id
        asset_rows.append(
            {
                "asset_id": asset_id,
                "package_id": package_ids[asset["bundle_id"]],
                "sha256": asset["sha256"],
                "media_type": "image/tiff",
                "frame_count": asset["frame_count"],
                "identity_state": "IDENTITY_CONFIRMED",
            }
        )

    page_rows = []
    class_counts: Counter[str] = Counter()
    for page in lineage["pages"]:
        asset = asset_by_key[page["asset_id"]]
        document_class = page.get("document_class", "UNKNOWN")
        class_counts[document_class] += 1
        page_rows.append(
            {
                "page_id": _stable_id(
                    "page",
                    asset_ids[page["asset_id"]],
                    page["page_number"],
                    page["page_content_sha256"],
                ),
                "asset_id": asset_ids[page["asset_id"]],
                "package_id": package_ids[page["bundle_id"]],
                "page_number": page["page_number"],
                "rendered_sha256": page["page_content_sha256"],
                "identity_state": "IDENTITY_CONFIRMED",
                "classification": document_class,
                "classification_scope": (
                    "BOUNDED_CONFIRMED" if document_class != "UNKNOWN" else "UNCLASSIFIED"
                ),
                "classification_authority": (
                    "BOUNDED_HUMAN_OBSERVATION" if document_class != "UNKNOWN" else "NONE"
                ),
                "cdp_page_id": None,
                "cdp_binding_state": "IDENTITY_UNAVAILABLE",
            }
        )

    # TIFF assets are safe candidates for document boundaries, never asserted documents.
    boundary_rows = [
        {
            "candidate_document_id": _stable_id("document_candidate", row["asset_id"]),
            "package_id": row["package_id"],
            "asset_ids": [row["asset_id"]],
            "boundary_state": "IDENTITY_CANDIDATE",
            "reason": "TIFF_ASSET_BOUNDARY_REQUIRES_PAGE_TRANSITION_REVIEW",
            "asserted_document": False,
        }
        for row in asset_rows
    ]

    trusted_labels = 0
    annotation = {
        "status": "BLOCKED",
        "authority_policy": {
            "final": ["SOURCE_SYSTEM_GROUND_TRUTH", "HUMAN_ADJUDICATED"],
            "development_only": ["HUMAN_SINGLE_REVIEW"],
            "prohibited": ["SYNTHETIC", "MODEL_GENERATED", "UNKNOWN"],
        },
        "assignments": [],
        "trusted_labels": trusted_labels,
        "required_workflow": "ANNOTATOR_A_AND_B_THEN_ADJUDICATOR_ON_DISAGREEMENT",
        "reason": "NO_TRUSTED_REAL_LABELS_OR_CONFIRMED_DOCUMENT_BOUNDARIES",
    }
    split = {
        "status": "BLOCKED",
        "split_unit": "PACKAGE",
        "development_package_ids": [],
        "frozen_holdout_package_ids": [],
        "package_leakage_count": 0,
        "reason": "SPLIT_REQUIRES_GOVERNED_SELECTED_AND_LABELED_CORPUS",
    }

    source_summary = {
        "status": STATUS,
        "archive": {
            "archive_id": archive_id,
            "identity_state": "IDENTITY_CONFIRMED",
            "sha256": manifest["sha256"],
            "byte_size": manifest["byte_size"],
            "file_count": manifest["file_count"],
            "uncompressed_size": manifest["uncompressed_size"],
            "raw_data_committed": False,
        },
        "identity_states": IDENTITY_STATES,
        "package_count": len(package_rows),
        "asset_count": len(asset_rows),
        "page_count": len(page_rows),
        "packages": package_rows,
        "assets": asset_rows,
        "integrity": {
            "corrupt": len(inventory["corrupt_files"]),
            "duplicates": len(inventory["duplicates_by_sha256"]),
            "zero_byte": len(inventory["zero_byte_files"]),
            "nested_archives": len(inventory["nested_archives"]),
        },
    }

    source_to_cdp = {
        "status": "BLOCKED",
        "source_page_count": len(page_rows),
        "source_internal_lineage_coverage": lineage["source_internal_lineage_coverage"],
        "cdp_bound_pages": sum(row["cdp_page_id"] is not None for row in page_rows),
        "source_to_cdp_coverage": lineage["cdp_binding_coverage"],
        "bindings": [],
        "candidate_matches_are_authoritative": False,
        "reason": "NO_DETERMINISTIC_SOURCE_TO_CDP_PAGE_MAPPING",
    }
    coverage = {
        "status": "BLOCKED_AT_CDP_BINDING",
        "stages": {
            "SOURCE_PRESENT": len(page_rows),
            "DISCOVERED": len(page_rows),
            "INGESTED": None,
            "CLASSIFIED": sum(v for k, v in class_counts.items() if k != "UNKNOWN"),
            "PAGE_CREATED": None,
            "PAGE_OBSERVED": None,
            "FIELD_LOCALIZED": None,
            "CANDIDATE_GENERATED": None,
            "VALIDATED": None,
            "ACCEPTED_OR_HITL": None,
        },
        "unknown_is_not_zero": True,
        "drop_records": [],
        "reason": "CLAIM_AND_CDP_PAGE_BINDING_UNAVAILABLE",
    }

    null_metrics = {
        "document_classification_accuracy": None,
        "page_classification_accuracy": None,
        "raw_field_accuracy": None,
        "critical_accuracy": None,
        "accepted_precision": None,
        "field_hitl": None,
        "claim_hitl": None,
        "claim_stp": None,
        "critical_false_accepts": None,
        "reason": "NO_TRUSTED_REAL_LABELS",
    }
    frozen = {
        "dataset": "SYNTHETIC_FROZEN_SOURCE_B",
        "production_authority": False,
        "metrics": p9d["metrics"],
        "phase9e_metrics": p9e["metrics"],
        "use": "REGRESSION_BASELINE_ONLY_NOT_REAL_DATA_EVALUATION",
    }

    claim_board = {
        "status": "BLOCKED_NO_CLAIM_IDENTITY",
        "row_scope": "STRUCTURAL_PACKAGE_NOT_CLAIM",
        "packages": [
            {
                "package_id": row["package_id"],
                "claim_id": None,
                "claim_identity_state": "IDENTITY_UNAVAILABLE",
                "current_blocker_count": None,
                "unlock_distance": None,
                "owner": "SOURCE_DOCUMENT_OWNER",
                "final_disposition": "WAITING_FOR_SOURCE_DOCUMENTS",
            }
            for row in package_rows
        ],
        "claims": [],
    }
    top_actions = [
        {
            "rank": 1,
            "action": "Establish package-to-claim identity mapping",
            "owner": "SOURCE_DOCUMENT_OWNER",
        },
        {
            "rank": 2,
            "action": "Confirm document boundaries across all packages",
            "owner": "SOURCE_DOCUMENT_OWNER",
        },
        {
            "rank": 3,
            "action": "Classify all pages with governed evidence",
            "owner": "MANDATORY_HUMAN_REVIEW",
        },
        {"rank": 4, "action": "Classify attachment semantics", "owner": "MANDATORY_HUMAN_REVIEW"},
        {
            "rank": 5,
            "action": "Bind source pages to normalized CDP pages",
            "owner": "CDP_EXTRACTION",
        },
        {"rank": 6, "action": "Dual-annotate critical fields", "owner": "MANDATORY_HUMAN_REVIEW"},
        {
            "rank": 7,
            "action": "Adjudicate annotation disagreements",
            "owner": "MANDATORY_HUMAN_REVIEW",
        },
        {
            "rank": 8,
            "action": "Freeze package-level development and holdout splits",
            "owner": "CDP_ACCEPTANCE_POLICY",
        },
        {
            "rank": 9,
            "action": "Provision versioned member/provider snapshots",
            "owner": "MEMBER_DATA_OWNER",
        },
        {
            "rank": 10,
            "action": "Run untuned real holdout and operational gates",
            "owner": "CDP_VALIDATION",
        },
    ]

    outputs: dict[str, Any] = {
        "source_inventory.json": source_summary,
        "package_identity.json": {
            "status": STATUS,
            "claim_identity_proven": False,
            "packages": package_rows,
        },
        "document_boundaries.json": {
            "status": "CANDIDATES_ONLY",
            "documents_confirmed": 0,
            "candidates": boundary_rows,
        },
        "page_classification.json": {
            "status": "INCOMPLETE",
            "counts": dict(sorted(class_counts.items())),
            "bounded_asset_classifications": classifications["counts"],
            "pages": page_rows,
        },
        "attachment_semantics.json": {
            "status": "UNAVAILABLE",
            "confirmed_attachments": attachments["confirmed_attachment_count"],
            "records": [],
            "unknown_packages": len(package_rows),
        },
        "annotation_manifest.json": annotation,
        "dataset_split_manifest.json": split,
        "source_to_cdp_lineage.json": source_to_cdp,
        "pipeline_coverage_matrix.json": coverage,
        "source_gap_analysis.json": {
            "status": "UNRESOLVED",
            "confirmed_gaps": [],
            "source_audit_replay": audit_replay,
            "reason": "CDP_BINDING_REQUIRED_BEFORE_ATTRIBUTION",
        },
        "error_cohort_analysis.json": {
            "status": "BLOCKED",
            "trusted_real_errors": 0,
            "cohorts": [],
            "reason": "NO_TRUSTED_REAL_LABELS",
        },
        "extraction_experiment_log.json": {
            "status": "NO_EXPERIMENTS_RUN",
            "experiments": [],
            "reason": "REAL_ERROR_COHORT_NOT_ESTABLISHED",
        },
        "development_metrics.json": {"status": "NOT_EVALUATION_READY", **null_metrics},
        "holdout_metrics.json": {
            "status": "NOT_EVALUATION_READY",
            **null_metrics,
            "tuning_performed": False,
        },
        "regression_analysis.json": {
            "status": "NO_NEW_ACCEPTANCE_DECISIONS",
            "real_metrics": null_metrics,
            "frozen_phase9_baseline": frozen,
            "audit_regression": audit_regression,
        },
        "evidence_policy_analysis.json": {
            "status": "UNCHANGED",
            "real_e2_opportunities": 0,
            "e2_audit": e2,
            "same_crop_multi_engine_is_independent": False,
            "llm_evidence_allowed": False,
        },
        "authoritative_data_request.json": {
            "status": "REQUESTED",
            "scope": [
                "member eligibility for bound evaluation claims",
                "provider master keyed by NPI",
                "versioned ICD/CPT/HCPCS references",
            ],
            "requirements": [
                "source system",
                "dataset and schema version",
                "export timestamp",
                "effective dates",
                "record and file hashes",
            ],
            "claim_scope_available": False,
        },
        "authoritative_snapshot_inventory.json": {"status": "NOT_AVAILABLE", **snapshots},
        "claim_closure_board.json": claim_board,
        "claim_unlock_distance.json": {
            "status": "UNAVAILABLE_FOR_REAL_ARCHIVE",
            "real_claims": 0,
            "distance_distribution": None,
            "frozen_phase9": p9e["metrics"]["distance_after"],
        },
        "stp_potential_analysis.json": {
            "status": "NOT_MEASURABLE_FOR_REAL_ARCHIVE",
            "real_archive": {
                "ACHIEVED_STP": None,
                "TECHNICAL_STP_POTENTIAL": None,
                "AUTHORITATIVE_DATA_STP_POTENTIAL": None,
                "FULL_EVIDENCE_STP_POTENTIAL": None,
            },
            "frozen_synthetic_phase9": ceilings,
            "eligible_evidence_denominator_available": False,
        },
        "performance_metrics.json": {
            "status": "NOT_MEASURED",
            **audit_perf,
            "inventory_only": True,
            "reason": "NO_END_TO_END_ARCHIVE_PROCESSING_RUN",
        },
    }
    outputs["final_closure_report.json"] = {
        "status": STATUS,
        "reason": "TRUSTED_REAL_LABELS_CLAIM_IDENTITY_DOCUMENT_BOUNDARIES_AND_CDP_BINDING_UNAVAILABLE",
        "dataset": {
            "packages": len(package_rows),
            "assets": len(asset_rows),
            "pages": len(page_rows),
            "classifications": dict(sorted(class_counts.items())),
            "trusted_labels": 0,
            "development_size": 0,
            "holdout_size": 0,
            "split_leakage": 0,
        },
        "lineage": {
            "source_internal_coverage": lineage["source_internal_lineage_coverage"],
            "source_to_cdp_coverage": lineage["cdp_binding_coverage"],
        },
        "real_accuracy": null_metrics,
        "claims": {
            "identified": 0,
            "achieved_stp": None,
            "claim_hitl": None,
            "claims_unlocked": 0,
            "frozen_phase9_blockers_by_owner": blocker_data["by_owner"],
        },
        "evidence": {
            "real_e2_opportunities": 0,
            "e7_snapshots": snapshots["authoritative_snapshots_loaded"],
            "phase9_authoritative_blockers": p9d["metrics"]["authoritative_data_required_blockers"],
            "phase9_source_blockers": p9d["metrics"]["source_evidence_required_blockers"],
        },
        "performance": outputs["performance_metrics.json"],
        "frozen_phase9_baseline": frozen,
        "top_10_remaining_actions": top_actions,
        "safety": {
            "new_acceptance_decisions": 0,
            "ocr_executed": False,
            "phi_written": False,
            "labels_fabricated": False,
        },
        "source_audit_verdict": audit_final["verdict"],
    }

    for name, value in outputs.items():
        _write(output_dir, name, value)
    return outputs["final_closure_report.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--phase9d-dir", type=Path, default=DEFAULT_PHASE9D)
    parser.add_argument("--phase9e-dir", type=Path, default=DEFAULT_PHASE9E)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_closure_program(
        args.audit_dir, args.phase9d_dir, args.phase9e_dir, args.output_dir
    )
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
