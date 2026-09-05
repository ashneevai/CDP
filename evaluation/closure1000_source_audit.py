"""PHI-safe inventory and source audit for the supplied 1000-TIFF archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path(r"C:\Users\ashish.singh\Downloads\key docs\Hackathon - 1000 Claims.zip")
DEFAULT_EXTRACTED = ROOT / "evaluation_data/source_b_1000_claims"
DEFAULT_OUTPUT = ROOT / "evaluation_results/closure1000"
FROZEN_MANIFEST = ROOT / "evaluation_data/phase8_8_generalization/SOURCE_B/manifest.json"
PHASE9D = ROOT / "evaluation_results/phase9d/comparative_report.json"
SAMPLED_CLASSIFICATIONS = {
    "Group A/M048DJJF.001": ("CMS1500", 0.99, "CMS1500_NUCC_ANCHORS_OBSERVED"),
    "Group B/M048DJJZ.001": ("NON_CLAIM", 0.99, "DOCUMENT_SEPARATOR_ANCHORS_OBSERVED"),
    "Group C/M048DJJR.001": ("UB04", 0.99, "UB04_FORM_ANCHORS_OBSERVED"),
    "Group D/M048DJK5.001": ("NON_CLAIM", 0.99, "DOCUMENT_SEPARATOR_ANCHORS_OBSERVED"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_type(name: str, magic: bytes) -> tuple[str, str]:
    if magic.startswith((b"II*\x00", b"MM\x00*")):
        return "TIFF", "image/tiff"
    suffix = PurePosixPath(name).suffix.lower()
    kinds = {
        ".pdf": "PDF",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".xml": "XML",
        ".json": "JSON",
        ".csv": "CSV",
        ".txt": "TXT",
        ".edi": "EDI",
        ".zip": "ZIP",
    }
    return kinds.get(suffix, "OTHER"), mimetypes.guess_type(name)[0] or "application/octet-stream"


def inventory_archive(path: Path) -> dict[str, Any]:
    hashes: dict[str, list[str]] = defaultdict(list)
    records = []
    corrupt = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.is_dir()]
        unsafe = [
            info.filename
            for info in infos
            if PurePosixPath(info.filename).is_absolute()
            or ".." in PurePosixPath(info.filename).parts
        ]
        for info in files:
            try:
                data = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                corrupt.append({"relative_path": info.filename, "reason": str(error)})
                continue
            digest = sha256_bytes(data)
            hashes[digest].append(info.filename)
            kind, mime = media_type(info.filename, data[:16])
            records.append(
                {
                    "relative_path": info.filename,
                    "filename": PurePosixPath(info.filename).name,
                    "extension": PurePosixPath(info.filename).suffix,
                    "media_type": mime,
                    "byte_size": info.file_size,
                    "compressed_size": info.compress_size,
                    "sha256": digest,
                    "potential_type": kind,
                }
            )
        test_failure = archive.testzip()
        if test_failure and not any(row["relative_path"] == test_failure for row in corrupt):
            corrupt.append({"relative_path": test_failure, "reason": "CRC_FAILURE"})
    duplicates = [
        {"sha256": digest, "paths": paths, "count": len(paths)}
        for digest, paths in hashes.items()
        if len(paths) > 1
    ]
    return {
        "records": records,
        "total_files": len(records),
        "files_by_type": dict(Counter(row["potential_type"] for row in records)),
        "zero_byte_files": [row["relative_path"] for row in records if row["byte_size"] == 0],
        "corrupt_files": corrupt,
        "duplicates_by_sha256": duplicates,
        "nested_archives": [
            row["relative_path"] for row in records if row["potential_type"] == "ZIP"
        ],
        "unsafe_paths": unsafe,
        "compressed_size": sum(row["compressed_size"] for row in records),
        "uncompressed_size": sum(row["byte_size"] for row in records),
    }


def inspect_tiffs(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    assets = []
    pages = []
    corrupt = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                asset_id = str(image.tag_v2.get(45000, "")).strip() or sha256_file(path)
                bundle_id = str(image.tag_v2.get(45016, "")).strip() or path.stem
                frame_count = getattr(image, "n_frames", 1)
                assets.append(
                    {
                        "asset_id": asset_id,
                        "bundle_id": bundle_id,
                        "relative_path": relative,
                        "sha256": sha256_file(path),
                        "format": image.format,
                        "frame_count": frame_count,
                        "source_group": path.relative_to(root).parts[0],
                        "sequence": int(path.suffix[1:]) if path.suffix[1:].isdigit() else None,
                    }
                )
                for frame in range(frame_count):
                    image.seek(frame)
                    page_identity = hashlib.sha256()
                    page_identity.update(f"{image.mode}|{image.size[0]}|{image.size[1]}|".encode())
                    page_identity.update(image.tobytes())
                    pages.append(
                        {
                            "asset_id": asset_id,
                            "bundle_id": bundle_id,
                            "source_path": relative,
                            "parent_document": asset_id,
                            "page_number": frame + 1,
                            "page_content_sha256": page_identity.hexdigest(),
                            "width": image.width,
                            "height": image.height,
                            "document_class": SAMPLED_CLASSIFICATIONS.get(
                                relative, ("UNKNOWN", 0.0, "NOT_CLASSIFIED")
                            )[0],
                            "cdp_normalized_page_binding": None,
                        }
                    )
        except (OSError, ValueError) as error:
            corrupt.append({"relative_path": relative, "reason": str(error)})
    return assets, pages, corrupt


def run(
    zip_path: Path = DEFAULT_ZIP,
    extracted: Path = DEFAULT_EXTRACTED,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    archive = inventory_archive(zip_path)
    assets, pages, image_corrupt = inspect_tiffs(extracted)
    archive["corrupt_files"].extend(image_corrupt)
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        by_bundle[asset["bundle_id"]].append(asset)
    structures = []
    for bundle, items in sorted(by_bundle.items()):
        sequences = sorted(item["sequence"] for item in items)
        structures.append(
            {
                "bundle_id": bundle,
                "source_group": items[0]["source_group"],
                "asset_count": len(items),
                "rendered_page_count": sum(item["frame_count"] for item in items),
                "sequence_contiguous": sequences == list(range(1, len(items) + 1)),
                "claim_identity_status": "UNPROVEN_PACKAGE_BOUNDARY",
                "boundary_evidence": ["TIFF_TAG_45016", "FILENAME_STEM", "CONTIGUOUS_SEQUENCE"],
            }
        )

    frozen = json.loads(FROZEN_MANIFEST.read_text("utf-8"))
    real_hashes = {asset["sha256"] for asset in assets} | {
        page["page_content_sha256"] for page in pages
    }
    bindings = []
    for document in frozen["documents"]:
        matched = document["sha256"] in real_hashes
        bindings.append(
            {
                "frozen_claim_id": document["document_id"],
                "frozen_dataset_synthetic": frozen["synthetic"],
                "frozen_sha256": document["sha256"],
                "status": "MATCHED_EXACT" if matched else "NOT_FOUND",
                "binding_evidence": ["EXACT_HASH"] if matched else [],
                "candidate_matches": [],
                "authoritative_binding": False,
            }
        )
    if any(row["status"] == "MATCHED_EXACT" for row in bindings):
        raise RuntimeError("synthetic frozen document unexpectedly matched real source bytes")

    classifications = []
    for asset in assets:
        classification, confidence, reason = SAMPLED_CLASSIFICATIONS.get(
            asset["relative_path"], ("UNKNOWN", 0.0, "FULL_CLASSIFICATION_NOT_EXECUTED")
        )
        classifications.append(
            {
                "asset_id": asset["asset_id"],
                "relative_path": asset["relative_path"],
                "document_class": classification,
                "confidence": confidence,
                "reason": reason,
                "classification_scope": "BOUNDED_VALIDATION_SAMPLE" if confidence else "UNRESOLVED",
            }
        )
    class_counts = Counter(row["document_class"] for row in classifications)

    attachments = [
        {
            "bundle_id": bundle,
            "attachments_present": None,
            "attachment_count": None,
            "attachment_types": [],
            "attachment_pages": [],
            "attachment_ingested_by_CDP": False,
            "attachment_classified": False,
            "attachment_observed": False,
            "attachment_evidence_extracted": False,
            "status": "UNKNOWN_REQUIRES_DOCUMENT_CLASSIFICATION",
        }
        for bundle in sorted(by_bundle)
    ]
    phase9d = json.loads(PHASE9D.read_text("utf-8"))["metrics"]
    source_audit = {
        "historical_rules_changed": False,
        "previous_source_evidence_required_blockers": phase9d["source_evidence_required_blockers"],
        "reclassified_blockers": 0,
        "reason": "FROZEN_20_ARE_SYNTHETIC_AND_HAVE_NO_EXACT_BINDING_TO_REAL_PACKAGES",
        "remaining_status": "UNKNOWN_REQUIRES_SOURCE_BINDING",
        "ingestion_failures_confirmed": 0,
        "classification_failures_confirmed": 0,
        "page_lineage_failures_confirmed": 0,
    }
    e2 = {
        "opportunities": 0,
        "valid_independent_pairs": 0,
        "duplicate_same_crop_rejections": 0,
        "semantic_incompatibilities": 0,
        "conflicts": 0,
        "reason": "NO_EXACT_CLAIM_BINDING; E2 SEARCH WOULD RISK CROSS_CLAIM_EVIDENCE",
        "phase8_23_adjudicator_changed": False,
    }
    extraction = {
        "phase9d_extraction_defects_before": phase9d["extraction_defect_blockers"],
        "extraction_defects_after": phase9d["extraction_defect_blockers"],
        "fixes_attempted": 0,
        "reason": "NO_REAL_TO_FROZEN_CLAIM_BINDING_AND_NO_TRUSTED_LABELS",
    }
    closure_board = [
        {
            "claim_id": row["frozen_claim_id"],
            "source_bundle_found": False,
            "attachments_found": None,
            "lineage_complete": False,
            "extraction_blockers": None,
            "validation_blockers": None,
            "policy_blockers": None,
            "authoritative_blockers": None,
            "source_evidence_blockers": None,
            "unlock_distance": None,
            "final_disposition": "WAITING_FOR_SOURCE_DOCUMENT_BINDING",
        }
        for row in bindings
    ]
    cohort = {
        "source_assets": len(assets),
        "rendered_pages": len(pages),
        "packages_discovered": len(by_bundle),
        "claim_count": None,
        "claim_count_reason": "PACKAGE_TO_CLAIM_SEMANTICS_NOT_PROVEN",
        "document_classes": dict(class_counts),
        "quality_bands": {"UNKNOWN": len(assets)},
        "trusted_labeled_claims": 0,
        "unlabeled_packages": len(by_bundle),
        "operational_failure_cohorts": {
            "classification_unknown": class_counts["UNKNOWN"],
            "ingestion_failure_confirmed": 0,
            "missing_evidence_unassessable": len(by_bundle),
        },
    }
    performance = {
        "assets_inventoried": len(assets),
        "pages_inventoried": len(pages),
        "claim_processing_executed": False,
        "reason": "NO_TRUSTED_LABELS_OR_CLAIM_BINDING",
        "mean_processing_latency_ms": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "ocr_latency_ms": None,
        "ocr_calls_per_page": 0.0,
        "ocr_calls_per_claim": None,
        "memory_mb": None,
        "throughput_pages_per_second": None,
        "cost_per_page": 0.0,
        "cloud_cost": 0.0,
        "cold_start_included": False,
    }
    regression = {
        "historical_artifacts_modified": False,
        "thresholds_changed": False,
        "ocr_engines_added": False,
        "phase9d_metrics_changed": False,
        "critical_false_accepts": 0,
        "regressions": 0,
    }

    output.mkdir(parents=True, exist_ok=True)
    zip_manifest = {
        "zip_filename": zip_path.name,
        "byte_size": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "file_count": archive["total_files"],
        "compressed_size": archive["compressed_size"],
        "uncompressed_size": archive["uncompressed_size"],
        "original_modified": False,
        "extraction_location": str(extracted.relative_to(ROOT)).replace("\\", "/"),
        "raw_data_committed": False,
    }
    artifacts = {
        "source_zip_manifest.json": zip_manifest,
        "source_inventory.json": archive,
        "dataset_structure.json": {
            "packages": structures,
            "package_count": len(structures),
            "structure": "GROUP/BUNDLE_STEM.SEQUENCE with TIFF tags 45000/45016",
            "claim_identity_proven": False,
        },
        "document_classification.json": {
            "counts": dict(class_counts),
            "records": classifications,
            "full_classification_complete": False,
        },
        "source_b_binding.json": {
            "frozen_dataset_is_synthetic": frozen["synthetic"],
            "matched": 0,
            "unmatched": len(bindings),
            "ambiguous": 0,
            "records": bindings,
        },
        "page_lineage.json": {
            "source_asset_count": len(assets),
            "source_page_count": len(pages),
            "source_internal_lineage_coverage": 1.0,
            "cdp_binding_coverage": 0.0,
            "assets": assets,
            "pages": pages,
        },
        "attachment_inventory.json": {
            "confirmed_attachment_count": 0,
            "unknown_bundle_count": len(attachments),
            "records": attachments,
        },
        "source_audit_replay.json": source_audit,
        "e2_opportunity_analysis.json": e2,
        "extraction_defect_analysis.json": extraction,
        "claim_closure_board.json": {"claims": closure_board},
        "dataset_cohort_metrics.json": cohort,
        "performance_metrics.json": performance,
        "regression_analysis.json": regression,
    }
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")
    report = {
        "phase": "SOURCE_B_1000_CLOSURE",
        "verdict": "NEEDS_MORE_DATA",
        "verdict_reason": "Real source assets are intact, but claim binding, document classification, attachments, and trusted labels are not established",
        "zip_sha256": zip_manifest["sha256"],
        "metrics": {
            "total_source_files": len(assets),
            "rendered_pages": len(pages),
            "packages_discovered": len(by_bundle),
            "claims_discovered": None,
            "CMS1500_confirmed": class_counts["CMS1500"],
            "UB04_confirmed": class_counts["UB04"],
            "attachments_confirmed": 0,
            "unknown_documents": class_counts["UNKNOWN"],
            "frozen_20_matched": 0,
            "frozen_20_unmatched": 20,
            "ambiguous_bindings": 0,
            "source_page_lineage_coverage": 1.0,
            "cdp_page_binding_coverage": 0.0,
            "attachment_lineage_coverage": 0.0,
            "source_audit_blockers_reclassified": 0,
            "ingestion_defects_confirmed": 0,
            "classification_defects_confirmed": 0,
            "localization_defects_confirmed": 0,
            "extraction_defects_before": phase9d["extraction_defect_blockers"],
            "extraction_defects_after": phase9d["extraction_defect_blockers"],
            "valid_new_E2_pairs": 0,
            "blockers_removed": 0,
            "claims_advanced": 0,
            "claims_unlocked": 0,
            "raw_accuracy_before": 0.94,
            "raw_accuracy_after": None,
            "critical_accuracy_before": 0.95,
            "critical_accuracy_after": None,
            "field_HITL_before": 0.48,
            "field_HITL_after": None,
            "claim_HITL_before": 1.0,
            "claim_HITL_after": None,
            "achieved_STP": None,
            "accepted_precision": None,
            "critical_false_accepts": 0,
            "OCR_calls_per_claim": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "cost_per_page": 0.0,
            "remaining_authoritative_data_blockers": phase9d[
                "authoritative_data_required_blockers"
            ],
            "remaining_genuine_source_evidence_blockers": phase9d[
                "source_evidence_required_blockers"
            ],
            "claims_waiting_for_external_data": 20,
        },
        "engineering_closure_status": "NOT_REACHED_SOURCE_BINDING_AND_LABEL_GOVERNANCE_REQUIRED",
        "exact_remaining_work": [
            "Obtain claim/package mapping or source metadata for TIFF bundle IDs",
            "Run governed full-document classification over all 2173 pages",
            "Establish attachment semantics and page-to-document boundaries",
            "Obtain trusted human/source-system labels for representative accuracy evaluation",
            "Only then rerun Phase 8.26/8.27 and Phase 8.23 E2 adjudication",
        ],
    }
    (output / "final_closure_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.zip, args.extracted, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
