"""Deterministic control plane for governed real-data evaluation.

The program consumes the PHI-safe closure inventory plus optional governed review,
annotation, identity, and CDP execution manifests.  Missing evidence stays missing:
candidate classifications are not labels and synthetic/model output is never truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOSURE = ROOT / "evaluation_results/closure"
DEFAULT_OUTPUT = ROOT / "evaluation_results/real_eval"
TRUSTED_AUTHORITIES = {"SOURCE_SYSTEM_GROUND_TRUTH", "HUMAN_ADJUDICATED"}
FINAL_CLASSES = {"CMS1500", "UB04", "ATTACHMENT", "SUPPORTING_DOCUMENT", "NON_CLAIM", "UNKNOWN"}
PIPELINE_STAGES = (
    "SOURCE_PRESENT",
    "DISCOVERED",
    "INGESTED",
    "CLASSIFIED",
    "PAGE_CREATED",
    "OBSERVED",
    "LOCALIZED",
    "OCR_EXECUTED",
    "CANDIDATE_GENERATED",
    "VALIDATED",
    "ACCEPTED_OR_HITL",
)


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write(directory: Path, name: str, value: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def is_trusted_label(row: dict[str, Any]) -> bool:
    authority = row.get("authority")
    if row.get("final_value") is None:
        return False
    if authority == "SOURCE_SYSTEM_GROUND_TRUTH":
        return bool(
            row.get("source_system") and row.get("source_record_id") and row.get("source_hash")
        )
    if authority == "HUMAN_ADJUDICATED":
        return bool(
            row.get("annotation_a_id")
            and row.get("annotation_b_id")
            and row.get("adjudication_id")
            and row.get("finalized") is True
        )
    return False


def _safe_annotation(row: dict[str, Any]) -> dict[str, Any]:
    """Persist audit metadata while excluding values, free text, OCR, and crops."""
    allowed = {
        "annotation_id",
        "field_instance_id",
        "page_id",
        "package_id",
        "field_name",
        "critical",
        "annotator_role",
        "authority",
        "annotation_a_id",
        "annotation_b_id",
        "adjudication_id",
        "finalized",
        "status",
        "timestamp",
        "annotation_version",
        "source_system",
        "source_record_id",
        "source_hash",
    }
    safe = {key: row[key] for key in sorted(allowed & row.keys())}
    safe["record_sha256"] = _digest(row)
    return safe


def blind_annotation_view(field: dict[str, Any]) -> dict[str, Any]:
    """Return an initial-annotation view with prediction-bearing keys removed."""
    hidden = {"prediction", "predicted_value", "candidate_value", "ocr_text"}
    return {key: value for key, value in field.items() if key not in hidden}


def adjudication_state(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    if not a or not b:
        return "INCOMPLETE"
    if a.get("normalized_value") == b.get("normalized_value"):
        return "AGREED_PENDING_FINALIZATION"
    return "ADJUDICATION_REQUIRED"


def package_split(package_ids: Iterable[str], holdout_percent: int = 30) -> dict[str, Any]:
    """Make a deterministic package-level split; callers freeze its hash."""
    packages = sorted(set(package_ids))
    holdout = [
        p
        for p in packages
        if int(hashlib.sha256(p.encode()).hexdigest(), 16) % 100 < holdout_percent
    ]
    development = [p for p in packages if p not in set(holdout)]
    return {
        "split_unit": "PACKAGE",
        "development_package_ids": development,
        "holdout_package_ids": holdout,
        "package_leakage_count": len(set(development) & set(holdout)),
        "immutable_manifest_sha256": _digest({"development": development, "holdout": holdout}),
    }


def deterministic_bindings(
    source_pages: list[dict[str, Any]], cdp_pages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind only unique exact representations; fuzzy identifiers are ignored."""
    indexes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cdp_pages:
        for key in ("rendered_sha256", "source_representation_id"):
            value = row.get(key)
            if value:
                indexes.setdefault((key, str(value)), []).append(row)
    bindings, unbound = [], []
    for source in source_pages:
        candidates: dict[str, dict[str, Any]] = {}
        evidence = []
        for key in ("rendered_sha256", "source_representation_id"):
            value = source.get(key)
            if not value:
                continue
            matches = indexes.get((key, str(value)), [])
            evidence.append({"key": key, "matches": len(matches)})
            for match in matches:
                candidates[str(match["cdp_page_id"])] = match
        state = (
            "MATCHED_EXACT"
            if len(candidates) == 1
            else ("AMBIGUOUS" if candidates else "NOT_FOUND")
        )
        if state == "MATCHED_EXACT":
            bindings.append(
                {
                    "source_page_id": source["page_id"],
                    "cdp_page_id": next(iter(candidates)),
                    "state": state,
                    "evidence": evidence,
                }
            )
        else:
            unbound.append(
                {"source_page_id": source["page_id"], "state": state, "evidence": evidence}
            )
    total = len(source_pages)
    return {
        "bindings": bindings,
        "unbound_pages": unbound,
        "source_page_count": total,
        "bound_page_count": len(bindings),
        "binding_coverage": len(bindings) / total if total else 0.0,
        "binding_method": "UNIQUE_EXACT_HASH_OR_REPRESENTATION_ID",
        "fuzzy_binding_allowed": False,
    }


def pipeline_coverage(selected_pages: list[str], events: list[dict[str, Any]]) -> dict[str, Any]:
    by_page = {page: set() for page in selected_pages}
    for event in events:
        if event.get("page_id") in by_page and event.get("stage") in PIPELINE_STAGES:
            by_page[event["page_id"]].add(event["stage"])
    counts = {
        stage: sum(stage in stages for stages in by_page.values()) for stage in PIPELINE_STAGES
    }
    gaps = [
        {
            "page_id": page,
            "first_missing_stage": next((s for s in PIPELINE_STAGES if s not in stages), None),
        }
        for page, stages in sorted(by_page.items())
        if len(stages) != len(PIPELINE_STAGES)
    ]
    return {"selected_pages": len(selected_pages), "stage_counts": counts, "gaps": gaps}


def experiment_verdict(experiment: dict[str, Any]) -> str:
    after = experiment.get("after", {})
    before = experiment.get("before", {})
    improves = any(
        after.get(key) is not None and before.get(key) is not None and after[key] > before[key]
        for key in ("raw_accuracy", "critical_accuracy")
    ) or (
        after.get("field_hitl") is not None
        and before.get("field_hitl") is not None
        and after["field_hitl"] < before["field_hitl"]
    )
    safe = (
        after.get("accepted_precision") is not None
        and after["accepted_precision"] >= 0.995
        and after.get("critical_false_accepts") == 0
        and not critical_regression(before, after)
    )
    return "KEEP" if safe and improves and experiment.get("cohort") == "DEVELOPMENT" else "REVERT"


def critical_regression(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key in ("critical_accuracy", "accepted_precision"):
        if before.get(key) is not None and after.get(key) is not None and after[key] < before[key]:
            return True
    return (after.get("critical_false_accepts") or 0) > (before.get("critical_false_accepts") or 0)


def build_real_evaluation_program(
    closure_dir: Path = DEFAULT_CLOSURE,
    output_dir: Path = DEFAULT_OUTPUT,
    inputs_dir: Path | None = None,
) -> dict[str, Any]:
    inputs_dir = inputs_dir or Path("__governed_inputs_not_supplied__")
    source = _read(closure_dir / "source_inventory.json")
    page_seed = _read(closure_dir / "page_classification.json")
    boundary_seed = _read(closure_dir / "document_boundaries.json")
    package_seed = _read(closure_dir / "package_identity.json")
    perf_seed = _read(closure_dir / "performance_metrics.json")
    closure_final = _read(closure_dir / "final_closure_report.json")

    reviews = _read(inputs_dir / "classification_reviews.json", {"records": []})["records"]
    boundary_reviews = _read(inputs_dir / "document_boundary_reviews.json", {"records": []})[
        "records"
    ]
    annotation_input = _read(inputs_dir / "annotations.json", {"records": []})
    annotations = annotation_input["records"]
    cohort_input = _read(inputs_dir / "evaluation_cohort.json", {"selected_page_ids": []})
    identities = _read(inputs_dir / "package_identities.json", {"records": []})["records"]
    cdp_pages = _read(inputs_dir / "cdp_pages.json", {"records": []})["records"]
    events = _read(inputs_dir / "pipeline_events.json", {"records": []})["records"]
    experiments = _read(inputs_dir / "optimization_experiments.json", {"experiments": []})[
        "experiments"
    ]

    reviewed = {
        row["page_id"]: row
        for row in reviews
        if row.get("class") in FINAL_CLASSES
        and row.get("review_state") == "CONFIRMED"
        and row.get("stable_lineage") is True
    }
    candidate_rows = []
    candidate_counts: Counter[str] = Counter()
    for page in page_seed["pages"]:
        candidate = page["classification"]
        confidence_band = "REVIEW_REQUIRED" if candidate != "UNKNOWN" else "UNKNOWN"
        candidate_counts[candidate] += 1
        candidate_rows.append(
            {
                "archive_id": source["archive"]["archive_id"],
                "package_id": page["package_id"],
                "source_asset_id": page["asset_id"],
                "source_page_id": page["page_id"],
                "candidate_class": candidate,
                "classification_confidence": None,
                "confidence_band": confidence_band,
                "template_registration": None,
                "anchor_evidence": [],
                "layout_evidence": [],
                "grid_line_evidence": [],
                "text_evidence": [],
                "reason_codes": ["CLOSURE_INVENTORY_SEED"],
                "review_state": "CONFIRMED" if page["page_id"] in reviewed else "REVIEW_REQUIRED",
            }
        )

    runtime_manifest = _read(inputs_dir / "page_classification_candidates.json", {"records": []})
    runtime_rows = runtime_manifest.get("records", [])
    expected_pages = {page["page_id"]: page for page in page_seed["pages"]}
    expected_page_ids = set(expected_pages)
    runtime_page_ids = [row.get("source_page_id") for row in runtime_rows]
    runtime_complete = (
        len(runtime_rows) == len(expected_page_ids)
        and set(runtime_page_ids) == expected_page_ids
        and len(set(runtime_page_ids)) == len(runtime_page_ids)
    )
    runtime_valid = runtime_complete and all(
        row.get("candidate_class") in FINAL_CLASSES
        and row.get("confidence_band")
        in {"HIGH_CONFIDENCE_CANDIDATE", "REVIEW_REQUIRED", "UNKNOWN"}
        and row.get("package_id") == expected_pages[row["source_page_id"]]["package_id"]
        and row.get("source_asset_id") == expected_pages[row["source_page_id"]]["asset_id"]
        and (
            row.get("classification_confidence") is None
            or isinstance(row.get("classification_confidence"), (int, float))
            and 0 <= row["classification_confidence"] <= 1
        )
        for row in runtime_rows
    )
    if runtime_valid:
        safe_runtime_rows = []
        for row in runtime_rows:
            safe_runtime_rows.append(
                {
                    "archive_id": source["archive"]["archive_id"],
                    "package_id": row["package_id"],
                    "source_asset_id": row["source_asset_id"],
                    "source_page_id": row["source_page_id"],
                    "candidate_class": row["candidate_class"],
                    "classification_confidence": row.get("classification_confidence"),
                    "confidence_band": row["confidence_band"],
                    "template_registration_present": row.get("template_registration") is not None,
                    "evidence_counts": {
                        key: value
                        for key in ("anchors", "layout", "grid_lines", "text_tokens")
                        if isinstance((value := row.get("evidence_counts", {}).get(key)), int)
                        and value >= 0
                    },
                    "candidate_record_sha256": _digest(row),
                    "review_state": "CONFIRMED"
                    if row["source_page_id"] in reviewed
                    else "REVIEW_REQUIRED",
                }
            )
        candidate_rows = safe_runtime_rows
        candidate_counts = Counter(row["candidate_class"] for row in candidate_rows)
    candidate_source = "FULL_CLASSIFIER_RUNTIME" if runtime_valid else "CLOSURE_INVENTORY_SEED"

    boundary_manifest = _read(inputs_dir / "document_boundary_candidates.json", {"records": []})
    runtime_boundary_rows = boundary_manifest.get("records", [])
    expected_package_ids = {page["package_id"] for page in page_seed["pages"]}
    boundary_ids = [
        row.get("candidate_document_id") or row.get("boundary_candidate_id")
        for row in runtime_boundary_rows
    ]
    boundary_packages = {row.get("package_id") for row in runtime_boundary_rows}
    runtime_boundaries_valid = (
        runtime_valid
        and boundary_manifest.get("complete") is True
        and bool(runtime_boundary_rows)
        and len(boundary_ids) == len(set(boundary_ids))
        and boundary_packages == expected_package_ids
        and all(
            bool(row.get("candidate_document_id") or row.get("boundary_candidate_id"))
            and row.get("boundary_state") in {"CANDIDATE", "AMBIGUOUS", "UNKNOWN"}
            and bool(row.get("source_page_ids") or row.get("source_page_id"))
            and all(
                page_id in expected_pages
                and expected_pages[page_id]["package_id"] == row.get("package_id")
                for page_id in (row.get("source_page_ids") or [row.get("source_page_id")])
            )
            for row in runtime_boundary_rows
        )
    )
    if runtime_boundaries_valid:
        boundary_candidates = [
            {
                "candidate_document_id": row.get("candidate_document_id")
                or row["boundary_candidate_id"],
                "package_id": row["package_id"],
                "source_page_ids": list(row.get("source_page_ids") or [row["source_page_id"]]),
                "boundary_state": row["boundary_state"],
                "asserted_document": False,
                "candidate_record_sha256": _digest(row),
            }
            for row in runtime_boundary_rows
        ]
        boundary_source = "FULL_CLASSIFIER_RUNTIME"
    else:
        boundary_candidates = boundary_seed["candidates"]
        boundary_source = "CLOSURE_INVENTORY_SEED"

    trusted = [row for row in annotations if is_trusted_label(row)]
    critical = [row for row in annotations if row.get("critical")]
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in critical:
        pairs.setdefault(str(row.get("field_instance_id")), {})[str(row.get("annotator_role"))] = (
            row
        )
    agreement_states = [
        adjudication_state(p.get("ANNOTATOR_A"), p.get("ANNOTATOR_B")) for p in pairs.values()
    ]
    agreement_counts = Counter(agreement_states)
    final_adjudications = [
        row
        for row in annotations
        if row.get("authority") == "HUMAN_ADJUDICATED" and is_trusted_label(row)
    ]

    confirmed_identity = {r["package_id"]: r for r in identities if r.get("state") == "CONFIRMED"}
    eligible_packages = sorted({r["package_id"] for r in trusted} & set(confirmed_identity))
    split = (
        package_split(eligible_packages)
        if eligible_packages
        else {
            "split_unit": "PACKAGE",
            "development_package_ids": [],
            "holdout_package_ids": [],
            "package_leakage_count": 0,
            "immutable_manifest_sha256": None,
        }
    )
    source_pages = [
        {
            "page_id": p["page_id"],
            "rendered_sha256": p["rendered_sha256"],
            "source_representation_id": p.get("source_representation_id"),
        }
        for p in page_seed["pages"]
    ]
    binding = deterministic_bindings(source_pages, cdp_pages)
    selected_pages = sorted(
        set(cohort_input.get("selected_page_ids", []))
        or {str(r.get("page_id")) for r in annotations if r.get("page_id")}
    )
    coverage = pipeline_coverage(selected_pages, events)

    # Predictions and labels must be supplied as explicitly separated evaluation rows.
    results = _read(inputs_dir / "evaluation_results.json", {"development": [], "holdout": []})

    def metrics(rows: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
        usable = [
            row
            for row in rows
            if is_trusted_label(
                {
                    **row,
                    "authority": row.get("label_authority"),
                    "final_value": row.get("label"),
                }
            )
        ]
        if not usable:
            return {
                "status": "BLOCKED_NO_TRUSTED_LABELS",
                "cohort": cohort,
                "raw_accuracy": None,
                "critical_accuracy": None,
                "accepted_precision": None,
                "field_hitl": None,
                "claim_hitl": None,
                "claim_stp": None,
                "critical_false_accepts": None,
                "evaluated_fields": 0,
            }
        correct = lambda r: r.get("prediction") == r.get("label")
        crit = [r for r in usable if r.get("critical")]
        accepted = [r for r in usable if r.get("decision") == "ACCEPTED"]
        bad_critical = [r for r in accepted if r.get("critical") and not correct(r)]
        claim_allowed = all(r.get("claim_id") for r in usable)
        claims: dict[str, list[dict[str, Any]]] = {}
        if claim_allowed:
            for row in usable:
                claims.setdefault(str(row["claim_id"]), []).append(row)
        hitl_claims = sum(
            any(row.get("decision") == "HITL" for row in claim_rows)
            for claim_rows in claims.values()
        )
        claim_count = len(claims)
        return {
            "status": "MEASURED",
            "cohort": cohort,
            "raw_accuracy": sum(map(correct, usable)) / len(usable),
            "critical_accuracy": sum(map(correct, crit)) / len(crit) if crit else None,
            "accepted_precision": sum(map(correct, accepted)) / len(accepted) if accepted else None,
            "field_hitl": sum(r.get("decision") == "HITL" for r in usable) / len(usable),
            "claim_hitl": None if not claim_allowed else hitl_claims / claim_count,
            "claim_stp": None if not claim_allowed else (claim_count - hitl_claims) / claim_count,
            "critical_false_accepts": len(bad_critical),
            "evaluated_fields": len(usable),
            "evaluated_claims": claim_count if claim_allowed else None,
        }

    dev = metrics(results.get("development", []), "DEVELOPMENT")
    holdout = metrics(results.get("holdout", []), "HOLDOUT")
    reviewed_classes = Counter(r["class"] for r in reviews if r.get("review_state") == "CONFIRMED")
    boundary_confirmed = sum(r.get("state") == "CONFIRMED" for r in boundary_reviews)
    reviewed_page_ids = set(reviewed)
    lineage_coverage = (
        sum(page in reviewed_page_ids for page in selected_pages) / len(selected_pages)
        if selected_pages
        else 0.0
    )
    finalized_adjudication_fields = {
        str(row.get("field_instance_id")) for row in final_adjudications if is_trusted_label(row)
    }
    adjudication_complete = all(
        state != "ADJUDICATION_REQUIRED" or field_id in finalized_adjudication_fields
        for field_id, state in zip(pairs, agreement_states, strict=True)
    )
    trusted_pages = {str(row.get("page_id")) for row in trusted if row.get("page_id")}
    labels_complete = (
        annotation_input.get("selected_cohort_complete") is True
        and set(selected_pages) <= trusted_pages
    )
    dataset_ready = (
        bool(selected_pages)
        and lineage_coverage >= 0.95
        and labels_complete
        and adjudication_complete
        and split["immutable_manifest_sha256"] is not None
        and split["package_leakage_count"] == 0
    )

    experiment_rows = [{**row, "verdict": experiment_verdict(row)} for row in experiments]
    regression = {
        "status": "PASS"
        if holdout["status"] == "MEASURED" and holdout["critical_false_accepts"] == 0
        else "NOT_RUN",
        "critical_regression": None if holdout["status"] != "MEASURED" else False,
        "historical_closure_status": closure_final["status"],
    }
    gates = {
        "GATE_1_DATASET_READY": {"passed": dataset_ready},
        "GATE_2_EXTRACTION_READY": {
            "passed": holdout.get("raw_accuracy") is not None
            and holdout["raw_accuracy"] >= 0.98
            and holdout.get("critical_accuracy") is not None
            and holdout["critical_accuracy"] >= 0.99
        },
        "GATE_3_DECISION_READY": {
            "passed": holdout.get("accepted_precision") is not None
            and holdout["accepted_precision"] >= 0.995
            and holdout.get("critical_false_accepts") == 0
        },
        "GATE_4_HITL_STP_READY": {
            "passed": holdout.get("field_hitl") is not None
            and holdout["field_hitl"] < 0.10
            and holdout.get("claim_hitl") is not None
            and holdout["claim_hitl"] <= 0.10
            and holdout.get("claim_stp") is not None
            and holdout["claim_stp"] >= 0.90
        },
        "GATE_5_OPERATIONS_READY": {
            "passed": False,
            "reason": "OPERATIONAL_THRESHOLDS_AND_FULL_RUN_NOT_SUPPLIED",
        },
    }
    status = "EVALUATION_READY" if dataset_ready else "NOT_EVALUATION_READY"
    if all(g["passed"] for g in gates.values()):
        status = "PRODUCTION_READY"

    visual_latencies = sorted(
        float(row["visual_latency_ms"])
        for row in runtime_rows
        if runtime_valid
        and isinstance(row.get("visual_latency_ms"), (int, float))
        and row["visual_latency_ms"] >= 0
    )
    performance = {**perf_seed, "full_archive_operational_run": False}
    if len(visual_latencies) == len(runtime_rows) and visual_latencies:

        def percentile(fraction: float) -> float:
            return visual_latencies[round((len(visual_latencies) - 1) * fraction)]

        total_visual_ms = sum(visual_latencies)
        performance.update(
            {
                "status": "PARTIAL_MEASURED",
                "inventory_only": False,
                "measured_stage": "VISUAL_PAGE_CLASSIFICATION",
                "measured_pages": len(visual_latencies),
                "mean_classification_latency_ms": total_visual_ms / len(visual_latencies),
                "p50_classification_latency_ms": percentile(0.50),
                "p95_classification_latency_ms": percentile(0.95),
                "p99_classification_latency_ms": percentile(0.99),
                "classification_throughput_pages_per_second": (
                    len(visual_latencies) / (total_visual_ms / 1000) if total_visual_ms else None
                ),
                "ocr_executed": False,
                "field_extraction_executed": False,
                "claim_processing_executed": False,
                "reason": "FULL_VISUAL_CLASSIFICATION_MEASURED; END_TO_END_PIPELINE_NOT_RUN",
            }
        )
    package_request = [
        {
            "package_id": r["package_id"],
            "requested": [
                "source_claim_or_control_id",
                "bundle_id",
                "document_ids",
                "page_ordering_metadata",
                "attachment_relationships",
            ],
        }
        for r in package_seed["packages"]
        if r["package_id"] not in confirmed_identity
    ]
    outputs: dict[str, Any] = {
        "source_inventory.json": source,
        "page_classification_candidates.json": {
            "counts": dict(sorted(candidate_counts.items())),
            "records": candidate_rows,
            "candidate_is_ground_truth": False,
            "source": candidate_source,
            "runtime_manifest_valid": runtime_valid,
        },
        "document_boundary_candidates.json": {
            "candidates": boundary_candidates,
            "reviews": boundary_reviews,
            "documents_confirmed": boundary_confirmed,
            "source": boundary_source,
            "runtime_manifest_valid": runtime_boundaries_valid,
        },
        "review_progress.json": {
            "pages_total": len(candidate_rows),
            "pages_reviewed": len(reviewed),
            "documents_confirmed": boundary_confirmed,
        },
        "attachment_semantics.json": {
            "records": [
                r for r in reviews if r.get("class") in {"ATTACHMENT", "SUPPORTING_DOCUMENT"}
            ],
            "unknown_allowed": True,
        },
        "package_identity.json": {
            "states": ["CONFIRMED", "CANDIDATE", "AMBIGUOUS", "UNAVAILABLE"],
            "confirmed": len(confirmed_identity),
            "records": identities,
        },
        "package_mapping_request.json": {
            "archive_sha256": source["archive"]["sha256"],
            "requests": package_request,
        },
        "annotation_manifest.json": {
            "blind_mode": True,
            "prediction_visible_during_initial_annotation": False,
            "records": [_safe_annotation(row) for row in annotations],
            "trusted_labels": len(trusted),
            "selected_cohort_complete": labels_complete,
        },
        "annotation_agreement.json": {
            "critical_field_instances": len(pairs),
            "counts": dict(sorted(agreement_counts.items())),
            "agreement_rate": agreement_counts["AGREED_PENDING_FINALIZATION"] / len(pairs)
            if pairs
            else None,
        },
        "adjudication_log.json": {
            "required": agreement_counts["ADJUDICATION_REQUIRED"],
            "completed": len(final_adjudications),
            "records": [_safe_annotation(row) for row in final_adjudications],
        },
        "evaluation_cohort.json": {
            "selected_page_ids": selected_pages,
            "size": len(selected_pages),
            "selection_unit": "PACKAGE_WHERE_AVAILABLE",
        },
        "dataset_split_manifest.json": split,
        "source_to_cdp_binding.json": binding,
        "pipeline_coverage.json": coverage,
        "development_baseline.json": dev,
        "holdout_baseline.json": holdout,
        "performance_baseline.json": performance,
        "real_error_cohorts.json": {
            "status": "BLOCKED_NO_TRUSTED_BASELINE"
            if dev["status"] != "MEASURED"
            else "READY_FOR_ANALYSIS",
            "cohorts": [],
        },
        "optimization_experiments.json": {
            "holdout_tuning_prohibited": True,
            "experiments": experiment_rows,
        },
        "regression_analysis.json": regression,
        "authoritative_data_request.json": {
            "status": "REQUESTED",
            "datasets": ["MEMBER_ELIGIBILITY_SNAPSHOT", "PROVIDER_MASTER_SNAPSHOT"],
            "synthetic_allowed": False,
            "required_metadata": [
                "source_system",
                "schema_version",
                "export_timestamp",
                "effective_dates",
                "file_hash",
            ],
        },
        "claim_closure_board.json": {
            "claim_metrics_available": bool(confirmed_identity),
            "rows": [
                {
                    "package_id": p["package_id"],
                    "claim_id": confirmed_identity.get(p["package_id"], {}).get("claim_id"),
                    "disposition": "ENGINEERING_REMAINING"
                    if p["package_id"] in confirmed_identity
                    else "IDENTITY_NOT_AVAILABLE",
                }
                for p in package_seed["packages"]
            ],
        },
        "readiness_gates.json": gates,
    }
    outputs["final_real_evaluation_report.json"] = {
        "status": status,
        "pages_processed": len(candidate_rows),
        "candidate_source": candidate_source,
        "runtime_manifest_valid": runtime_valid,
        "boundary_candidate_source": boundary_source,
        "boundary_runtime_manifest_valid": runtime_boundaries_valid,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "confirmed_class_counts": dict(sorted(reviewed_classes.items())),
        "documents_confirmed": boundary_confirmed,
        "package_identities_confirmed": len(confirmed_identity),
        "annotation_cohort_size": len(selected_pages),
        "trusted_labels_completed": len(trusted),
        "development_size": len(split["development_package_ids"]),
        "holdout_size": len(split["holdout_package_ids"]),
        "package_leakage": split["package_leakage_count"],
        "source_to_cdp_binding": binding["binding_coverage"],
        "pipeline_gap_count": len(coverage["gaps"]),
        "development_metrics": dev,
        "holdout_metrics": holdout,
        "readiness_gates": gates,
        "safety": {
            "thresholds_changed": False,
            "new_ocr_engine": False,
            "synthetic_truth_used": False,
            "raw_phi_written": False,
        },
    }
    for name, value in outputs.items():
        _write(output_dir, name, value)
    return outputs["final_real_evaluation_report.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-dir", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inputs-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_real_evaluation_program(args.closure_dir, args.output_dir, args.inputs_dir),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
