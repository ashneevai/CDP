"""Hash-safe, resumable and bounded-parallel strict identity replay.

OCR-bearing cache records live below ``evaluation_data`` and must never be
committed. Page checkpoints and published reports contain no recognized text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil
from PIL import Image, ImageDraw

from evaluation.real_archive_classification import (
    Observation,
    PageRef,
    RapidOCRPageObserver,
    _safe_record,
    discover_pages,
)
from packages.document_routing import (
    DocumentRoutingDecisionService,
    MultiSignalRoute,
    MultiSignalRouter,
)
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.ocr import RapidOCRProvider
from packages.processing_routes.contracts import ProcessingRoute
from packages.processing_routes.resolver import ProcessingRouteResolver
from packages.standard_form_verification.cms1500 import CMS1500Verifier
from packages.standard_form_verification.evidence import evidence_from_router_features
from packages.standard_form_verification.ub04 import UB04Verifier
from workers.page_detection.text_extraction import TextLine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "evaluation_data/source_b_1000_claims"
DEFAULT_OUTPUT = ROOT / "evaluation_data/strict_identity_replay_v3"
DEFAULT_LEGACY_OCR_CACHE = ROOT / "evaluation_data/strict_identity_replay_v2/ocr_cache"
DEFAULT_REPORT = ROOT / "evaluation_results/strict_identity_replay"
CACHE_KEY_VERSION = "strict-identity-ocr-cache-v2"
LEGACY_CACHE_KEY_VERSION = "strict-identity-ocr-cache-v1"
OCR_CONFIG_VERSION = "rapidocr-full-page-routing-v1"
PREPROCESSING_VERSION = "AUTO"
RUNNER_VERSION = "strict-identity-cached-replay-v2"
DECISION_SCHEMA_VERSION = "strict-identity-decision-v2"
MAX_OCR_RETRIES = 2
OCR_TIMEOUT_SECONDS = 120.0
WORKER_MAX_TASKS = 50
SUMMARY_INTERVAL = 25
_WORKER_OBSERVER: RapidOCRPageObserver | None = None
_WORKER_ROUTER: MultiSignalRouter | None = None


def _digest(value: bytes | str) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def rapidocr_version() -> str:
    try:
        return metadata.version("rapidocr-onnxruntime")
    except metadata.PackageNotFoundError:
        return "unknown"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


OCR_CONFIG_SHA256 = _digest(OCR_CONFIG_VERSION)
PREPROCESSING_CONFIG_SHA256 = _digest(PREPROCESSING_VERSION)
POLICY_FILES = (
    "config/document_routing.yaml",
    "packages/document_routing/router.py",
    "packages/document_routing/decision_service.py",
    "packages/document_routing/hierarchical.py",
    "packages/standard_form_verification/contracts.py",
    "packages/standard_form_verification/evidence.py",
    "packages/standard_form_verification/cms1500.py",
    "packages/standard_form_verification/ub04.py",
    "packages/processing_routes/resolver.py",
)


def decision_policy_manifest() -> dict[str, Any]:
    router = MultiSignalRouter.load()
    file_hashes = {name: _file_digest(ROOT / name) for name in POLICY_FILES}
    policy = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "source_tree_sha256": _digest(
            json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
        ),
        "policy_file_sha256": file_hashes,
        "router_code_sha256": file_hashes["packages/document_routing/router.py"],
        "router_config_sha256": file_hashes["config/document_routing.yaml"],
        "router_version": router.config["router_version"],
        "identity_policy_version": router.config["form_identity"]["policy_version"],
        "cms1500_verifier_policy_version": CMS1500Verifier.policy_version,
        "ub04_verifier_policy_version": UB04Verifier.policy_version,
        "processing_route_policy_version": ProcessingRouteResolver.policy_version,
        "decision_service_version": DocumentRoutingDecisionService.version,
    }
    policy["decision_policy_sha256"] = _digest(
        json.dumps(policy, sort_keys=True, separators=(",", ":"))
    )
    return policy


def ocr_cache_key(page: PageRef, engine_version: str) -> str:
    payload = {
        "cache_key_version": CACHE_KEY_VERSION,
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
        "ocr_config_sha256": OCR_CONFIG_SHA256,
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def legacy_ocr_cache_key(page: PageRef, engine_version: str) -> str:
    payload = {
        "cache_key_version": LEGACY_CACHE_KEY_VERSION,
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "ocr_config_version": OCR_CONFIG_VERSION,
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def decision_checkpoint_key(page: PageRef, policy: dict[str, Any]) -> str:
    payload = {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "decision_policy_sha256": policy["decision_policy_sha256"],
        "decision_schema_version": policy["decision_schema_version"],
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _expected_provenance(page: PageRef, engine_version: str) -> dict[str, Any]:
    return {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
        "ocr_config_version": OCR_CONFIG_VERSION,
        "ocr_config_sha256": OCR_CONFIG_SHA256,
        "cache_key": ocr_cache_key(page, engine_version),
    }


def _tokens_are_valid(record: dict[str, Any]) -> bool:
    tokens = record.get("tokens")
    return isinstance(tokens, list) and all(
        isinstance(token, dict)
        and isinstance(token.get("text"), str)
        and isinstance(token.get("bbox"), list)
        and len(token["bbox"]) == 4
        and isinstance(token.get("confidence"), (int, float))
        for token in tokens
    )


def valid_cache_record(record: dict[str, Any], page: PageRef, engine_version: str) -> bool:
    expected = _expected_provenance(page, engine_version)
    return (
        all(record.get(key) == value for key, value in expected.items())
        and record.get("status") in {"OCR_EXECUTED", "CACHE_HIT"}
        and _tokens_are_valid(record)
    )


def valid_legacy_cache_record(record: dict[str, Any], page: PageRef, engine_version: str) -> bool:
    expected = {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "ocr_config_version": OCR_CONFIG_VERSION,
        "cache_key": legacy_ocr_cache_key(page, engine_version),
    }
    return (
        all(record.get(key) == value for key, value in expected.items())
        and record.get("status") in {"OCR_EXECUTED", "CACHE_HIT"}
        and _tokens_are_valid(record)
    )


def observation_from_cache(record: dict[str, Any]) -> Observation:
    return Observation(
        lines=tuple(
            TextLine(token["text"], *token["bbox"], float(token["confidence"]))
            for token in record["tokens"]
        ),
        latency_ms=float(record.get("runtime_ms", 0.0)),
        engine=record["ocr_engine"],
        engine_version=record["ocr_engine_version"],
        cache_hit=True,
    )


def cache_record(page: PageRef, observation: Observation, engine_version: str) -> dict[str, Any]:
    return {
        **_expected_provenance(page, engine_version),
        "schema_version": "1.0",
        "source_asset_path": str(page.asset_path.resolve().relative_to(ROOT)),
        "source_page_id": page.page_id,
        "tokens": [
            {
                "text": line.text,
                "bbox": [line.x0, line.y0, line.x1, line.y1],
                "confidence": line.confidence,
            }
            for line in observation.lines
        ],
        "runtime_ms": observation.latency_ms,
        "cache_source": "LOCAL_REAL_SOURCE_REPLAY",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "OCR_EXECUTED",
    }


def valid_page_checkpoint(
    record: dict[str, Any],
    page: PageRef,
    engine_version: str,
    policy: dict[str, Any] | None = None,
) -> bool:
    provenance = record.get("ocr_provenance", {})
    common_ocr = {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
    }
    if any(provenance.get(key) != value for key, value in common_ocr.items()):
        return False
    decision = record.get("decision_provenance", {})
    if policy is not None:
        if decision.get("decision_checkpoint_key") != decision_checkpoint_key(page, policy):
            return False
        if decision.get("decision_policy_sha256") != policy["decision_policy_sha256"]:
            return False
        if any(decision.get(key) != value for key, value in policy.items()):
            return False
    chain = record.get("production_chain")
    return (
        record.get("source_page_id") == page.page_id
        and record.get("source_page_sha256") == page.page_sha256
        and isinstance(record.get("candidate_class"), str)
        and isinstance(record.get("form_identity"), dict)
        and isinstance(record.get("routing_result"), dict)
        and isinstance(chain, dict)
        and isinstance(chain.get("processing_route"), str)
        and isinstance(chain.get("fixed_extractor_authorized"), bool)
        and isinstance(chain.get("localization_authorized"), bool)
        and chain.get("actual_localization_invoked") is False
    )


def safe_worker_count(requested: int, *, free_memory_mb: int, logical_cpus: int) -> int:
    configured = max(1, min(requested, 8))
    memory_cap = max(1, free_memory_mb // 768)
    cpu_cap = max(1, logical_cpus // 2)
    return min(configured, memory_cap, cpu_cap)


def available_memory_mb() -> int:
    return int(psutil.virtual_memory().available // (1024 * 1024))


def _worker_init() -> None:
    global _WORKER_OBSERVER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    _WORKER_OBSERVER = RapidOCRPageObserver(RapidOCRProvider(session_threads=1))


def _load_page(page: PageRef) -> Image.Image:
    with Image.open(page.asset_path) as source:
        source.seek(page.page_number - 1)
        image = source.copy()
    digest = hashlib.sha256()
    digest.update(f"{image.mode}|{image.width}|{image.height}|".encode())
    digest.update(image.tobytes())
    if digest.hexdigest() != page.page_sha256:
        raise ValueError("RENDERED_PAGE_HASH_MISMATCH")
    return image


def _fresh_ocr(page: PageRef) -> dict[str, Any]:
    if _WORKER_OBSERVER is None:
        _worker_init()
    assert _WORKER_OBSERVER is not None
    observation = asyncio.run(_WORKER_OBSERVER(_load_page(page), page))
    return {
        "lines": [asdict(line) for line in observation.lines],
        "latency_ms": observation.latency_ms,
        "engine": observation.engine,
        "engine_version": observation.engine_version,
    }


def _observation(value: dict[str, Any]) -> Observation:
    return Observation(
        tuple(TextLine(**line) for line in value["lines"]),
        float(value["latency_ms"]),
        value["engine"],
        value["engine_version"],
        False,
    )


def _page_result(
    page: PageRef,
    image: Image.Image,
    observation: Observation,
    router: MultiSignalRouter,
    engine_version: str,
    execution_status: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    routing = router.route(image, list(observation.lines))
    result = _safe_record(page, image, observation, router, routing)
    result["schema_version"] = DECISION_SCHEMA_VERSION
    result["ocr_provenance"] = {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": page.page_number - 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
        "ocr_config_sha256": OCR_CONFIG_SHA256,
        "status": execution_status,
    }
    standard_evidence = None
    if routing.route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
        standard_evidence = evidence_from_router_features(
            DocumentClass(routing.route.value), None, routing
        )
    chain = DocumentRoutingDecisionService().decide(
        page.package_id,
        page.page_id,
        routing,
        standard_evidence,
        evaluation_only=True,
    )
    verification = chain.standard_verification
    fixed_routes = {
        ProcessingRoute.CMS_STANDARD_EXTRACTOR,
        ProcessingRoute.UB_STANDARD_EXTRACTOR,
    }
    fixed_authorized = chain.processing_route in fixed_routes
    localization_authorized = fixed_authorized and routing.localization_allowed
    result["form_identity"]["localization_allowed"] = localization_authorized
    result["routing_result"] = {
        "router_nomination": routing.route.value,
        "identity_state": routing.identity_state,
        "router_localization_eligible": routing.localization_allowed,
    }
    result["production_chain"] = {
        "classification_standard_candidate": chain.classification.standard_candidate,
        "classification_document_subtype": chain.classification.document_subtype.value,
        "verification_status": verification.status.value if verification else None,
        "verified_identity_family": (
            verification.form_identity.family.value
            if verification and verification.form_identity.family
            else None
        ),
        "fixed_extractor_authorized": fixed_authorized,
        "processing_route": chain.processing_route.value,
        "localization_authorized": localization_authorized,
        "actual_localization_invoked": False,
        "decision_reason_codes": list(chain.route_reason_codes),
    }
    result["decision_provenance"] = {
        **policy,
        "decision_checkpoint_key": decision_checkpoint_key(page, policy),
    }
    return result


def _cached_decision(
    page: PageRef,
    observation: Observation,
    engine_version: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    global _WORKER_ROUTER
    if _WORKER_ROUTER is None:
        _WORKER_ROUTER = MultiSignalRouter.load()
    return _page_result(
        page,
        _load_page(page),
        observation,
        _WORKER_ROUTER,
        engine_version,
        "CACHE_HIT",
        policy,
    )


def _failure_result(
    page: PageRef,
    engine_version: str,
    error: BaseException,
    policy: dict[str, Any] | None = None,
    attempts: int = 1,
) -> dict[str, Any]:
    policy = policy or decision_policy_manifest()
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_page_id": page.page_id,
        "source_page_sha256": page.page_sha256,
        "ocr_provenance": {
            "source_asset_sha256": page.asset_sha256,
            "frame_index": page.page_number - 1,
            "rendered_page_sha256": page.page_sha256,
            "ocr_engine": "rapidocr",
            "ocr_engine_version": engine_version,
            "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
            "ocr_config_sha256": OCR_CONFIG_SHA256,
            "status": "OCR_FAILED",
        },
        "decision_provenance": {
            **policy,
            "decision_checkpoint_key": decision_checkpoint_key(page, policy),
        },
        "failure": {
            "error_type": type(error).__name__,
            "message_persisted": False,
            "attempts": attempts,
        },
        "reason_codes": ["OBSERVATION_FAILED"],
    }


def _summary(
    pages: Sequence[PageRef | None],
    records: list[dict[str, Any]],
    started: float,
    workers: int,
    *,
    failed_records: list[dict[str, Any]] | None = None,
    run_stats: dict[str, int] | None = None,
    prior_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    failed_records = failed_records or []
    run_stats = run_stats or {}
    elapsed = prior_elapsed_seconds + max(time.perf_counter() - started, 0.0)
    successful = len(records)
    failed = len(failed_records)
    counts = Counter(record["candidate_class"] for record in records)
    identity_classes = (
        "CMS1500",
        "UB04",
        "OTHER_CLAIM_FORM",
        "UNKNOWN",
        "NON_CLAIM",
    )
    identity_distribution = {name: counts[name] for name in identity_classes}
    identity_distribution.update(
        {name: count for name, count in sorted(counts.items()) if name not in identity_distribution}
    )
    statuses = Counter(record["ocr_provenance"]["status"] for record in records)
    latencies = [float(record.get("ocr", {}).get("latency_ms", 0.0)) for record in records]
    sorted_latency = sorted(latencies)

    def percentile(q: float) -> float | None:
        if not sorted_latency:
            return None
        return sorted_latency[min(len(sorted_latency) - 1, math.ceil(q * len(sorted_latency)) - 1)]

    rate = successful / elapsed * 60 if elapsed > 0 else 0.0
    remaining = max(0, len(pages) - successful)
    fixed = [
        record for record in records if record["production_chain"]["fixed_extractor_authorized"]
    ]
    router_nominations = Counter(
        record["routing_result"]["router_nomination"] for record in records
    )
    verified_identities = Counter(
        record["production_chain"]["verified_identity_family"]
        for record in records
        if record["production_chain"]["verified_identity_family"]
        and record["production_chain"]["verification_status"] == "VERIFIED"
    )
    fixed_authorizations = Counter(
        record["production_chain"]["processing_route"] for record in fixed
    )
    policy_hashes = {record["decision_provenance"]["decision_policy_sha256"] for record in records}
    page_ids = {record["source_page_id"] for record in records}
    failed_page_ids = {record["source_page_id"] for record in failed_records}
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "total_pages_discovered": len(pages),
        "successful_pages": successful,
        "accounted_pages": len(page_ids | failed_page_ids),
        "failed_pages": failed,
        "unique_page_ids": len(page_ids | failed_page_ids),
        "pages_completed": successful,
        "pages_remaining": remaining,
        "checkpoint_hits": run_stats.get("checkpoint_hits", 0),
        "ocr_cache_hits": statuses["CACHE_HIT"],
        "cache_hits": statuses["CACHE_HIT"],
        "cache_hit_rate": statuses["CACHE_HIT"] / successful if successful else 0.0,
        "fresh_ocr_executions": statuses["OCR_EXECUTED"],
        "ocr_failures": failed,
        "retry_attempts": run_stats.get("retry_attempts", 0),
        "retries": run_stats.get("retry_attempts", 0),
        "worker_restarts": run_stats.get("worker_restarts", 0),
        "stale_decisions_rejected": run_stats.get("stale_decisions_rejected", 0),
        "stale_ocr_records_rejected": run_stats.get("stale_ocr_records_rejected", 0),
        "identity_distribution": identity_distribution,
        "router_nominations": dict(sorted(router_nominations.items())),
        "verified_canonical_identities": dict(sorted(verified_identities.items())),
        "fixed_extractor_authorizations": dict(sorted(fixed_authorizations.items())),
        "localization_authorizations": sum(
            record["production_chain"]["localization_authorized"] for record in records
        ),
        "actual_localization_invocations": sum(
            record["production_chain"]["actual_localization_invoked"] for record in records
        ),
        "cms1500_localization_authorizations": sum(
            record["production_chain"]["processing_route"] == "CMS_STANDARD_EXTRACTOR"
            and record["production_chain"]["localization_authorized"]
            for record in records
        ),
        "ub04_localization_authorizations": sum(
            record["production_chain"]["processing_route"] == "UB_STANDARD_EXTRACTOR"
            and record["production_chain"]["localization_authorized"]
            for record in records
        ),
        "other_claim_form_localization_authorizations": sum(
            record["candidate_class"] == "OTHER_CLAIM_FORM"
            and record["production_chain"]["localization_authorized"]
            for record in records
        ),
        "unknown_localization_authorizations": sum(
            record["candidate_class"] in {"UNKNOWN", "SUPPORTING_DOCUMENT"}
            and record["production_chain"]["localization_authorized"]
            for record in records
        ),
        "family_mismatch_blocks": sum(
            "STANDARD_IDENTITY_CLASSIFICATION_MISMATCH"
            in record["production_chain"]["decision_reason_codes"]
            for record in records
        ),
        "structured_fallbacks": sum(
            record["production_chain"]["processing_route"] == "LAYOUT_STRUCTURED_EXTRACTOR"
            for record in records
        ),
        "unknown_or_abstained_pages": sum(
            record["production_chain"]["processing_route"] == "SAFE_UNKNOWN"
            or record["candidate_class"] in {"UNKNOWN", "SUPPORTING_DOCUMENT"}
            for record in records
        ),
        "review_required_pages": sum(
            record.get("confidence_band") in {"REVIEW_REQUIRED", "UNKNOWN"}
            or record["production_chain"]["processing_route"] == "SAFE_UNKNOWN"
            for record in records
        ),
        "conflicting_identity_evidence": sum(
            any(
                bool(values)
                for values in record["form_identity"].get("conflicting_anchors", {}).values()
            )
            for record in records
        ),
        "decision_policy_hashes": sorted(policy_hashes),
        "all_decision_policy_hashes_identical": len(policy_hashes) <= 1,
        "worker_count": workers,
        "wall_clock_seconds": elapsed,
        "effective_pages_per_minute": rate,
        "eta_seconds": remaining / rate * 60 if rate else None,
        "mean_ocr_runtime_ms": statistics.fmean(latencies) if latencies else None,
        "p50_ocr_runtime_ms": percentile(0.50),
        "p95_ocr_runtime_ms": percentile(0.95),
        "p99_ocr_runtime_ms": percentile(0.99),
        "peak_memory_mb": None,
        "peak_memory_status": "NOT_CAPTURED",
        "trusted_label_metrics": "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "critical_false_authorizations": "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "projected_hitl_rate": "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "cost_per_page": "NOT_MEASURED",
        "production_promotion": "NOT_AUTHORIZED",
    }


def _canaries(router: MultiSignalRouter) -> list[dict[str, Any]]:
    image = Image.new("L", (1000, 1300), 255)
    draw = ImageDraw.Draw(image)
    for y in range(150, 1100, 100):
        draw.line((40, y, 960, y), fill=0, width=2)
    fixtures = [
        ("REIMBURSEMENT REQUEST", "PATIENT PROVIDER CLAIM", "TYPE OF BILL", "TOTAL CHARGES"),
        ("PROPRIETARY CLAIM FORM", "PATIENT CONTROL", "REVENUE CODE", "HCPCS UNITS TOTAL CHARGES"),
        ("LEGACY CLAIM FORM", "STATEMENT COVERS", "MEDICAL RECORD", "PRINCIPAL DIAGNOSIS"),
    ]
    results = []
    for index, values in enumerate(fixtures, 1):
        lines = [
            TextLine(value, 10, offset * 30, 500, offset * 30 + 20, 0.95)
            for offset, value in enumerate(values)
        ]
        routing = router.route(image, lines)
        decision = DocumentRoutingDecisionService().decide("canary", str(index), routing)
        results.append(
            {
                "canary": index,
                "router_route": routing.route.value,
                "processing_route": decision.processing_route.value,
                "ub04_rejected": decision.processing_route != ProcessingRoute.UB_STANDARD_EXTRACTOR,
                "ub04_localization_authorizations": int(
                    decision.processing_route == ProcessingRoute.UB_STANDARD_EXTRACTOR
                    and routing.localization_allowed
                ),
            }
        )
    return results


def _terminate_executor(pool: ProcessPoolExecutor) -> None:
    terminate = getattr(pool, "terminate_workers", None)
    if callable(terminate):
        terminate()
        return
    for process in getattr(pool, "_processes", {}).values():
        if process.is_alive():
            process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


def _publish_report(report_dir: Path, summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "final_report.json", summary)
    markdown = [
        "# Strict identity replay final report",
        "",
        f"- Successful pages: {summary['successful_pages']}",
        f"- Failed pages: {summary['failed_pages']}",
        f"- Input manifest SHA-256: {summary['manifest']['input_manifest_sha256']}",
        f"- Decision policy SHA-256: {summary['manifest']['decision_policy']['decision_policy_sha256']}",
        f"- Decision checkpoint hits: {summary['checkpoint_hits']}",
        f"- OCR cache hits: {summary['ocr_cache_hits']}",
        f"- Fresh OCR pages: {summary['fresh_ocr_executions']}",
        f"- Retry attempts: {summary['retry_attempts']}",
        f"- Worker restarts: {summary['worker_restarts']}",
        f"- Fixed-extractor authorizations: {json.dumps(summary['fixed_extractor_authorizations'], sort_keys=True)}",
        f"- Localization authorizations: {summary['localization_authorizations']}",
        f"- Actual localization invocations: {summary['actual_localization_invocations']}",
        f"- Critical routing violations: {summary['critical_routing_violations']}",
        f"- P95 OCR runtime (ms): {summary['p95_ocr_runtime_ms']}",
        "- Trusted-label accuracy: NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "- Critical false authorizations: NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "- Projected HITL rate: NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        "- Cost per page: NOT_MEASURED",
        "- Production promotion: NOT_AUTHORIZED",
        "",
        "OCR-bearing cache records remain local under evaluation_data and are not committed.",
    ]
    (report_dir / "final_report.md").write_text("\n".join(markdown) + "\n", "utf-8")


def _strict_completion(summary: dict[str, Any]) -> bool:
    return (
        summary["successful_pages"] == summary["total_pages_discovered"]
        and summary["accounted_pages"] == summary["total_pages_discovered"]
        and summary["unique_page_ids"] == summary["total_pages_discovered"]
        and summary["failed_pages"] == 0
        and summary["ocr_failures"] == 0
        and summary["all_decision_policy_hashes_identical"]
        and len(summary["decision_policy_hashes"]) == 1
        and summary["decision_policy_hashes"][0]
        == summary["manifest"]["decision_policy"]["decision_policy_sha256"]
    )


def run_replay(
    source: Path,
    output: Path,
    report_dir: Path,
    archive_sha256: str,
    *,
    workers: int,
    limit: int | None = None,
    legacy_cache_dir: Path | None = DEFAULT_LEGACY_OCR_CACHE,
    retries: int = MAX_OCR_RETRIES,
    timeout_seconds: float = OCR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    router = MultiSignalRouter.load()
    policy = decision_policy_manifest()
    canaries = _canaries(router)
    if not all(
        item["ub04_rejected"] and item["ub04_localization_authorizations"] == 0 for item in canaries
    ):
        raise RuntimeError("FALSE_UB04_CANARY_FAILED")

    engine_version = rapidocr_version()
    pages = [page for page, _ in discover_pages(source, archive_sha256)]
    if limit is not None:
        pages = pages[:limit]
    output.mkdir(parents=True, exist_ok=True)
    page_dir = output / "pages"
    failure_dir = output / "failures"
    cache_dir = output / "ocr_cache"
    for directory in (page_dir, failure_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_hash = _digest(
        "\n".join(f"{p.asset_sha256}:{p.page_number - 1}:{p.page_sha256}" for p in pages)
    )
    manifest = {
        "schema_version": "2.0",
        "runner_version": RUNNER_VERSION,
        "archive_sha256": archive_sha256,
        "assets": len({page.asset_id for page in pages}),
        "rendered_pages": len(pages),
        "package_count": len({page.package_id for page in pages}),
        "input_manifest_sha256": input_hash,
        "cache_key_version": CACHE_KEY_VERSION,
        "legacy_cache_key_version": LEGACY_CACHE_KEY_VERSION,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": engine_version,
        "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
        "ocr_config_sha256": OCR_CONFIG_SHA256,
        "decision_policy": policy,
    }
    atomic_write_json(output / "manifest.json", manifest)

    previous_state = {}
    state_path = output / "run_state.json"
    if state_path.exists():
        try:
            previous_state = json.loads(state_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_state = {}
    prior_elapsed = float(previous_state.get("cumulative_elapsed_seconds", 0.0))
    started = time.perf_counter()
    stats = {
        "checkpoint_hits": 0,
        "retry_attempts": 0,
        "worker_restarts": 0,
        "stale_decisions_rejected": 0,
        "stale_ocr_records_rejected": 0,
    }
    records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    pending: list[PageRef] = []

    for page in pages:
        checkpoint_path = page_dir / f"{page.page_id}.json"
        if checkpoint_path.exists():
            try:
                record = json.loads(checkpoint_path.read_text("utf-8"))
                if valid_page_checkpoint(record, page, engine_version, policy):
                    records.append(record)
                    stats["checkpoint_hits"] += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            stats["stale_decisions_rejected"] += 1
        pending.append(page)

    def complete(page: PageRef, observation: Observation, status: str) -> None:
        image = _load_page(page)
        record = _page_result(page, image, observation, router, engine_version, status, policy)
        atomic_write_json(page_dir / f"{page.page_id}.json", record)
        (failure_dir / f"{page.page_id}.json").unlink(missing_ok=True)
        records.append(record)

    fresh: deque[PageRef] = deque()
    cached_work: list[tuple[PageRef, Observation]] = []
    cache_locations = [cache_dir]
    if legacy_cache_dir is not None:
        cache_locations.append(legacy_cache_dir)
    for page in pending:
        found = False
        for directory in cache_locations:
            key = (
                ocr_cache_key(page, engine_version)
                if directory == cache_dir
                else legacy_ocr_cache_key(page, engine_version)
            )
            cache_path = directory / f"{key}.json"
            if not cache_path.exists():
                continue
            try:
                cached = json.loads(cache_path.read_text("utf-8"))
                validator = (
                    valid_cache_record if directory == cache_dir else valid_legacy_cache_record
                )
                if validator(cached, page, engine_version):
                    cached_work.append((page, observation_from_cache(cached)))
                    found = True
                    break
            except (OSError, json.JSONDecodeError, ValueError):
                pass
            stats["stale_ocr_records_rejected"] += 1
        if not found:
            fresh.append(page)

    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=WORKER_MAX_TASKS,
    ) as decision_pool:
        futures = {
            decision_pool.submit(_cached_decision, page, observation, engine_version, policy): page
            for page, observation in cached_work
        }
        for future in as_completed(futures):
            page = futures[future]
            try:
                record = future.result()
                atomic_write_json(page_dir / f"{page.page_id}.json", record)
                (failure_dir / f"{page.page_id}.json").unlink(missing_ok=True)
                records.append(record)
            except BaseException as error:  # noqa: BLE001 - isolate decision failures
                failure = _failure_result(page, engine_version, error, policy)
                atomic_write_json(failure_dir / f"{page.page_id}.json", failure)
                failed_records.append(failure)

    while fresh:
        page = fresh.popleft()
        last_error: BaseException | None = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt + 1
            pool = ProcessPoolExecutor(
                max_workers=1,
                initializer=_worker_init,
                max_tasks_per_child=WORKER_MAX_TASKS,
            )
            future = pool.submit(_fresh_ocr, page)
            try:
                value = future.result(timeout=timeout_seconds)
                pool.shutdown(wait=True)
                observation = _observation(value)
                cached_record = cache_record(page, observation, engine_version)
                atomic_write_json(cache_dir / f"{cached_record['cache_key']}.json", cached_record)
                complete(page, observation, "OCR_EXECUTED")
                last_error = None
                break
            except BaseException as error:  # noqa: BLE001 - isolate page failures
                last_error = error
                stats["worker_restarts"] += 1
                _terminate_executor(pool)
                if attempt < retries:
                    stats["retry_attempts"] += 1
        if last_error is not None:
            failure = _failure_result(page, engine_version, last_error, policy, attempts)
            atomic_write_json(failure_dir / f"{page.page_id}.json", failure)
            failed_records.append(failure)
        if (len(records) + len(failed_records)) % SUMMARY_INTERVAL == 0:
            partial = _summary(
                pages,
                records,
                started,
                workers,
                failed_records=failed_records,
                run_stats=stats,
                prior_elapsed_seconds=prior_elapsed,
            )
            partial.update({"manifest": manifest, "canaries": canaries, "complete": False})
            atomic_write_json(output / "summary_partial.json", partial)

    records.sort(key=lambda item: item["source_page_id"])
    failed_records = []
    for path in sorted(failure_dir.glob("*.json")):
        try:
            failure = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if failure.get("source_page_id") in {page.page_id for page in pages}:
            failed_records.append(failure)

    summary = _summary(
        pages,
        records,
        started,
        workers,
        failed_records=failed_records,
        run_stats=stats,
        prior_elapsed_seconds=prior_elapsed,
    )
    summary.update(
        {
            "manifest": manifest,
            "canaries": canaries,
            "critical_routing_violations": (
                summary["other_claim_form_localization_authorizations"]
                + summary["unknown_localization_authorizations"]
                + sum(not item["ub04_rejected"] for item in canaries)
            ),
            "real_data_classification_accuracy": "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        }
    )
    summary["complete"] = _strict_completion(summary)
    summary["all_input_pages_accounted_for"] = summary["accounted_pages"] == len(pages) and summary[
        "unique_page_ids"
    ] == len(pages)
    atomic_write_json(output / "summary_partial.json", summary)
    atomic_write_json(
        state_path,
        {
            "runner_version": RUNNER_VERSION,
            "input_manifest_sha256": input_hash,
            "decision_policy_sha256": policy["decision_policy_sha256"],
            "cumulative_elapsed_seconds": summary["wall_clock_seconds"],
            "successful_pages": summary["successful_pages"],
            "failed_pages": summary["failed_pages"],
        },
    )
    if summary["complete"]:
        _publish_report(report_dir, summary)
    return summary


def finalize_existing_replay(output: Path, report_dir: Path) -> dict[str, Any]:
    """Recompute every completion gate and publish without rerunning OCR."""
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    original = json.loads((output / "summary_partial.json").read_text("utf-8"))
    total = int(manifest["rendered_pages"])
    records = [
        json.loads(path.read_text("utf-8")) for path in sorted((output / "pages").glob("*.json"))
    ]
    failures = [
        json.loads(path.read_text("utf-8")) for path in sorted((output / "failures").glob("*.json"))
    ]
    policy = manifest["decision_policy"]
    policy_hash = policy["decision_policy_sha256"]
    valid_records = [
        record
        for record in records
        if record.get("decision_provenance", {}).get("decision_policy_sha256") == policy_hash
        and record.get("schema_version") == DECISION_SCHEMA_VERSION
        and isinstance(record.get("production_chain"), dict)
    ]
    run_stats = {
        "checkpoint_hits": int(original.get("checkpoint_hits", 0)),
        "retry_attempts": int(original.get("retry_attempts", 0)),
        "worker_restarts": int(original.get("worker_restarts", 0)),
        "stale_decisions_rejected": int(original.get("stale_decisions_rejected", 0)),
        "stale_ocr_records_rejected": int(original.get("stale_ocr_records_rejected", 0)),
    }
    started = time.perf_counter()
    summary = _summary(
        [None] * total,
        valid_records,
        started,
        int(original["worker_count"]),
        failed_records=failures,
        run_stats=run_stats,
        prior_elapsed_seconds=float(original.get("wall_clock_seconds", 0.0)),
    )
    summary.update(
        {
            "manifest": manifest,
            "canaries": original.get("canaries", []),
            "critical_routing_violations": (
                summary["other_claim_form_localization_authorizations"]
                + summary["unknown_localization_authorizations"]
                + sum(not item.get("ub04_rejected", False) for item in original.get("canaries", []))
            ),
            "real_data_classification_accuracy": "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS",
        }
    )
    summary["complete"] = _strict_completion(summary)
    summary["all_input_pages_accounted_for"] = (
        summary["accounted_pages"] == total and summary["unique_page_ids"] == total
    )
    if not summary["complete"]:
        raise RuntimeError("INCOMPLETE_OR_FAILED_REPLAY_CANNOT_BE_FINALIZED")
    memory_path = output / "memory_peak.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text("utf-8"))
        summary["peak_memory_mb"] = round(
            float(memory["peak_worker_tree_memory_bytes"]) / (1024 * 1024), 3
        )
        summary["peak_memory_status"] = "OBSERVED_DURING_PARTIAL_REPLAY_WINDOW"
    atomic_write_json(output / "summary_partial.json", summary)
    _publish_report(report_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument(
        "--workers", type=int, default=int(os.getenv("STRICT_IDENTITY_REPLAY_WORKERS", "4"))
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--legacy-cache-dir", type=Path, default=DEFAULT_LEGACY_OCR_CACHE)
    parser.add_argument("--retries", type=int, default=MAX_OCR_RETRIES)
    parser.add_argument("--timeout-seconds", type=float, default=OCR_TIMEOUT_SECONDS)
    args = parser.parse_args()
    workers = safe_worker_count(
        args.workers,
        free_memory_mb=available_memory_mb(),
        logical_cpus=os.cpu_count() or 1,
    )
    print(
        json.dumps(
            run_replay(
                args.source,
                args.output,
                args.report_dir,
                args.archive_sha256,
                workers=workers,
                limit=args.limit,
                legacy_cache_dir=args.legacy_cache_dir,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
