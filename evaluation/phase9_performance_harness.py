"""Truth-blind, resumable Phase 9A performance development harness."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from evaluation.phase10_stratified_pilot import select
from evaluation.run_production_holdout_v2 import infer


STAGE_NAMES = (
    "preparation", "classification", "registration", "ocr", "layout",
    "evidence", "claim_decision", "image_decode", "orientation_detection",
    "orientation_apply", "skew_detection", "deskew", "denoise",
)
DEVELOPMENT_ALLOCATION = {"Group A": 15, "Group B": 6, "Group C": 5, "Group D": 4}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "total": sum(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, .50),
        "p95": _percentile(values, .95),
        "p99": _percentile(values, .99),
    }


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _configure_worker(thread_cap: int) -> None:
    if thread_cap <= 0:
        return
    cap = str(thread_cap)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "ORT_NUM_THREADS"):
        os.environ[name] = cap
    os.environ["CDP_INTERNAL_THREAD_CAP"] = cap
    try:
        import cv2
        cv2.setNumThreads(thread_cap)
    except ImportError:
        pass


def _run_page(task: dict[str, Any]) -> dict[str, Any]:
    ordinal = int(task["ordinal"])
    before_rss = _rss_bytes()
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    try:
        predictions = infer(
            Path(task["dataset"]), Path(task["worker_output"]),
            limit=1, offset=ordinal,
        )
        if len(predictions) != 1:
            raise RuntimeError(f"EXPECTED_ONE_PREDICTION:actual={len(predictions)}")
        prediction = predictions[0]
        error = None
    except Exception as exc:  # failure isolation is part of the harness contract
        prediction = None
        error = {"type": type(exc).__name__, "message": str(exc)}
    after_rss = _rss_bytes()
    return {
        "ordinal": ordinal,
        "document_id": task["document_id"],
        "prediction": prediction,
        "error": error,
        "harness_wall_seconds": time.perf_counter() - wall_started,
        "harness_cpu_seconds": time.process_time() - cpu_started,
        "rss_before_bytes": before_rss,
        "rss_after_bytes": after_rss,
        "rss_high_water_bytes": max(
            value for value in (before_rss, after_rss) if value is not None
        ) if before_rss is not None or after_rss is not None else None,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", "utf-8")
    os.replace(temporary, path)


def validate_execution_fingerprint(path: Path, fingerprint: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text("utf-8"))
        if existing != fingerprint:
            raise ValueError("RESUME_FINGERPRINT_MISMATCH")
        if not resume:
            raise ValueError("OUTPUT_ALREADY_INITIALIZED_USE_RESUME")
    else:
        _atomic_json(path, fingerprint)


def ordered_unique_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identifiers = [row["document_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("DUPLICATE_RESULT_DOCUMENT_ID")
    return sorted(rows, key=lambda row: row["ordinal"])


def _semantic_projection(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = prediction.get("fields") or {}
    return {
        "document_id": prediction.get("document_id"),
        "route": prediction.get("route"),
        "schema": prediction.get("schema"),
        "fields": {
            name: {
                "value": field.get("value"),
                "disposition": (field.get("decision") or {}).get("disposition"),
            }
            for name, field in sorted(fields.items())
        },
        "claim_disposition": (prediction.get("claim_decision") or {}).get("disposition"),
    }


def compare_semantics(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["document_id"]: row for row in baseline}
    right = {row["document_id"]: row for row in candidate}
    differences = []
    for document_id in sorted(set(left) | set(right)):
        a, b = left.get(document_id), right.get(document_id)
        if a is None or b is None or _semantic_projection(a) != _semantic_projection(b):
            differences.append({"document_id": document_id, "classification": "SEMANTIC_REGRESSION"})
    return {
        "classification": "IDENTICAL" if not differences else "SEMANTIC_REGRESSION",
        "differences": differences,
    }


def aggregate(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    successful = [row for row in rows if row["prediction"] is not None]
    predictions = [row["prediction"] for row in successful]
    stage_values = {
        stage: [float((item.get("stage_seconds") or {}).get(stage, 0)) for item in predictions]
        for stage in STAGE_NAMES
    }
    total_wall = [float(item.get("wall_seconds") or 0) for item in predictions]
    total_cpu = [float(item.get("cpu_seconds") or 0) for item in predictions]
    rss = [row["rss_high_water_bytes"] for row in successful
           if row["rss_high_water_bytes"] is not None]
    counters: dict[str, int] = {}
    for prediction in predictions:
        for name, value in (prediction.get("counters") or {}).items():
            counters[name] = counters.get(name, 0) + int(value)
    stage_total = {name: sum(values) for name, values in stage_values.items()}
    measured_stage_total = sum(stage_total.values())
    return {
        "pages_requested": len(rows),
        "pages_succeeded": len(successful),
        "failure_count": len(rows) - len(successful),
        "throughput_pages_per_second": len(successful) / elapsed if elapsed else None,
        "elapsed_seconds": elapsed,
        "wall_seconds": _summary(total_wall),
        "cpu_seconds": _summary(total_cpu),
        "memory_high_water_bytes": _summary([float(value) for value in rss]),
        "stages": {
            name: {
                **_summary(values),
                "percent_of_measured_stage_latency": (
                    stage_total[name] / measured_stage_total if measured_stage_total else None
                ),
            }
            for name, values in stage_values.items()
        },
        "ocr_calls": counters,
        "routes": {
            route: sum(1 for item in predictions if (item.get("route") or "UNKNOWN") == route)
            for route in sorted({item.get("route") or "UNKNOWN" for item in predictions})
        },
        "field_count": _summary([float(len(item.get("fields") or {})) for item in predictions]),
        "failures": [row for row in rows if row["error"] is not None],
    }


def run(*, manifest: Path, output: Path, workers: int, thread_cap: int,
        runtime_manifest_id: str, corpus_fingerprint: str, code_sha: str,
        seed: int = 20260824, resume: bool = False) -> dict[str, Any]:
    selected = select(manifest, seed=seed, allocation=DEVELOPMENT_ALLOCATION)
    selection = [{"document_id": row["document_id"], "path": row["path"]} for row in selected]
    fingerprint = {
        "selection_sha256": _canonical_hash(selection),
        "runtime_manifest_id": runtime_manifest_id,
        "corpus_fingerprint": corpus_fingerprint,
        "code_sha": code_sha,
        "seed": seed,
        "thread_cap": thread_cap,
    }
    output.mkdir(parents=True, exist_ok=True)
    fingerprint_path = output / "execution_fingerprint.json"
    validate_execution_fingerprint(fingerprint_path, fingerprint, resume=resume)

    dataset = output / "dataset"
    metadata = dataset / "metadata" / "document_metadata.jsonl"
    if not metadata.exists():
        (dataset / "metadata").mkdir(parents=True, exist_ok=True)
        metadata.write_text("".join(json.dumps(row) + "\n" for row in selection), "utf-8")

    checkpoints = output / "checkpoints"
    tasks = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(selection):
        checkpoint = checkpoints / f"{ordinal:04d}-{row['document_id']}.json"
        if checkpoint.exists():
            value = json.loads(checkpoint.read_text("utf-8"))
            if value["document_id"] in seen:
                raise ValueError("DUPLICATE_CHECKPOINT_DOCUMENT_ID")
            seen.add(value["document_id"])
            rows.append(value)
            continue
        tasks.append({
            "ordinal": ordinal, "document_id": row["document_id"],
            "dataset": str(dataset), "worker_output": str(output / "worker-output" / str(ordinal)),
        })

    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_configure_worker, initargs=(thread_cap,)
    ) as pool:
        futures = {pool.submit(_run_page, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result["document_id"] in seen:
                raise ValueError("DUPLICATE_RESULT_DOCUMENT_ID")
            seen.add(result["document_id"])
            rows.append(result)
            checkpoint = checkpoints / f"{result['ordinal']:04d}-{result['document_id']}.json"
            _atomic_json(checkpoint, result)
    elapsed = time.perf_counter() - started
    rows = ordered_unique_results(rows)
    report = {
        "fingerprint": fingerprint,
        "workers": workers,
        "thread_cap": thread_cap,
        "selection": {"pages": len(selection), "seed": seed,
                      "allocation": DEVELOPMENT_ALLOCATION,
                      "document_ids": [row["document_id"] for row in selection],
                      "truth_used": False, "predictions_used": False},
        "metrics": aggregate(rows, elapsed),
    }
    _atomic_json(output / "performance_report.json", report)
    _atomic_json(output / "ordered_results.json", rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8), required=True)
    parser.add_argument("--thread-cap", type=int, default=0,
                        help="0 preserves library defaults; positive values cap internal threads")
    parser.add_argument("--runtime-manifest-id", required=True)
    parser.add_argument("--corpus-fingerprint", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
