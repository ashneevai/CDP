"""Run the canonical truth-blind CDP runtime against an external frozen ZIP corpus.

Raw predictions may contain PHI and are written only to the caller-provided local
output directory. Git-safe output is limited to aggregate counts, latency and
runtime/corpus fingerprints. Accuracy is intentionally not computed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import zipfile

from evaluation.prediction_freeze import freeze_predictions
from evaluation.reference_corpus import assert_frozen_corpus
from evaluation.run_production_holdout_v2 import infer
from packages.runtime_manifest import manifest_from_mapping


def _hash_id(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:20]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def build_truth_blind_dataset(corpus_zip: Path, dataset_root: Path) -> dict[str, dict[str, str]]:
    """Extract pages and build minimal metadata without exposing source filenames downstream."""
    metadata_dir = dataset_root / "metadata"
    pages_dir = dataset_root / "pages"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    lineage: dict[str, dict[str, str]] = {}
    metadata_rows: list[dict[str, str]] = []
    with zipfile.ZipFile(corpus_zip) as archive:
        members = sorted(
            (member for member in archive.infolist() if not member.is_dir()),
            key=lambda member: member.filename,
        )
        for ordinal, member in enumerate(members, start=1):
            parts = Path(member.filename).parts
            group = parts[0] if len(parts) > 1 else "UNGROUPED"
            basename = Path(member.filename).name
            package_source = basename.rsplit(".", 1)[0]
            page_id = _hash_id(f"page:{member.filename}")
            package_id = _hash_id(f"package:{group}:{package_source}")
            local_rel = Path("pages") / f"{ordinal:04d}.tif"
            target = dataset_root / local_rel
            with archive.open(member) as source, target.open("wb") as sink:
                sink.write(source.read())
            metadata_rows.append({"document_id": page_id, "path": local_rel.as_posix()})
            lineage[page_id] = {"group": group, "package_id": package_id}

    metadata_path = metadata_dir / "document_metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    return lineage


def write_raw_prediction_jsonl(predictions: list[dict], lineage: dict[str, dict[str, str]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"RAW_PREDICTIONS_ALREADY_EXIST:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            page_id = prediction["document_id"]
            record = dict(prediction)
            record["group"] = lineage[page_id]["group"]
            record["package_id"] = lineage[page_id]["package_id"]
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def aggregate_predictions(predictions: list[dict], lineage: dict[str, dict[str, str]]) -> dict:
    routes = Counter()
    schemas = Counter()
    field_dispositions = Counter()
    claim_dispositions = Counter()
    group_routes: dict[str, Counter] = defaultdict(Counter)
    stage_seconds: dict[str, list[float]] = defaultdict(list)
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    field_counts: list[int] = []
    pages_with_claim_decision = 0
    packages_with_review: set[str] = set()
    packages_seen: set[str] = set()

    accepted = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED", "HUMAN_CONFIRMED"}
    accepted_fields = 0
    decided_fields = 0

    for prediction in predictions:
        page_id = prediction["document_id"]
        info = lineage[page_id]
        group = info["group"]
        package_id = info["package_id"]
        packages_seen.add(package_id)

        route = prediction.get("route") or "UNKNOWN"
        schema = prediction.get("schema") or "UNKNOWN"
        routes[route] += 1
        schemas[schema] += 1
        group_routes[group][route] += 1

        fields = prediction.get("fields") or {}
        field_counts.append(len(fields))
        page_review = False
        for field in fields.values():
            decision = field.get("decision") or {}
            disposition = decision.get("disposition")
            if disposition:
                field_dispositions[disposition] += 1
                decided_fields += 1
                if disposition in accepted:
                    accepted_fields += 1
                else:
                    page_review = True
            else:
                page_review = True

        claim = prediction.get("claim_decision")
        if claim:
            pages_with_claim_decision += 1
            disposition = claim.get("disposition", "UNKNOWN")
            claim_dispositions[disposition] += 1
            if "REVIEW" in disposition or "BLOCK" in disposition:
                page_review = True

        if page_review:
            packages_with_review.add(package_id)

        wall = prediction.get("wall_seconds")
        cpu = prediction.get("cpu_seconds")
        if isinstance(wall, (int, float)):
            wall_seconds.append(float(wall))
        if isinstance(cpu, (int, float)):
            cpu_seconds.append(float(cpu))
        for name, value in (prediction.get("stage_seconds") or {}).items():
            if isinstance(value, (int, float)):
                stage_seconds[name].append(float(value))

    safe_coverage = accepted_fields / decided_fields if decided_fields else None
    package_review_signal_rate = len(packages_with_review) / len(packages_seen) if packages_seen else None

    return {
        "qualification": {
            "truth_blind": True,
            "accuracy_scored": False,
            "false_accepts_scored": False,
            "production_authority": "SHADOW_READINESS_ONLY",
            "raw_predictions_git_safe": False,
            "aggregate_report_git_safe": True,
        },
        "pages": len(predictions),
        "packages": len(packages_seen),
        "routing": {
            "distribution": dict(sorted(routes.items())),
            "schema_distribution": dict(sorted(schemas.items())),
            "per_group": {group: dict(sorted(values.items())) for group, values in sorted(group_routes.items())},
        },
        "extraction": {
            "fields_emitted": sum(field_counts),
            "mean_fields_per_page": statistics.fmean(field_counts) if field_counts else 0,
            "pages_with_zero_fields": sum(1 for value in field_counts if value == 0),
        },
        "decision": {
            "field_dispositions": dict(sorted(field_dispositions.items())),
            "decided_fields": decided_fields,
            "accepted_fields": accepted_fields,
            "safe_coverage_signal": safe_coverage,
            "note": "Safe coverage is a decision-disposition signal only; correctness requires independent truth.",
        },
        "claim": {
            "pages_with_claim_decision": pages_with_claim_decision,
            "dispositions": dict(sorted(claim_dispositions.items())),
            "packages_with_review_signal": len(packages_with_review),
            "package_review_signal_rate": package_review_signal_rate,
            "note": "This is not claim STP accuracy; package truth/assembly labels are not available.",
        },
        "latency_seconds": {
            "mean": statistics.fmean(wall_seconds) if wall_seconds else None,
            "p50": _percentile(wall_seconds, 0.50),
            "p95": _percentile(wall_seconds, 0.95),
            "p99": _percentile(wall_seconds, 0.99),
            "cpu_mean": statistics.fmean(cpu_seconds) if cpu_seconds else None,
            "stage_mean": {
                name: statistics.fmean(values) for name, values in sorted(stage_seconds.items()) if values
            },
        },
        "cost": {"cloud_cost_usd": sum(float(p.get("cloud_cost_usd") or 0) for p in predictions)},
    }


def run(corpus_zip: Path, corpus_manifest: Path, runtime_manifest_path: Path, output: Path) -> dict:
    expected = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    corpus = assert_frozen_corpus(corpus_zip, expected)
    runtime_mapping = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    runtime_manifest = manifest_from_mapping(runtime_mapping)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_DIRECTORY_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="cdp_external_corpus_") as temp:
        dataset = Path(temp) / "dataset"
        runtime_output = Path(temp) / "runtime"
        lineage = build_truth_blind_dataset(corpus_zip, dataset)
        predictions = infer(dataset, runtime_output)

    raw_predictions = output / "raw_predictions.jsonl"
    write_raw_prediction_jsonl(predictions, lineage, raw_predictions)
    freeze = freeze_predictions(
        corpus_zip=corpus_zip,
        corpus_manifest=corpus_manifest,
        runtime_manifest=runtime_manifest,
        predictions_jsonl=raw_predictions,
        output=output / "prediction_freeze.json",
    )
    aggregate = aggregate_predictions(predictions, lineage)
    aggregate.update({
        "corpus_id": expected["corpus_id"],
        "corpus_sha256": corpus.sha256,
        "runtime_manifest_id": runtime_manifest.manifest_id,
        "prediction_sha256": freeze.prediction_sha256,
    })
    (output / "aggregate_report.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-zip", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True,
                        help="Explicit production-equivalent RuntimeManifest JSON; missing identity fails closed.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Local/private output directory. raw_predictions.jsonl may contain PHI.")
    args = parser.parse_args()
    result = run(args.corpus_zip, args.corpus_manifest, args.runtime_manifest, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
