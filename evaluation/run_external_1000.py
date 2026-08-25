"""End-to-end external 1,000-page qualification orchestration.

This command keeps PHI-bearing images, predictions, and truth in a private local
work directory. It safely extracts the supplied ZIP, builds a hashed manifest,
runs the canonical CDP production inference adapter over exactly 1,000 pages,
freezes predictions before truth, creates a prediction-blind annotation template,
and optionally scores when independent truth is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import zipfile

from evaluation.build_external_corpus_manifest import build_manifest
from evaluation.build_truth_annotation_template import build_template
from evaluation.external_qualification import run_inference, score_frozen, verify_freeze


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"UNSAFE_ZIP_MEMBER:{member.filename}")
        archive.extractall(destination)


def run(*, zip_path: str | Path, workdir: str | Path, runtime_manifest_id: str,
        corpus_id: str = "hackathon-1000-claims-v1", truth_jsonl: str | Path | None = None,
        fully_loaded_cost_usd: float | None = None) -> dict:
    source_zip = Path(zip_path).resolve()
    root = Path(workdir).resolve()
    if not source_zip.is_file():
        raise ValueError(f"ZIP_MISSING:{source_zip}")
    if root.exists() and any(root.iterdir()):
        raise ValueError("WORKDIR_MUST_BE_EMPTY")
    root.mkdir(parents=True, exist_ok=True)

    corpus_dir = root / "corpus"
    corpus_dir.mkdir()
    _safe_extract(source_zip, corpus_dir)

    manifest = root / "private_corpus_manifest.jsonl"
    manifest_summary = build_manifest(corpus_root=corpus_dir, output_jsonl=manifest)
    if manifest_summary["pages"] != 1000:
        raise ValueError(
            f"CORPUS_PAGE_COUNT_MISMATCH:expected=1000:actual={manifest_summary['pages']}"
        )

    predictions = root / "raw_predictions.jsonl"
    freeze = root / "prediction_freeze.json"
    annotation_template = root / "independent_truth_template.jsonl"

    python = shlex.quote(sys.executable)
    command_template = (
        f"{python} -m evaluation.external_single_page_inference "
        '--input "{input}" --document-id "{document_id}" --output "{output}"'
    )
    freeze_payload = run_inference(
        manifest_jsonl=manifest,
        predictions_jsonl=predictions,
        command_template=command_template,
        runtime_manifest_id=runtime_manifest_id,
        corpus_id=corpus_id,
        freeze_json=freeze,
        expected_pages=1000,
    )
    verify_freeze(predictions_jsonl=predictions, freeze_json=freeze)
    template_summary = build_template(
        manifest_jsonl=manifest,
        output_jsonl=annotation_template,
    )

    result = {
        "manifest": manifest_summary,
        "freeze": freeze_payload,
        "annotation_template": template_summary,
        "predictions": str(predictions),
        "workdir": str(root),
        "scored": False,
    }
    if truth_jsonl is not None:
        score_output = root / "qualification_score.json"
        result["score"] = score_frozen(
            predictions_jsonl=predictions,
            freeze_json=freeze,
            truth_jsonl=truth_jsonl,
            output_json=score_output,
            fully_loaded_cost_usd=fully_loaded_cost_usd,
        )
        result["score_output"] = str(score_output)
        result["scored"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--runtime-manifest-id", required=True)
    parser.add_argument("--corpus-id", default="hackathon-1000-claims-v1")
    parser.add_argument("--truth")
    parser.add_argument("--fully-loaded-cost-usd", type=float)
    args = parser.parse_args()
    result = run(
        zip_path=args.zip,
        workdir=args.workdir,
        runtime_manifest_id=args.runtime_manifest_id,
        corpus_id=args.corpus_id,
        truth_jsonl=args.truth,
        fully_loaded_cost_usd=args.fully_loaded_cost_usd,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
