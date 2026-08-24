"""End-to-end 50-page specialist pilot orchestration.

Selects a deterministic stratified sample from the external manifest, runs the
canonical external qualification path for those pages, freezes predictions, and
scores canonical plus specialist-shadow output once independent truth exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluation.specialist_shadow_score import score_shadow, evaluate_activation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--runtime-manifest-id", required=True)
    parser.add_argument("--truth")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pilot_manifest = workdir / "pilot_50_manifest.jsonl"
    predictions = workdir / "pilot_50_predictions.jsonl"
    freeze = workdir / "pilot_50_freeze.json"
    safety_report = workdir / "pilot_50_safety_report.json"
    shadow_report_path = workdir / "pilot_50_shadow_report.json"
    activation_path = workdir / "pilot_50_activation.json"

    subprocess.run([
        sys.executable,
        "-m",
        "evaluation.select_stratified_pilot",
        "--manifest",
        args.manifest,
        "--output",
        str(pilot_manifest),
        "--seed",
        str(args.seed),
    ], check=True)

    inference_command = (
        f"{sys.executable} -m evaluation.external_single_page_inference "
        "--input {input} --document-id {document_id} --output {output}"
    )
    subprocess.run([
        sys.executable,
        "-m",
        "evaluation.external_qualification",
        "run",
        "--manifest",
        str(pilot_manifest),
        "--predictions",
        str(predictions),
        "--freeze",
        str(freeze),
        "--runtime-manifest-id",
        args.runtime_manifest_id,
        "--corpus-id",
        "hackathon-1000-claims-pilot-50-v1",
        "--inference-command",
        inference_command,
        "--expected-pages",
        "50",
    ], check=True)

    status = {
        "pilot_manifest": str(pilot_manifest),
        "predictions": str(predictions),
        "freeze": str(freeze),
        "truth_available": bool(args.truth),
    }
    if args.truth:
        subprocess.run([
            sys.executable,
            "-m",
            "evaluation.external_qualification",
            "score",
            "--predictions",
            str(predictions),
            "--freeze",
            str(freeze),
            "--truth",
            args.truth,
            "--output",
            str(safety_report),
        ], check=True)
        safety = json.loads(safety_report.read_text(encoding="utf-8"))
        shadow = score_shadow(predictions_jsonl=predictions, truth_jsonl=args.truth)
        shadow_report_path.write_text(json.dumps(shadow, indent=2) + "\n", encoding="utf-8")
        activation = evaluate_activation(shadow_report=shadow, safety_report=safety)
        activation_path.write_text(json.dumps(activation, indent=2) + "\n", encoding="utf-8")
        status.update({
            "safety_report": str(safety_report),
            "shadow_report": str(shadow_report_path),
            "activation_report": str(activation_path),
            "activation": activation,
        })

    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
