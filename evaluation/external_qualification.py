"""Fail-closed external-corpus production qualification runner.

Raw source images, predictions, and truth remain outside Git. The runner enforces
an immutable sequence: validate corpus -> run production inference -> freeze
predictions -> independently reveal truth -> aggregate score.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any, Iterable

from evaluation.promotion_scorer import score


EXPECTED_PAGES = 1000


@dataclass(frozen=True, slots=True)
class CorpusItem:
    document_id: str
    path: Path
    sha256: str
    group: str | None = None
    package_id: str | None = None


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _load_manifest(path: str | Path, expected_pages: int = EXPECTED_PAGES) -> list[CorpusItem]:
    manifest_path = Path(path)
    rows: list[CorpusItem] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"INVALID_CORPUS_MANIFEST_JSONL:line={line_no}") from exc
            document_id = str(raw.get("document_id") or "").strip()
            source_path = Path(str(raw.get("path") or ""))
            expected_sha = str(raw.get("sha256") or "").lower()
            if not document_id or not str(source_path) or len(expected_sha) != 64:
                raise ValueError(f"CORPUS_IDENTITY_MISSING:line={line_no}")
            if document_id in seen:
                raise ValueError(f"DUPLICATE_CORPUS_DOCUMENT:{document_id}")
            if not source_path.is_file():
                raise ValueError(f"CORPUS_FILE_MISSING:{document_id}")
            actual_sha = _sha_file(source_path)
            if actual_sha != expected_sha:
                raise ValueError(f"CORPUS_FILE_HASH_MISMATCH:{document_id}")
            seen.add(document_id)
            rows.append(
                CorpusItem(
                    document_id=document_id,
                    path=source_path,
                    sha256=expected_sha,
                    group=(str(raw["group"]) if raw.get("group") is not None else None),
                    package_id=(
                        str(raw["package_id"])
                        if raw.get("package_id") is not None
                        else None
                    ),
                )
            )
    if len(rows) != expected_pages:
        raise ValueError(
            f"CORPUS_PAGE_COUNT_MISMATCH:expected={expected_pages}:actual={len(rows)}"
        )
    return rows


def _corpus_sha(items: Iterable[CorpusItem]) -> str:
    normalized = [
        {
            "document_id": item.document_id,
            "sha256": item.sha256,
            "group": item.group,
            "package_id": item.package_id,
        }
        for item in items
    ]
    return sha256(_canonical_json(normalized)).hexdigest()


def _read_prediction(path: Path, document_id: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"INVALID_INFERENCE_OUTPUT:{document_id}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"INVALID_INFERENCE_OUTPUT:{document_id}")
    output_id = str(raw.get("document_id") or document_id)
    if output_id != document_id:
        raise ValueError(f"INFERENCE_DOCUMENT_ID_MISMATCH:{document_id}")
    raw["document_id"] = document_id
    return raw


def run_inference(
    *,
    manifest_jsonl: str | Path,
    predictions_jsonl: str | Path,
    command_template: str,
    runtime_manifest_id: str,
    corpus_id: str,
    freeze_json: str | Path,
    expected_pages: int = EXPECTED_PAGES,
) -> dict[str, Any]:
    if not command_template.strip():
        raise ValueError("INFERENCE_COMMAND_REQUIRED")
    items = _load_manifest(manifest_jsonl, expected_pages)
    predictions_path = Path(predictions_jsonl)
    freeze_path = Path(freeze_json)
    if predictions_path.exists() or freeze_path.exists():
        raise ValueError("QUALIFICATION_OUTPUT_ALREADY_EXISTS")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cdp-qualification-") as tmp:
        tmp_path = Path(tmp)
        with predictions_path.open("x", encoding="utf-8") as output:
            for index, item in enumerate(items, 1):
                page_output = tmp_path / f"{index:04d}.json"
                substitutions = {
                    "input": str(item.path),
                    "document_id": item.document_id,
                    "output": str(page_output),
                }
                try:
                    rendered = command_template.format(**substitutions)
                except KeyError as exc:
                    raise ValueError(f"UNKNOWN_COMMAND_PLACEHOLDER:{exc.args[0]}") from exc
                started = time.perf_counter()
                completed = subprocess.run(
                    shlex.split(rendered),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                wall_seconds = time.perf_counter() - started
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"INFERENCE_FAILED:{item.document_id}:exit={completed.returncode}:"
                        f"stderr={completed.stderr[-1000:]}"
                    )
                if not page_output.is_file():
                    raise RuntimeError(f"INFERENCE_OUTPUT_MISSING:{item.document_id}")
                prediction = _read_prediction(page_output, item.document_id)
                prediction["wall_seconds"] = wall_seconds
                if item.group is not None:
                    prediction.setdefault("group", item.group)
                output.write(json.dumps(prediction, separators=(",", ":")) + "\n")
                output.flush()

    prediction_sha = _sha_file(predictions_path)
    corpus_sha = _corpus_sha(items)
    freeze = {
        "freeze_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus_id,
        "corpus_sha256": corpus_sha,
        "runtime_manifest_id": runtime_manifest_id,
        "prediction_sha256": prediction_sha,
        "prediction_records": len(items),
        "expected_pages": expected_pages,
        "truth_present": False,
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return freeze


def verify_freeze(
    *, predictions_jsonl: str | Path, freeze_json: str | Path
) -> dict[str, Any]:
    freeze = json.loads(Path(freeze_json).read_text(encoding="utf-8"))
    actual = _sha_file(Path(predictions_jsonl))
    if actual != freeze.get("prediction_sha256"):
        raise ValueError("PREDICTION_HASH_MISMATCH")
    if freeze.get("truth_present") is not False:
        raise ValueError("INVALID_FREEZE_TRUTH_STATE")
    return freeze


def score_frozen(
    *,
    predictions_jsonl: str | Path,
    freeze_json: str | Path,
    truth_jsonl: str | Path,
    output_json: str | Path,
    fully_loaded_cost_usd: float | None = None,
) -> dict[str, Any]:
    freeze = verify_freeze(predictions_jsonl=predictions_jsonl, freeze_json=freeze_json)
    truth_path = Path(truth_jsonl)
    if not truth_path.is_file():
        raise ValueError("INDEPENDENT_TRUTH_REQUIRED")
    return score(
        predictions_jsonl=predictions_jsonl,
        prediction_freeze_json=freeze_json,
        truth_jsonl=truth_jsonl,
        output_json=output_json,
        runtime_manifest_id=str(freeze["runtime_manifest_id"]),
        corpus_sha256=str(freeze["corpus_sha256"]),
        fully_loaded_cost_usd=fully_loaded_cost_usd,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run all pages and freeze predictions before truth.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--predictions", required=True)
    run.add_argument("--freeze", required=True)
    run.add_argument("--runtime-manifest-id", required=True)
    run.add_argument("--corpus-id", required=True)
    run.add_argument("--inference-command", required=True)
    run.add_argument("--expected-pages", type=int, default=EXPECTED_PAGES)

    verify = sub.add_parser("verify", help="Verify frozen predictions are unchanged.")
    verify.add_argument("--predictions", required=True)
    verify.add_argument("--freeze", required=True)

    scoring = sub.add_parser("score", help="Score only after independent truth is available.")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--freeze", required=True)
    scoring.add_argument("--truth", required=True)
    scoring.add_argument("--output", required=True)
    scoring.add_argument("--fully-loaded-cost-usd", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        freeze = run_inference(
            manifest_jsonl=args.manifest,
            predictions_jsonl=args.predictions,
            command_template=args.inference_command,
            runtime_manifest_id=args.runtime_manifest_id,
            corpus_id=args.corpus_id,
            freeze_json=args.freeze,
            expected_pages=args.expected_pages,
        )
        print(json.dumps(freeze, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_freeze(
            predictions_jsonl=args.predictions, freeze_json=args.freeze
        ), indent=2))
    else:
        report = score_frozen(
            predictions_jsonl=args.predictions,
            freeze_json=args.freeze,
            truth_jsonl=args.truth,
            output_json=args.output,
            fully_loaded_cost_usd=args.fully_loaded_cost_usd,
        )
        print(json.dumps(report["promotion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
