"""Freeze external-corpus predictions before independent truth is created.

This module deliberately does not score accuracy. It binds a prediction artifact
to the frozen corpus identity and RuntimeManifest so later truth can be compared
without contaminating the external benchmark through threshold/model tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

from evaluation.reference_corpus import assert_frozen_corpus
from packages.runtime_manifest import RuntimeManifest, manifest_from_mapping


@dataclass(frozen=True, slots=True)
class PredictionFreeze:
    corpus_id: str
    corpus_sha256: str
    runtime_manifest_id: str
    prediction_sha256: str
    prediction_records: int
    truth_present: bool
    scoring_allowed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"INVALID_PREDICTION_JSONL:line={line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"INVALID_PREDICTION_RECORD:line={line_no}")
            count += 1
    if count == 0:
        raise ValueError("EMPTY_PREDICTION_SET")
    return count


def freeze_predictions(
    *,
    corpus_zip: str | Path,
    corpus_manifest: str | Path,
    runtime_manifest: RuntimeManifest | Mapping[str, object],
    predictions_jsonl: str | Path,
    output: str | Path,
    truth_path: str | Path | None = None,
) -> PredictionFreeze:
    corpus_manifest_path = Path(corpus_manifest)
    expected = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    summary = assert_frozen_corpus(corpus_zip, expected)

    if not isinstance(runtime_manifest, RuntimeManifest):
        runtime_manifest = manifest_from_mapping(runtime_manifest)

    predictions = Path(predictions_jsonl)
    records = count_jsonl(predictions)

    truth_present = truth_path is not None and Path(truth_path).exists()
    # This function is the pre-truth freeze. It must never score.
    freeze = PredictionFreeze(
        corpus_id=str(expected["corpus_id"]),
        corpus_sha256=summary.sha256,
        runtime_manifest_id=runtime_manifest.manifest_id,
        prediction_sha256=_sha_file(predictions),
        prediction_records=records,
        truth_present=truth_present,
        scoring_allowed=False,
    )

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"PREDICTION_FREEZE_ALREADY_EXISTS:{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze.as_dict(), indent=2) + "\n", encoding="utf-8")
    return freeze
