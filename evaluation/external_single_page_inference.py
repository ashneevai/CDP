"""Single-page adapter for external qualification using the canonical CDP runtime.

This module deliberately reuses ``evaluation.run_production_holdout_v2.infer``
instead of reimplementing OCR/routing/decision logic. Raw external source images
and predictions remain outside Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from evaluation.run_production_holdout_v2 import infer


def infer_single_page(*, input_path: str | Path, document_id: str) -> dict:
    source = Path(input_path)
    if not source.is_file():
        raise ValueError(f"INPUT_FILE_MISSING:{source}")
    if not document_id.strip():
        raise ValueError("DOCUMENT_ID_REQUIRED")

    with tempfile.TemporaryDirectory(prefix="cdp-external-page-") as tmp:
        root = Path(tmp)
        pages = root / "pages"
        metadata_dir = root / "metadata"
        output = root / "output"
        pages.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        # Preserve the original suffix so Pillow uses the source image decoder.
        local_name = f"page{source.suffix or '.tif'}"
        local_path = pages / local_name
        shutil.copy2(source, local_path)
        metadata = {
            "document_id": document_id,
            "path": f"pages/{local_name}",
        }
        (metadata_dir / "document_metadata.jsonl").write_text(
            json.dumps(metadata, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        predictions = infer(root, output, limit=1)
        if len(predictions) != 1:
            raise RuntimeError(
                f"UNEXPECTED_PREDICTION_COUNT:expected=1:actual={len(predictions)}"
            )
        prediction = predictions[0]
        if str(prediction.get("document_id")) != document_id:
            raise RuntimeError("PREDICTION_DOCUMENT_ID_MISMATCH")
        return prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction = infer_single_page(
        input_path=args.input,
        document_id=args.document_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
