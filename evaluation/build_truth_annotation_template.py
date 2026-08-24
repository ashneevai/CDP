"""Create a prediction-blind annotation template from the private corpus manifest.

The template intentionally contains no CDP predictions, routes, values, or
confidence scores. It is suitable for an independent annotator who can access
the source images through the private local path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_template(*, manifest_jsonl: str | Path, output_jsonl: str | Path) -> dict:
    manifest = Path(manifest_jsonl)
    output = Path(output_jsonl)
    if not manifest.is_file():
        raise ValueError("CORPUS_MANIFEST_REQUIRED")
    if output.exists():
        raise ValueError("TRUTH_TEMPLATE_ALREADY_EXISTS")

    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            document_id = str(raw.get("document_id") or "")
            package_id = str(raw.get("package_id") or "")
            source_path = str(raw.get("path") or "")
            if not document_id or not package_id or not source_path:
                raise ValueError(f"MANIFEST_IDENTITY_MISSING:line={line_no}")
            rows.append({
                "document_id": document_id,
                "package_id": package_id,
                "source_path": source_path,
                "document_type": None,
                "fields": {},
                "annotation_status": "UNLABELED",
                "annotator_id": None,
                "adjudication_status": "PENDING",
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"documents": len(rows), "output": str(output), "prediction_blind": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_template(
        manifest_jsonl=args.manifest,
        output_jsonl=args.output,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
