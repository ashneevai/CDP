"""Build the private execution manifest for an external qualification corpus.

The generated manifest is intentionally NOT Git-safe because it contains local
source paths. Document/package IDs are synthetic and derived from hashes rather
than source filenames so aggregate reports need not expose source identifiers.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIFF_MAGIC = {b"II*\x00", b"MM\x00*"}


def _is_supported_image(path: Path) -> bool:
    if path.suffix.lower() in SUPPORTED_SUFFIXES:
        return True
    with path.open("rb") as handle:
        return handle.read(4) in TIFF_MAGIC


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_key(path: Path, corpus_root: Path) -> str:
    # Page bundles use suffixes such as .001/.002; strip only the final suffix.
    relative = path.relative_to(corpus_root)
    stem = relative.name.rsplit(".", 1)[0]
    parent = relative.parent.as_posix()
    return f"{parent}/{stem}"


def build_manifest(*, corpus_root: str | Path, output_jsonl: str | Path) -> dict:
    root = Path(corpus_root).resolve()
    if not root.is_dir():
        raise ValueError(f"CORPUS_ROOT_MISSING:{root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and _is_supported_image(path)
    )
    if not files:
        raise ValueError("CORPUS_HAS_NO_SUPPORTED_IMAGES")

    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files, 1):
        file_sha = _sha_file(path)
        package_sha = sha256(_package_key(path, root).encode("utf-8")).hexdigest()
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else "UNGROUPED"
        row = {
            "document_id": f"ext-{index:04d}-{file_sha[:12]}",
            "package_id": f"pkg-{package_sha[:16]}",
            "group": group,
            "path": str(path),
            "sha256": file_sha,
        }
        rows.append(row)

    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    package_count = len({row["package_id"] for row in rows})
    groups: dict[str, int] = {}
    for row in rows:
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    return {
        "pages": len(rows),
        "packages": package_count,
        "groups": dict(sorted(groups.items())),
        "manifest": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-pages", type=int, default=1000)
    args = parser.parse_args()
    summary = build_manifest(corpus_root=args.corpus_root, output_jsonl=args.output)
    if summary["pages"] != args.expected_pages:
        raise ValueError(
            f"CORPUS_PAGE_COUNT_MISMATCH:expected={args.expected_pages}:"
            f"actual={summary['pages']}"
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
