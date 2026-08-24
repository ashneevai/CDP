"""Utilities for validating frozen external reference corpora without storing PHI in git."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections import Counter
import zipfile


TIFF_LE_MAGIC = b"II*\x00"


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    sha256: str
    pages: int
    packages: int
    groups: dict[str, int]
    package_counts: dict[str, int]
    all_tiff: bool


def summarize_claim_zip(path: str | Path) -> CorpusSummary:
    path = Path(path)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    group_pages: Counter[str] = Counter()
    group_packages: dict[str, set[str]] = {}
    all_tiff = True

    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        for member in members:
            parts = Path(member.filename).parts
            group = parts[0] if len(parts) > 1 else "UNGROUPED"
            group_pages[group] += 1
            package_id = Path(member.filename).name.rsplit(".", 1)[0]
            group_packages.setdefault(group, set()).add(package_id)
            with archive.open(member) as source:
                if source.read(4) != TIFF_LE_MAGIC:
                    all_tiff = False

    package_counts = {group: len(ids) for group, ids in sorted(group_packages.items())}
    return CorpusSummary(
        sha256=digest.hexdigest(),
        pages=sum(group_pages.values()),
        packages=sum(package_counts.values()),
        groups=dict(sorted(group_pages.items())),
        package_counts=package_counts,
        all_tiff=all_tiff,
    )


def assert_frozen_corpus(path: str | Path, expected: dict) -> CorpusSummary:
    summary = summarize_claim_zip(path)
    mismatches: dict[str, object] = {}
    for key in ("sha256", "pages", "packages", "groups", "package_counts", "all_tiff"):
        actual = getattr(summary, key)
        if actual != expected[key]:
            mismatches[key] = {"expected": expected[key], "actual": actual}
    if mismatches:
        raise ValueError(f"REFERENCE_CORPUS_MISMATCH:{mismatches}")
    return summary
