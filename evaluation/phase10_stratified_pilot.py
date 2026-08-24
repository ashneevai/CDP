"""Create a deterministic stratified pilot from the frozen external corpus.

Selection is prediction-blind and truth-blind.  It uses only group/package
metadata present in the external corpus manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any


DEFAULT_ALLOCATION = {"Group A": 25, "Group B": 10, "Group C": 8, "Group D": 7}


def _read(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"INVALID_MANIFEST_RECORD:line={line_no}")
            rows.append(row)
    return rows


def select(
    manifest_jsonl: str | Path,
    *,
    seed: int = 20260824,
    allocation: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    allocation = allocation or DEFAULT_ALLOCATION
    rows = _read(manifest_jsonl)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group") or "UNKNOWN")].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group, requested in allocation.items():
        candidates = groups.get(group, [])
        if len(candidates) < requested:
            raise ValueError(
                f"PILOT_GROUP_TOO_SMALL:{group}:required={requested}:available={len(candidates)}"
            )
        # Prefer package diversity first, then fill remaining slots randomly.
        by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_package[str(row.get("package_id") or row.get("document_id"))].append(row)
        packages = list(by_package)
        rng.shuffle(packages)
        group_selected: list[dict[str, Any]] = []
        for package in packages:
            if len(group_selected) >= requested:
                break
            group_selected.append(rng.choice(by_package[package]))
        if len(group_selected) < requested:
            chosen = {str(row.get("document_id")) for row in group_selected}
            remaining = [row for row in candidates if str(row.get("document_id")) not in chosen]
            group_selected.extend(rng.sample(remaining, requested - len(group_selected)))
        selected.extend(group_selected)
    rng.shuffle(selected)
    return selected


def write_pilot(
    manifest_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    seed: int = 20260824,
) -> dict[str, Any]:
    selected = select(manifest_jsonl, seed=seed)
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "seed": seed,
        "pages": len(selected),
        "allocation": DEFAULT_ALLOCATION,
        "document_ids": [str(row.get("document_id")) for row in selected],
        "truth_used_for_selection": False,
        "predictions_used_for_selection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    report = write_pilot(args.manifest, args.output, seed=args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
