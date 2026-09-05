"""Privacy-preserving overlap inputs for shadow qualification."""

from __future__ import annotations

import json
from pathlib import Path

from packages.shadow_evaluation.capture import identity_fingerprint


def fingerprinted_source_groups(
    dataset: Path,
    splits: set[str],
    *,
    identity_key: bytes,
) -> set[str]:
    """Load raw learning identities as ledger-compatible HMAC fingerprints."""
    if not identity_key:
        raise ValueError("a non-empty shadow identity key is required for overlap checks")
    if not dataset.is_dir():
        raise ValueError(f"correction dataset directory not found: {dataset}")
    groups: set[str] = set()
    for split in sorted(splits):
        target = dataset / f"{split}.jsonl"
        if not target.is_file():
            continue
        for line_number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            source_group = str(row.get("source_group_id") or "").strip()
            if not source_group:
                raise ValueError(
                    f"missing source_group_id in {target.name} line {line_number}"
                )
            groups.add(identity_fingerprint(source_group, identity_key))
    return groups
