"""Verification helpers for immutable, checksum-pinned extraction releases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    payload = path.read_bytes()
    # Git stores these governed text configurations with LF. A Windows
    # checkout may materialize CRLF without changing repository content.
    # Verify the canonical repository bytes so the freeze is cross-platform.
    if path.suffix.casefold() in {".yaml", ".yml", ".json", ".toml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_release_manifest(path: Path) -> None:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise ValueError("release manifest is not frozen")
    for relative, expected in manifest.get("configuration_hashes", {}).items():
        actual = sha256_file(Path(relative))
        if actual != expected:
            raise ValueError(
                f"frozen configuration changed without a new release: {relative}"
            )
