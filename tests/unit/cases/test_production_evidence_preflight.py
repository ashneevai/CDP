from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.production_evidence import EvidenceRequirements, load_and_inspect


def _write_config(path: Path, expected: str | None) -> None:
    value = "null" if expected is None else expected
    path.write_text(
        "version: test-v1\n"
        "artifacts:\n"
        "  - id: truth\n"
        "    path: evidence/truth.json\n"
        "    gate: independent_holdout\n"
        f"    expected_sha256: {value}\n",
        "utf-8",
    )


def test_preflight_distinguishes_missing_unbound_and_verified_artifacts(tmp_path):
    config = tmp_path / "requirements.yaml"
    _write_config(config, None)
    missing = load_and_inspect(config, tmp_path)
    assert missing["status"] == "BLOCKED_EXTERNAL_EVIDENCE"
    assert missing["records"][0]["status"] == "MISSING"

    artifact = tmp_path / "evidence/truth.json"
    artifact.parent.mkdir()
    artifact.write_text("trusted truth\n", "utf-8")
    unbound = load_and_inspect(config, tmp_path)
    assert unbound["records"][0]["status"] == "UNBOUND_HASH"

    expected = sha256(artifact.read_bytes()).hexdigest()
    _write_config(config, expected)
    verified = load_and_inspect(config, tmp_path)
    assert verified["status"] == "VERIFIED"
    assert verified["blocking_artifacts"] == 0


def test_preflight_rejects_path_escape():
    with pytest.raises(ValidationError, match="ARTIFACT_PATH_MUST_BE_REPOSITORY_RELATIVE"):
        EvidenceRequirements.model_validate(
            {
                "version": "test-v1",
                "artifacts": [
                    {"id": "escape", "path": "../truth.json", "gate": "holdout"}
                ],
            }
        )
