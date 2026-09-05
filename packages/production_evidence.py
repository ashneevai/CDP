from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from packages.domain.common import DomainModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RequiredArtifact(DomainModel):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    gate: str = Field(min_length=1)
    expected_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def path_is_repository_relative(self):
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ARTIFACT_PATH_MUST_BE_REPOSITORY_RELATIVE")
        return self


class EvidenceRequirements(DomainModel):
    version: str = Field(min_length=1)
    artifacts: list[RequiredArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def artifact_ids_and_paths_are_unique(self):
        ids = [artifact.id for artifact in self.artifacts]
        paths = [artifact.path for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_ARTIFACT_ID")
        if len(paths) != len(set(paths)):
            raise ValueError("DUPLICATE_ARTIFACT_PATH")
        return self


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_production_evidence(
    requirements: EvidenceRequirements, repository_root: Path
) -> dict[str, Any]:
    root = repository_root.resolve()
    records = []
    for artifact in requirements.artifacts:
        path = (root / artifact.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("ARTIFACT_PATH_ESCAPES_REPOSITORY") from exc
        observed_sha256 = _file_sha256(path) if path.is_file() else None
        if observed_sha256 is None:
            status = "MISSING"
        elif artifact.expected_sha256 is None:
            status = "UNBOUND_HASH"
        elif observed_sha256 != artifact.expected_sha256:
            status = "HASH_MISMATCH"
        else:
            status = "VERIFIED"
        records.append(
            {
                "id": artifact.id,
                "path": artifact.path,
                "gate": artifact.gate,
                "status": status,
                "observed_sha256": observed_sha256,
                "expected_sha256": artifact.expected_sha256,
            }
        )
    blockers = [record for record in records if record["status"] != "VERIFIED"]
    gates = sorted({str(record["gate"]) for record in blockers})
    return {
        "schema_version": "production-evidence-preflight-v1",
        "requirements_version": requirements.version,
        "status": "VERIFIED" if not blockers else "BLOCKED_EXTERNAL_EVIDENCE",
        "required_artifacts": len(records),
        "verified_artifacts": len(records) - len(blockers),
        "blocking_artifacts": len(blockers),
        "blocking_gates": gates,
        "records": records,
    }


def load_and_inspect(
    config_path: Path, repository_root: Path | None = None
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text("utf-8"))
    requirements = EvidenceRequirements.model_validate(config)
    return inspect_production_evidence(
        requirements,
        repository_root or config_path.resolve().parents[1],
    )
