from __future__ import annotations

from enum import StrEnum
from pydantic import Field

from packages.domain.common import DomainModel


class ArtifactStatus(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


class ArtifactKind(StrEnum):
    ROUTER = "ROUTER"
    EXTRACTOR = "EXTRACTOR"
    OCR = "OCR"
    VLM = "VLM"
    CANDIDATE_FUSION = "CANDIDATE_FUSION"
    CONFIDENCE_CALIBRATOR = "CONFIDENCE_CALIBRATOR"
    EVIDENCE_POLICY = "EVIDENCE_POLICY"
    CLAIM_POLICY = "CLAIM_POLICY"


class RegisteredArtifact(DomainModel):
    artifact_id: str
    kind: ArtifactKind
    version: str
    status: ArtifactStatus
    training_dataset_id: str | None = None
    evaluation_dataset_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    runtime_compatible: bool = False
    notes: list[str] = Field(default_factory=list)


class ModelPolicyRegistry:
    def __init__(self, artifacts: list[RegisteredArtifact] | None = None) -> None:
        self._items = {item.artifact_id: item for item in (artifacts or [])}

    def register(self, artifact: RegisteredArtifact) -> None:
        if artifact.artifact_id in self._items:
            raise ValueError(f"ARTIFACT_ALREADY_REGISTERED:{artifact.artifact_id}")
        self._items[artifact.artifact_id] = artifact

    def production(self, kind: ArtifactKind) -> list[RegisteredArtifact]:
        return [item for item in self._items.values() if item.kind is kind and item.status is ArtifactStatus.PRODUCTION]
