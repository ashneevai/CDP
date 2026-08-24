from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class EvidenceClass(StrEnum):
    REFERENCE = "REFERENCE"
    DETERMINISTIC = "DETERMINISTIC"
    STRUCTURAL = "STRUCTURAL"
    CROSS_FIELD = "CROSS_FIELD"
    DOCUMENT_INTERNAL = "DOCUMENT_INTERNAL"
    EXTERNAL_AUTHORIZED = "EXTERNAL_AUTHORIZED"
    HISTORICAL = "HISTORICAL"


class EvidenceAuthority(StrEnum):
    AUTHORITATIVE = "AUTHORITATIVE"
    STRONG = "STRONG"
    SUPPORTING = "SUPPORTING"
    ADVISORY = "ADVISORY"


class EvidenceOutcome(StrEnum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    field_name: str
    candidate_value: str | None
    document_family: str
    claim_context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    provider_id: str
    evidence_class: EvidenceClass
    authority: EvidenceAuthority
    outcome: EvidenceOutcome
    reason_code: str
    lineage_key: str
    value: str | None = None
    source_version: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.outcome in {EvidenceOutcome.SUPPORT, EvidenceOutcome.CONTRADICT}
