from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import EvidenceClass, EvidenceObservation, EvidenceOutcome, EvidenceRequest
from .providers import DEFAULT_PROVIDERS, IndependentEvidenceProvider


@dataclass(frozen=True, slots=True)
class EvidenceEnrichment:
    observations: tuple[EvidenceObservation, ...]
    deterministic_evidence: frozenset[str]
    cross_field_evidence: frozenset[str]
    contradictions: frozenset[str]


class IndependentEvidenceService:
    """Collect evidence without making field acceptance/rejection decisions.

    Evidence is de-duplicated by lineage_key so multiple providers derived from the
    same underlying fact cannot accidentally masquerade as independent support.
    """

    def __init__(self, providers: Iterable[IndependentEvidenceProvider] = DEFAULT_PROVIDERS) -> None:
        self.providers = tuple(providers)

    def collect(self, request: EvidenceRequest) -> EvidenceEnrichment:
        by_lineage: dict[str, EvidenceObservation] = {}
        for provider in self.providers:
            if not provider.supports(request):
                continue
            observation = provider.collect(request)
            existing = by_lineage.get(observation.lineage_key)
            if existing is None or self._priority(observation) > self._priority(existing):
                by_lineage[observation.lineage_key] = observation

        observations = tuple(sorted(by_lineage.values(), key=lambda item: item.lineage_key))
        deterministic = frozenset(
            item.reason_code
            for item in observations
            if item.outcome is EvidenceOutcome.SUPPORT
            and item.evidence_class is EvidenceClass.DETERMINISTIC
        )
        cross_field = frozenset(
            item.reason_code
            for item in observations
            if item.outcome is EvidenceOutcome.SUPPORT
            and item.evidence_class is EvidenceClass.CROSS_FIELD
        )
        contradictions = frozenset(
            item.reason_code for item in observations if item.outcome is EvidenceOutcome.CONTRADICT
        )
        return EvidenceEnrichment(
            observations=observations,
            deterministic_evidence=deterministic,
            cross_field_evidence=cross_field,
            contradictions=contradictions,
        )

    @staticmethod
    def _priority(observation: EvidenceObservation) -> tuple[int, int]:
        outcome_rank = {
            EvidenceOutcome.CONTRADICT: 4,
            EvidenceOutcome.SUPPORT: 3,
            EvidenceOutcome.INCONCLUSIVE: 2,
            EvidenceOutcome.UNAVAILABLE: 1,
        }[observation.outcome]
        authority_rank = {
            "AUTHORITATIVE": 4,
            "STRONG": 3,
            "SUPPORTING": 2,
            "ADVISORY": 1,
        }[observation.authority.value]
        return outcome_rank, authority_rank
