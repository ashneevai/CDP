from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import EvidenceObservation, EvidenceOutcome
from .service import EvidenceEnrichment


@dataclass(frozen=True, slots=True)
class DecisionEvidenceInputs:
    deterministic_evidence: frozenset[str]
    cross_field_evidence: frozenset[str]
    contradiction_reason_codes: frozenset[str]
    audit_observations: tuple[EvidenceObservation, ...]


def merge_with_existing_evidence(
    enrichment: EvidenceEnrichment,
    *,
    deterministic_evidence: Iterable[str] = (),
    cross_field_evidence: Iterable[str] = (),
) -> DecisionEvidenceInputs:
    """Prepare enriched evidence for the canonical DecisionContext construction site.

    This helper intentionally does not construct a FieldDecision and does not alter
    hard_validation_passed. Contradictions are surfaced separately so the canonical
    validation worker can fail closed according to field-specific policy.
    """
    return DecisionEvidenceInputs(
        deterministic_evidence=frozenset(deterministic_evidence) | enrichment.deterministic_evidence,
        cross_field_evidence=frozenset(cross_field_evidence) | enrichment.cross_field_evidence,
        contradiction_reason_codes=enrichment.contradictions,
        audit_observations=enrichment.observations,
    )
