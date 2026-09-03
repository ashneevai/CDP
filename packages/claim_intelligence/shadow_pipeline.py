from __future__ import annotations

from dataclasses import dataclass

from packages.candidate_fusion.service import CandidateFusionService, CandidateObservation, FusedCandidate
from packages.classification_v5.contracts import ClassificationSignal, PageClassificationV5
from packages.classification_v5.service import ClassificationV5Service
from packages.independent_evidence.contracts import EvidenceObservation, EvidenceRequest
from packages.independent_evidence.service import IndependentEvidenceService


@dataclass(frozen=True)
class ShadowFieldAssessment:
    field_name: str
    fused_candidates: tuple[FusedCandidate, ...]
    evidence: tuple[EvidenceObservation, ...]
    deterministic_evidence: frozenset[str]
    cross_field_evidence: frozenset[str]
    contradictions: frozenset[str]
    decision_authority: str = "NONE_SHADOW_ONLY"


@dataclass(frozen=True)
class ShadowPageAssessment:
    classification: PageClassificationV5
    fields: tuple[ShadowFieldAssessment, ...]
    production_mutation_allowed: bool = False


class ClaimIntelligenceShadowPipeline:
    """Exercises next-generation intelligence without mutating production decisions."""

    def __init__(
        self,
        classifier: ClassificationV5Service | None = None,
        fusion: CandidateFusionService | None = None,
        evidence: IndependentEvidenceService | None = None,
    ) -> None:
        self.classifier = classifier or ClassificationV5Service()
        self.fusion = fusion or CandidateFusionService()
        self.evidence = evidence or IndependentEvidenceService()

    def assess(
        self,
        *,
        page_id: str,
        classification_signals: list[ClassificationSignal],
        field_observations: dict[str, list[CandidateObservation]],
        claim_values: dict[str, object],
    ) -> ShadowPageAssessment:
        classification = self.classifier.classify(page_id, classification_signals)
        fields: list[ShadowFieldAssessment] = []
        for field_name, observations in sorted(field_observations.items()):
            fused = self.fusion.fuse(observations)
            candidate_value = fused[0].value if fused else None
            enrichment = self.evidence.collect(
                EvidenceRequest(
                    field_name=field_name,
                    candidate_value=candidate_value,
                    document_family=classification.family.value,
                    claim_context=claim_values,
                )
            )
            fields.append(
                ShadowFieldAssessment(
                    field_name=field_name,
                    fused_candidates=tuple(fused),
                    evidence=enrichment.observations,
                    deterministic_evidence=enrichment.deterministic_evidence,
                    cross_field_evidence=enrichment.cross_field_evidence,
                    contradictions=enrichment.contradictions,
                )
            )
        return ShadowPageAssessment(classification=classification, fields=tuple(fields))
