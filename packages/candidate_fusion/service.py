from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateObservation:
    value: str
    source: str
    lineage_id: str
    confidence: float
    spatial_confidence: float = 0.0
    structural_confidence: float = 0.0
    reference_support: float = 0.0
    cross_field_support: float = 0.0


@dataclass(frozen=True)
class FusedCandidate:
    value: str
    score: float
    sources: tuple[str, ...]
    lineages: tuple[str, ...]
    reason_codes: tuple[str, ...]


class CandidateFusionService:
    """Ranks candidate values; never accepts fields or bypasses EvidenceDecisionService."""

    def fuse(self, observations: list[CandidateObservation]) -> list[FusedCandidate]:
        grouped: dict[str, list[CandidateObservation]] = defaultdict(list)
        for observation in observations:
            normalized = self._normalize(observation.value)
            if normalized:
                grouped[normalized].append(observation)

        fused: list[FusedCandidate] = []
        for normalized, group in grouped.items():
            by_lineage: dict[str, CandidateObservation] = {}
            for item in group:
                current = by_lineage.get(item.lineage_id)
                if current is None or item.confidence > current.confidence:
                    by_lineage[item.lineage_id] = item
            independent = list(by_lineage.values())
            if not independent:
                continue
            base = sum(item.confidence for item in independent) / len(independent)
            spatial = max((item.spatial_confidence for item in independent), default=0.0)
            structural = max((item.structural_confidence for item in independent), default=0.0)
            reference = max((item.reference_support for item in independent), default=0.0)
            cross = max((item.cross_field_support for item in independent), default=0.0)
            diversity_bonus = min(0.08, 0.02 * max(0, len(independent) - 1))
            score = min(1.0, 0.55 * base + 0.12 * spatial + 0.12 * structural + 0.12 * reference + 0.09 * cross + diversity_bonus)
            reasons = ["CANDIDATE_FUSION_RANKED"]
            if len(independent) > 1:
                reasons.append("INDEPENDENT_LINEAGE_AGREEMENT")
            if reference > 0:
                reasons.append("REFERENCE_SUPPORT")
            if cross > 0:
                reasons.append("CROSS_FIELD_SUPPORT")
            fused.append(
                FusedCandidate(
                    value=group[0].value,
                    score=score,
                    sources=tuple(sorted({item.source for item in independent})),
                    lineages=tuple(sorted(by_lineage)),
                    reason_codes=tuple(reasons),
                )
            )
        return sorted(fused, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join((value or "").strip().casefold().split())
