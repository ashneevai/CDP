"""Field-specific candidate fusion for claims extraction.

Candidate fusion proposes a preferred candidate and supporting rationale.  It
never emits a final field disposition.  EvidenceDecisionService remains the
sole decision authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from packages.claims_specialist.cms import FieldPolicy, policy_for


@dataclass(frozen=True, slots=True)
class FusionCandidate:
    value: str
    source: str
    confidence: float
    lineage: str
    spatial_confidence: float | None = None
    handwriting: bool = False


@dataclass(frozen=True, slots=True)
class FusionResult:
    field_name: str
    value: str | None
    score: float
    valid: bool
    source_count: int
    independent_lineage_count: int
    agreeing_sources: tuple[str, ...]
    reason_codes: tuple[str, ...]
    needs_handwriting_fallback: bool


def _source_weight(policy: FieldPolicy, source: str) -> float:
    try:
        rank = policy.preferred_sources.index(source)
    except ValueError:
        return 0.72
    return max(0.75, 1.00 - rank * 0.05)


def fuse(field_name: str, candidates: Iterable[FusionCandidate]) -> FusionResult:
    policy = policy_for(field_name)
    rows = list(candidates)
    if policy is None:
        return FusionResult(
            field_name=field_name, value=None, score=0.0, valid=False,
            source_count=len(rows), independent_lineage_count=0,
            agreeing_sources=(), reason_codes=("NO_SPECIALIST_POLICY",),
            needs_handwriting_fallback=False,
        )
    if not rows:
        return FusionResult(
            field_name=field_name, value=None, score=0.0, valid=False,
            source_count=0, independent_lineage_count=0, agreeing_sources=(),
            reason_codes=("NO_CANDIDATES",), needs_handwriting_fallback=True,
        )

    grouped: dict[str, list[FusionCandidate]] = {}
    normalized_lookup: dict[str, str] = {}
    for row in rows:
        normalized = policy.normalizer(row.value)
        if not normalized:
            continue
        grouped.setdefault(normalized, []).append(row)
        normalized_lookup[normalized] = normalized

    if not grouped:
        return FusionResult(
            field_name=field_name, value=None, score=0.0, valid=False,
            source_count=len(rows), independent_lineage_count=0,
            agreeing_sources=(), reason_codes=("EMPTY_AFTER_NORMALIZATION",),
            needs_handwriting_fallback=True,
        )

    best_value: str | None = None
    best_score = -1.0
    best_rows: list[FusionCandidate] = []
    for normalized, agreeing in grouped.items():
        lineages = {row.lineage for row in agreeing if row.lineage}
        source_scores = []
        for row in agreeing:
            spatial = row.spatial_confidence if row.spatial_confidence is not None else 0.5
            score = max(0.0, min(1.0, row.confidence))
            score *= _source_weight(policy, row.source)
            score *= 0.85 + 0.15 * max(0.0, min(1.0, spatial))
            if row.handwriting:
                score *= 0.92
            source_scores.append(score)
        base = max(source_scores) if source_scores else 0.0
        independent_bonus = min(0.12, max(0, len(lineages) - 1) * 0.04)
        agreement_bonus = min(0.08, max(0, len(agreeing) - 1) * 0.02)
        validation_bonus = 0.08 if policy.validator(normalized) else -0.20
        combined = max(0.0, min(1.0, base + independent_bonus + agreement_bonus + validation_bonus))
        if combined > best_score:
            best_value = normalized_lookup[normalized]
            best_score = combined
            best_rows = agreeing

    assert best_value is not None
    valid = policy.validator(best_value)
    lineages = {row.lineage for row in best_rows if row.lineage}
    reasons = ["SPECIALIST_FUSION"]
    if valid:
        reasons.append("DETERMINISTIC_VALIDATION_PASS")
    else:
        reasons.append("DETERMINISTIC_VALIDATION_FAIL")
    if len(best_rows) > 1:
        reasons.append("CANDIDATE_AGREEMENT")
    if len(lineages) > 1:
        reasons.append("INDEPENDENT_LINEAGE_CORROBORATION")
    handwriting_present = any(row.handwriting for row in rows)
    needs_handwriting = handwriting_present and (not valid or best_score < 0.90)
    if needs_handwriting:
        reasons.append("HANDWRITING_FALLBACK_RECOMMENDED")

    return FusionResult(
        field_name=field_name,
        value=best_value,
        score=best_score,
        valid=valid,
        source_count=len(rows),
        independent_lineage_count=len(lineages),
        agreeing_sources=tuple(sorted({row.source for row in best_rows})),
        reason_codes=tuple(reasons),
        needs_handwriting_fallback=needs_handwriting,
    )
