"""Selective PP-OCRv5 routing for Source B blocker crops."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

ELIGIBLE_REASONS = frozenset({
    "PRIMARY_EMPTY", "VALIDATION_FAILED", "LOW_OCR_CONFIDENCE",
    "OCR_CHARACTER_ERROR", "OCR_SEGMENTATION_ERROR", "LOCALIZATION_RECOVERED",
})


@dataclass(frozen=True)
class SourceBChallengeContext:
    source: str
    document_family: str
    current_claim_blocker: bool
    crop_safety_status: str
    primary_resolved: bool
    failure_reason: str
    request_scope: str = "FIELD_CROP"
    nonblocking_unlock_possible: bool = False


class ChallengerBudget:
    """Atomic per-evaluation budget; never grants candidate authority."""

    def __init__(self, blocker_fields: int, maximum_rate: float = .30) -> None:
        if blocker_fields < 0:
            raise ValueError("blocker_fields must be non-negative")
        if not 0.0 <= maximum_rate <= 0.30:
            raise ValueError("maximum_rate must be between zero and 0.30")
        # Small blocker bundles must receive one useful challenge. Replay and
        # batch callers share one budget sized to the full eligible population,
        # which preserves the strict aggregate ceiling.
        self.limit = 0 if blocker_fields == 0 else max(1, int(blocker_fields * maximum_rate))
        self.used = 0
        self._lock = Lock()

    def claim(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def route_to_ppocr_v5(context: SourceBChallengeContext, budget: ChallengerBudget) -> tuple[bool, str]:
    if context.request_scope != "FIELD_CROP":
        return False, "FULL_PAGE_REJECTED"
    if context.source != "SOURCE_B":
        return False, "SOURCE_NOT_ELIGIBLE"
    if context.document_family not in {"CMS1500", "UB04"}:
        return False, "FAMILY_NOT_ELIGIBLE"
    if not context.current_claim_blocker and not context.nonblocking_unlock_possible:
        return False, "NO_CLAIM_UNLOCK_IMPACT"
    if context.crop_safety_status != "CROP_SAFE":
        return False, "UNSAFE_CROP"
    if context.primary_resolved:
        return False, "VALID_PRIMARY_BYPASS"
    if context.failure_reason not in ELIGIBLE_REASONS:
        return False, "FAILURE_REASON_NOT_ELIGIBLE"
    if not budget.claim():
        return False, "INVOCATION_BUDGET_EXHAUSTED"
    return True, "SOURCE_B_BLOCKER_CHALLENGE"
