"""Strict, truth-blind claim-total reconciliation for claim closure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class ChargeReconciliationState(StrEnum):
    EXACT_MATCH = "EXACT_MATCH"
    DETERMINISTIC_NORMALIZATION_MATCH = "DETERMINISTIC_NORMALIZATION_MATCH"
    ARITHMETIC_CONFLICT = "ARITHMETIC_CONFLICT"
    MISSING_SERVICE_LINES = "MISSING_SERVICE_LINES"
    MISSING_TOTAL = "MISSING_TOTAL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ChargeReconciliation:
    reported_total: Decimal | None
    service_line_charges: tuple[Decimal, ...]
    calculated_sum: Decimal | None
    difference: Decimal | None
    state: ChargeReconciliationState
    normalization_applied: bool

    @property
    def safe(self) -> bool:
        return self.state in {
            ChargeReconciliationState.EXACT_MATCH,
            ChargeReconciliationState.DETERMINISTIC_NORMALIZATION_MATCH,
        }


def normalize_money(value: object) -> tuple[Decimal | None, bool]:
    """Normalize formatting only; never substitute or infer OCR characters."""
    original = str(value or "")
    raw = original.strip()
    if not raw:
        return None, False
    cleaned = re.sub(r"[$,\s]", "", raw)
    if not re.fullmatch(r"(?:\d+|\d+\.\d{1,2})", cleaned):
        return None, cleaned != original
    try:
        return Decimal(cleaned).quantize(Decimal("0.01")), cleaned != original
    except InvalidOperation:
        return None, cleaned != original


def reconcile_total(
    reported_total: object,
    service_line_charges: list[object] | tuple[object, ...],
) -> ChargeReconciliation:
    total, total_changed = normalize_money(reported_total)
    if total is None:
        return ChargeReconciliation(
            None, (), None, None, ChargeReconciliationState.MISSING_TOTAL, total_changed
        )
    if not service_line_charges:
        return ChargeReconciliation(
            total, (), None, None, ChargeReconciliationState.MISSING_SERVICE_LINES, total_changed
        )
    parsed: list[Decimal] = []
    changed = total_changed
    for value in service_line_charges:
        amount, normalized = normalize_money(value)
        changed = changed or normalized
        if amount is None:
            return ChargeReconciliation(
                total, tuple(parsed), None, None, ChargeReconciliationState.AMBIGUOUS, changed
            )
        parsed.append(amount)
    calculated = sum(parsed, Decimal("0.00"))
    difference = total - calculated
    if difference:
        state = ChargeReconciliationState.ARITHMETIC_CONFLICT
    elif changed:
        state = ChargeReconciliationState.DETERMINISTIC_NORMALIZATION_MATCH
    else:
        state = ChargeReconciliationState.EXACT_MATCH
    return ChargeReconciliation(total, tuple(parsed), calculated, difference, state, changed)
