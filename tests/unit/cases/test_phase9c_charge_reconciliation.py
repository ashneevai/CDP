from decimal import Decimal

from packages.claim_evidence.charge_reconciliation import (
    ChargeReconciliationState,
    normalize_money,
    reconcile_total,
)


def test_exact_service_line_sum_is_safe_without_tolerance():
    result = reconcile_total("425.50", ["100.00", "250.00", "75.50"])

    assert result.state == ChargeReconciliationState.EXACT_MATCH
    assert result.calculated_sum == Decimal("425.50")
    assert result.difference == Decimal("0.00")
    assert result.safe is True


def test_formatting_only_normalization_is_recorded():
    result = reconcile_total("$1,250.00", ["1,000.00", "$250.00"])

    assert result.state == ChargeReconciliationState.DETERMINISTIC_NORMALIZATION_MATCH
    assert result.normalization_applied is True
    assert result.safe is True


def test_ambiguous_decimal_and_arithmetic_conflict_fail_closed():
    assert reconcile_total("42550", ["100.00", "250.00", "75.50"]).state == (
        ChargeReconciliationState.ARITHMETIC_CONFLICT
    )
    assert reconcile_total("425.OO", ["425.00"]).state == ChargeReconciliationState.MISSING_TOTAL
    assert reconcile_total("425.00", ["BAD"]).state == ChargeReconciliationState.AMBIGUOUS


def test_money_normalization_preserves_leading_zeroes_without_character_guessing():
    assert normalize_money(" 00012.30 ") == (Decimal("12.30"), True)
    assert normalize_money("12O.30")[0] is None
