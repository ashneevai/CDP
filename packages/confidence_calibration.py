from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    predicted_mean: float
    actual_accuracy: float
    count: int


@dataclass(frozen=True)
class CalibrationReport:
    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    brier_score: float


def calibration_report(predictions: list[float], outcomes: list[bool], bins: int = 10) -> CalibrationReport:
    if len(predictions) != len(outcomes):
        raise ValueError("PREDICTION_OUTCOME_LENGTH_MISMATCH")
    if not predictions:
        return CalibrationReport(bins=(), expected_calibration_error=0.0, brier_score=0.0)
    if bins <= 0:
        raise ValueError("BINS_MUST_BE_POSITIVE")

    bucketed: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for prediction, outcome in zip(predictions, outcomes, strict=True):
        value = max(0.0, min(1.0, float(prediction)))
        index = min(bins - 1, int(value * bins))
        bucketed[index].append((value, bool(outcome)))

    result: list[CalibrationBin] = []
    total = len(predictions)
    ece = 0.0
    for index, bucket in enumerate(bucketed):
        if not bucket:
            continue
        predicted_mean = sum(value for value, _ in bucket) / len(bucket)
        actual_accuracy = sum(1.0 for _, outcome in bucket if outcome) / len(bucket)
        ece += (len(bucket) / total) * abs(predicted_mean - actual_accuracy)
        result.append(
            CalibrationBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                predicted_mean=predicted_mean,
                actual_accuracy=actual_accuracy,
                count=len(bucket),
            )
        )
    brier = sum((float(p) - (1.0 if o else 0.0)) ** 2 for p, o in zip(predictions, outcomes, strict=True)) / total
    return CalibrationReport(bins=tuple(result), expected_calibration_error=ece, brier_score=brier)
