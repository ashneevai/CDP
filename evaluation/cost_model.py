"""Fully loaded benchmark cost model.

This module prices measured compute time plus cloud/API spend. Rates are explicit
inputs so reports never silently invent infrastructure cost assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostRates:
    cpu_usd_per_hour: float = 0.0
    gpu_usd_per_hour: float = 0.0
    fixed_run_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class CostResult:
    cpu_cost_usd: float
    gpu_cost_usd: float
    cloud_api_cost_usd: float
    fixed_cost_usd: float
    total_cost_usd: float
    cost_usd_per_page: float


@dataclass(frozen=True, slots=True)
class OptionalCostRates:
    cpu_hourly_rate: float | None = None
    gpu_hourly_rate: float | None = None
    reviewer_hourly_rate: float | None = None
    fixed_run_cost: float | None = None


def resource_cost_report(
    *, pages: int, cpu_seconds: float, wall_seconds: float,
    peak_memory_bytes: int | None, ocr_calls: int,
    gpu_seconds: float = 0.0, reviewer_seconds: float = 0.0,
    cloud_api_cost: float | None = None, rates: OptionalCostRates,
) -> dict:
    if pages <= 0 or min(cpu_seconds, wall_seconds, gpu_seconds, reviewer_seconds, ocr_calls) < 0:
        raise ValueError("RESOURCE_INPUT_MUST_BE_NON_NEGATIVE")
    supplied = [value for value in (
        rates.cpu_hourly_rate, rates.gpu_hourly_rate,
        rates.reviewer_hourly_rate, rates.fixed_run_cost,
    ) if value is not None]
    if any(value < 0 for value in supplied) or (cloud_api_cost is not None and cloud_api_cost < 0):
        raise ValueError("COST_RATE_MUST_BE_NON_NEGATIVE")
    components = {
        "cpu_cost": (cpu_seconds * rates.cpu_hourly_rate / 3600
                     if rates.cpu_hourly_rate is not None else None),
        "gpu_cost": (gpu_seconds * rates.gpu_hourly_rate / 3600
                     if rates.gpu_hourly_rate is not None else None),
        "reviewer_cost": (reviewer_seconds * rates.reviewer_hourly_rate / 3600
                          if rates.reviewer_hourly_rate is not None else None),
        "cloud_api_cost": cloud_api_cost,
        "fixed_run_cost": rates.fixed_run_cost,
    }
    monetary_complete = all(value is not None for value in components.values())
    total = sum(components.values()) if monetary_complete else None
    return {
        "resources": {
            "pages": pages, "cpu_seconds_per_page": cpu_seconds / pages,
            "wall_seconds_per_page": wall_seconds / pages,
            "peak_memory_bytes": peak_memory_bytes,
            "ocr_calls_per_page": ocr_calls / pages,
        },
        "rates": {name: getattr(rates, name) for name in rates.__dataclass_fields__},
        "cost_components": components,
        "monetary_cost_status": "COMPLETE" if monetary_complete else "NOT_PROVIDED",
        "total_cost": total,
        "cost_per_page": total / pages if total is not None else None,
    }


def calculate_cost(
    *,
    pages: int,
    cpu_seconds: float,
    gpu_seconds: float = 0.0,
    cloud_api_cost_usd: float = 0.0,
    rates: CostRates,
) -> CostResult:
    if pages <= 0:
        raise ValueError("PAGES_MUST_BE_POSITIVE")
    if min(cpu_seconds, gpu_seconds, cloud_api_cost_usd) < 0:
        raise ValueError("COST_INPUT_MUST_BE_NON_NEGATIVE")
    if min(rates.cpu_usd_per_hour, rates.gpu_usd_per_hour, rates.fixed_run_usd) < 0:
        raise ValueError("COST_RATE_MUST_BE_NON_NEGATIVE")

    cpu_cost = cpu_seconds * rates.cpu_usd_per_hour / 3600.0
    gpu_cost = gpu_seconds * rates.gpu_usd_per_hour / 3600.0
    total = cpu_cost + gpu_cost + cloud_api_cost_usd + rates.fixed_run_usd
    return CostResult(
        cpu_cost_usd=cpu_cost,
        gpu_cost_usd=gpu_cost,
        cloud_api_cost_usd=cloud_api_cost_usd,
        fixed_cost_usd=rates.fixed_run_usd,
        total_cost_usd=total,
        cost_usd_per_page=total / pages,
    )
