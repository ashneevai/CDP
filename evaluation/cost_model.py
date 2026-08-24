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
