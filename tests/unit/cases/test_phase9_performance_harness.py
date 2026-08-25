import json

import pytest

from evaluation.cost_model import OptionalCostRates, resource_cost_report
from evaluation.phase9_performance_harness import (
    _run_page,
    aggregate,
    compare_semantics,
    ordered_unique_results,
    validate_execution_fingerprint,
)
from evaluation import phase9_performance_harness as harness
from evaluation.run_production_holdout_v2 import _runtime_components


def _prediction(document_id="d1", value="A"):
    return {
        "document_id": document_id, "route": "CMS1500", "schema": "CMS1500",
        "fields": {"member_id": {"value": value, "decision": {"disposition": "HITL_REQUIRED"}}},
        "claim_decision": {"disposition": "REVIEW_REQUIRED"},
        "wall_seconds": 2.0, "cpu_seconds": 1.0,
        "stage_seconds": {"ocr": 1.5, "classification": .5},
        "counters": {"rapidocr_calls": 1},
    }


def test_ordered_results_are_deterministic_and_duplicates_fail():
    rows = [{"ordinal": 1, "document_id": "b"}, {"ordinal": 0, "document_id": "a"}]
    assert [row["document_id"] for row in ordered_unique_results(rows)] == ["a", "b"]
    with pytest.raises(ValueError, match="DUPLICATE_RESULT_DOCUMENT_ID"):
        ordered_unique_results(rows + [{"ordinal": 2, "document_id": "a"}])


def test_worker_failure_is_returned_not_raised(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("page failed")
    monkeypatch.setattr(harness, "infer", fail)
    result = _run_page({"ordinal": 0, "document_id": "d1", "dataset": str(tmp_path),
                        "worker_output": str(tmp_path / "out")})
    assert result["prediction"] is None
    assert result["error"] == {"type": "RuntimeError", "message": "page failed"}


def test_resume_requires_identical_fingerprint(tmp_path):
    path = tmp_path / "fingerprint.json"
    validate_execution_fingerprint(path, {"code_sha": "a"}, resume=False)
    validate_execution_fingerprint(path, {"code_sha": "a"}, resume=True)
    with pytest.raises(ValueError, match="RESUME_FINGERPRINT_MISMATCH"):
        validate_execution_fingerprint(path, {"code_sha": "b"}, resume=True)


def test_semantic_guard_and_resource_aggregation():
    baseline = [_prediction()]
    assert compare_semantics(baseline, [_prediction()])["classification"] == "IDENTICAL"
    assert compare_semantics(baseline, [_prediction(value="B")])["classification"] == "SEMANTIC_REGRESSION"
    rows = [{"ordinal": 0, "document_id": "d1", "prediction": baseline[0], "error": None,
             "rss_high_water_bytes": 100, "rss_before_bytes": 50, "rss_after_bytes": 100,
             "harness_wall_seconds": 2, "harness_cpu_seconds": 1}]
    report = aggregate(rows, 2)
    assert report["throughput_pages_per_second"] == .5
    assert report["ocr_calls"] == {"rapidocr_calls": 1}


def test_runtime_components_are_initialized_once_per_process():
    _runtime_components.cache_clear()
    assert _runtime_components() is _runtime_components()
    assert _runtime_components.cache_info().misses == 1


def test_unknown_cost_rates_remain_not_provided():
    report = resource_cost_report(
        pages=2, cpu_seconds=4, wall_seconds=3, peak_memory_bytes=100,
        ocr_calls=6, rates=OptionalCostRates(),
    )
    assert report["resources"]["cpu_seconds_per_page"] == 2
    assert report["monetary_cost_status"] == "NOT_PROVIDED"
    assert report["total_cost"] is None
