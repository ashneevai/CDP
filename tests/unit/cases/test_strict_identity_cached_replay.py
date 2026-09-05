import json
from pathlib import Path

import pytest

from evaluation.real_archive_classification import Observation, PageRef
from evaluation.strict_identity_cached_replay import (
    DECISION_SCHEMA_VERSION,
    OCR_CONFIG_SHA256,
    PREPROCESSING_CONFIG_SHA256,
    _failure_result,
    _strict_completion,
    _summary,
    atomic_write_json,
    available_memory_mb,
    decision_checkpoint_key,
    decision_policy_manifest,
    finalize_existing_replay,
    observation_from_cache,
    ocr_cache_key,
    safe_worker_count,
    valid_cache_record,
    valid_page_checkpoint,
)
from workers.page_detection.text_extraction import TextLine


def _page(tmp_path: Path) -> PageRef:
    return PageRef(
        archive_id="archive",
        package_id="package",
        asset_id="asset",
        page_id="page",
        page_number=2,
        asset_page_count=3,
        asset_sequence=1,
        asset_path=tmp_path / "source.tif",
        asset_sha256="a" * 64,
        page_sha256="b" * 64,
    )


def _cache(page: PageRef, version: str = "1.4.4") -> dict:
    return {
        "source_asset_sha256": page.asset_sha256,
        "frame_index": 1,
        "rendered_page_sha256": page.page_sha256,
        "ocr_engine": "rapidocr",
        "ocr_engine_version": version,
        "preprocessing_version": "AUTO",
        "preprocessing_config_sha256": PREPROCESSING_CONFIG_SHA256,
        "ocr_config_version": "rapidocr-full-page-routing-v1",
        "ocr_config_sha256": OCR_CONFIG_SHA256,
        "cache_key": ocr_cache_key(page, version),
        "status": "OCR_EXECUTED",
        "tokens": [{"text": "UB-04", "bbox": [1, 2, 3, 4], "confidence": 0.99}],
    }


def _chain() -> dict:
    return {
        "classification_standard_candidate": False,
        "classification_document_subtype": "OTHER_CLAIM_FORM",
        "verification_status": None,
        "verified_identity_family": None,
        "fixed_extractor_authorized": False,
        "processing_route": "LAYOUT_STRUCTURED_EXTRACTOR",
        "localization_authorized": False,
        "actual_localization_invoked": False,
        "decision_reason_codes": ["STANDARD_IDENTITY_CLASSIFICATION_MISMATCH"],
    }


def _checkpoint(page: PageRef, policy: dict, status: str = "CACHE_HIT") -> dict:
    cache = _cache(page)
    cache["status"] = status
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_page_id": page.page_id,
        "source_page_sha256": page.page_sha256,
        "candidate_class": "OTHER_CLAIM_FORM",
        "form_identity": {
            "localization_allowed": False,
            "conflicting_anchors": {"CMS1500": ["NONCANONICAL_CLAIM"]},
        },
        "routing_result": {"router_nomination": "OTHER_CLAIM_FORM"},
        "ocr": {"latency_ms": 12.0},
        "ocr_provenance": cache,
        "production_chain": _chain(),
        "decision_provenance": {
            **policy,
            "decision_checkpoint_key": decision_checkpoint_key(page, policy),
        },
    }


def test_valid_cache_hit_and_cached_contract_matches_fresh(tmp_path):
    page = _page(tmp_path)
    record = _cache(page)
    assert valid_cache_record(record, page, "1.4.4")
    cached = observation_from_cache(record)
    fresh = Observation((TextLine("UB-04", 1, 2, 3, 4, 0.99),), 10, "rapidocr", "1.4.4")
    assert cached.lines == fresh.lines
    assert cached.engine == fresh.engine
    assert cached.engine_version == fresh.engine_version


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rendered_page_sha256", "c" * 64),
        ("ocr_config_sha256", "c" * 64),
        ("preprocessing_config_sha256", "d" * 64),
    ],
)
def test_cache_provenance_change_invalidates_record(tmp_path, field, value):
    page = _page(tmp_path)
    record = _cache(page)
    record[field] = value
    assert not valid_cache_record(record, page, "1.4.4")


def test_resume_requires_complete_policy_bound_checkpoint(tmp_path):
    page = _page(tmp_path)
    policy = decision_policy_manifest()
    checkpoint = _checkpoint(page, policy)
    assert valid_page_checkpoint(checkpoint, page, "1.4.4", policy)
    checkpoint["decision_provenance"]["decision_policy_sha256"] = "c" * 64
    assert not valid_page_checkpoint(checkpoint, page, "1.4.4", policy)


def test_policy_change_invalidates_decision_but_not_ocr_cache(tmp_path):
    page = _page(tmp_path)
    policy = decision_policy_manifest()
    checkpoint = _checkpoint(page, policy)
    changed = {**policy, "decision_policy_sha256": "d" * 64}
    assert valid_cache_record(_cache(page), page, "1.4.4")
    assert not valid_page_checkpoint(checkpoint, page, "1.4.4", changed)


def test_atomic_checkpoint_leaves_complete_json_and_ignores_orphan_temp(tmp_path):
    target = tmp_path / "pages" / "page.json"
    atomic_write_json(target, {"complete": True})
    target.with_name(".page.json.orphan.tmp").write_text("{", "utf-8")
    assert json.loads(target.read_text("utf-8")) == {"complete": True}


def test_failed_ocr_is_separate_phi_safe_and_fail_closed(tmp_path):
    page = _page(tmp_path)
    failed = _failure_result(page, "1.4.4", RuntimeError("sensitive message"), attempts=3)
    assert failed["ocr_provenance"]["status"] == "OCR_FAILED"
    assert "candidate_class" not in failed
    assert "production_chain" not in failed
    assert failed["failure"] == {
        "error_type": "RuntimeError",
        "message_persisted": False,
        "attempts": 3,
    }


def test_available_memory_uses_current_system_measurement(monkeypatch):
    memory = type("Memory", (), {"available": 3 * 1024 * 1024})()
    monkeypatch.setattr(
        "evaluation.strict_identity_cached_replay.psutil.virtual_memory", lambda: memory
    )
    assert available_memory_mb() == 3


def test_worker_pool_is_bounded_by_config_memory_and_cpu():
    assert safe_worker_count(99, free_memory_mb=50_000, logical_cpus=16) == 8
    assert safe_worker_count(8, free_memory_mb=1_500, logical_cpus=16) == 1
    assert safe_worker_count(8, free_memory_mb=50_000, logical_cpus=4) == 2


def test_summary_separates_success_failure_cache_and_decisions(tmp_path, monkeypatch):
    page = _page(tmp_path)
    policy = decision_policy_manifest()
    record = _checkpoint(page, policy)
    failure_page = PageRef(**{**page.__dict__, "page_id": "failed"})
    failure = _failure_result(failure_page, "1.4.4", TimeoutError(), policy)
    monkeypatch.setattr("evaluation.strict_identity_cached_replay.time.perf_counter", lambda: 2.0)
    summary = _summary(
        [page, failure_page],
        [record],
        1.0,
        1,
        failed_records=[failure],
        run_stats={"checkpoint_hits": 1, "retry_attempts": 2, "worker_restarts": 1},
    )
    assert summary["successful_pages"] == 1
    assert summary["failed_pages"] == 1
    assert summary["accounted_pages"] == 2
    assert summary["checkpoint_hits"] == 1
    assert summary["ocr_cache_hits"] == 1
    assert summary["retry_attempts"] == 2
    assert summary["actual_localization_invocations"] == 0
    assert summary["critical_false_authorizations"] == "NOT_EVALUABLE_WITHOUT_TRUSTED_LABELS"


def _write_finalizable_run(output: Path, page: PageRef, policy: dict) -> None:
    (output / "pages").mkdir(parents=True)
    (output / "failures").mkdir()
    record = _checkpoint(page, policy, "OCR_EXECUTED")
    atomic_write_json(output / "pages" / "page.json", record)
    manifest = {
        "rendered_pages": 1,
        "assets": 1,
        "package_count": 1,
        "input_manifest_sha256": "a" * 64,
        "decision_policy": policy,
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(
        output / "summary_partial.json",
        {
            "wall_clock_seconds": 10.0,
            "worker_count": 1,
            "checkpoint_hits": 0,
            "retry_attempts": 0,
            "worker_restarts": 0,
            "stale_decisions_rejected": 0,
            "stale_ocr_records_rejected": 0,
            "canaries": [
                {
                    "ub04_rejected": True,
                    "ub04_localization_authorizations": 0,
                }
            ],
        },
    )


def test_finalize_recomputes_strict_gates_without_ocr(tmp_path):
    output = tmp_path / "run"
    report = tmp_path / "report"
    page = _page(tmp_path)
    policy = decision_policy_manifest()
    _write_finalizable_run(output, page, policy)

    summary = finalize_existing_replay(output, report)

    assert summary["complete"]
    assert summary["successful_pages"] == 1
    assert summary["failed_pages"] == 0
    assert summary["production_promotion"] == "NOT_AUTHORIZED"
    assert json.loads((report / "final_report.json").read_text("utf-8")) == summary


def test_finalize_refuses_failure_even_when_all_pages_accounted(tmp_path):
    output = tmp_path / "run"
    report = tmp_path / "report"
    page = _page(tmp_path)
    policy = decision_policy_manifest()
    _write_finalizable_run(output, page, policy)
    (output / "pages" / "page.json").unlink()
    atomic_write_json(
        output / "failures" / "page.json",
        _failure_result(page, "1.4.4", TimeoutError(), policy),
    )

    with pytest.raises(RuntimeError, match="INCOMPLETE_OR_FAILED"):
        finalize_existing_replay(output, report)
    assert not report.exists()


def test_strict_completion_rejects_duplicate_or_mixed_policy():
    summary = {
        "successful_pages": 2,
        "total_pages_discovered": 2,
        "accounted_pages": 2,
        "unique_page_ids": 1,
        "failed_pages": 0,
        "ocr_failures": 0,
        "all_decision_policy_hashes_identical": True,
        "decision_policy_hashes": ["a"],
        "manifest": {"decision_policy": {"decision_policy_sha256": "a"}},
    }
    assert not _strict_completion(summary)
