from evaluation.phase8_24_evidence_extension import run
from packages.claim_evidence.pair_assessment import assess_evidence_pair


def _candidate(value, section, crop, region):
    return {"value": value, "semantic_section": section, "provenance": {
        "source_candidate_id": region, "invocation_id": f"i-{region}",
        "crop_sha256": crop, "localization_region_id": region,
        "shared_dependency_ids": [f"crop:{crop}"],
    }}


def test_pair_requires_distinct_crop_region_and_semantic_section():
    left = _candidate("1234567893", "billing_provider", "same", "same")
    right = _candidate("1234567893", "service_facility", "same", "same")
    result = assess_evidence_pair("provider_npi", left, right)
    assert not result.genuinely_independent
    assert "SAME_CROP_SHA256" in result.rejection_reasons


def test_material_conflict_is_annotated_and_cannot_agree():
    left = _candidate("1234567893", "billing_provider", "one", "r1")
    right = _candidate("1234567890", "service_facility", "two", "r2")
    result = assess_evidence_pair("provider_npi", left, right)
    assert result.genuinely_independent and result.semantically_compatible
    assert result.conflicting and not result.agreeing


def test_extension_has_separate_verdict_and_preserves_frozen_phase(tmp_path):
    report = run(tmp_path)
    assert report["frozen_phase8_23_verdict"] == "NEEDS_MORE_DATA"
    assert report["extension_verdict"] == "PASS"
    assert report["metrics"]["critical_false_accepts"] == 0
    assert report["metrics"]["accepted_precision"] == 1.0
    assert {path.name for path in tmp_path.iterdir()} == {
        "pair_annotations.jsonl", "e2_coverage_metrics.json",
        "frozen_phase8_23_integrity.json", "comparative_report.json",
    }
