from packages.claim_evidence.corroboration import independent_ocr_evidence


def _candidate(value: str, invocation: str, crop: str, region: str, dependencies: list[str]):
    return {"value": value, "provenance": {"invocation_id": invocation, "crop_sha256": crop,
            "localization_region_id": region, "source_candidate_id": invocation,
            "shared_dependency_ids": dependencies}}


def test_shared_crop_cannot_supply_independent_evidence():
    row = {"candidates": [_candidate("123", "a", "same", "r", ["crop:same"]),
                           _candidate("123", "b", "same", "r", ["crop:same"])]}
    assert independent_ocr_evidence("member_id", row) is None


def test_distinct_agreeing_observations_supply_e2():
    row = {"candidates": [_candidate("ABC123", "a", "one", "r1", ["crop:one"]),
                           _candidate("ABC123", "b", "two", "r2", ["crop:two"])]}
    assert independent_ocr_evidence("member_id", row).evidence_class == "E2"
