from evaluation.claim_bundle_analysis import build_claim_blocker_analysis, failure_category


def field(name: str, *, exact: bool = True, layer: str = "PASS") -> dict:
    return {
        "document_id": "claim-1", "field_name": name, "criticality": "C2",
        "critical": True, "exact": exact, "disposition": "ESCALATE",
        "reason_codes": ["HARD_VALIDATION_PASSED"], "missing_evidence": ["E6"],
        "failure_layer": layer,
    }


def test_claim_matrix_reports_complete_bundle_unlock_opportunity():
    claim = {
        "claim_id": "claim-1", "family": "CMS1500", "source": "SOURCE_A",
        "blocking_unresolved_fields": ["member_id", "patient_name", "patient_dob"],
        "disposition": "FIELD_REVIEW_REQUIRED", "reason_codes": ["BLOCKING_FIELDS_UNRESOLVED"],
    }
    result = build_claim_blocker_analysis(
        [claim], [field("member_id"), field("patient_name"), field("patient_dob")],
        effort={"IDENTITY": 2},
    )
    row = result["claim_blocker_matrix"][0]
    identity = next(item for item in result["bundle_pareto"] if item["bundle"] == "IDENTITY")
    assert row["blocker_count"] == 3
    assert row["quality_segment"] == "UNKNOWN_NOT_CAPTURED"
    assert identity["complete_claim_opportunities"] == 1
    assert identity["safe_claims_per_effort"] == .5
    assert identity["production_authority"] is False


def test_mixed_bundle_is_not_reported_as_complete_unlock():
    claim = {
        "claim_id": "claim-1",
        "blocking_unresolved_fields": ["member_id", "provider_npi"],
    }
    result = build_claim_blocker_analysis(
        [claim], [field("member_id"), field("provider_npi")]
    )
    assert all(
        item["complete_claim_opportunities"] == 0 for item in result["bundle_pareto"]
    )


def test_incorrect_bundle_is_not_a_safe_unlock_opportunity():
    claim = {"claim_id": "claim-1", "blocking_unresolved_fields": ["member_id"]}
    result = build_claim_blocker_analysis([claim], [field("member_id", exact=False)])
    identity = next(item for item in result["bundle_pareto"] if item["bundle"] == "IDENTITY")
    assert identity["complete_claim_opportunities"] == 0
    assert result["incorrect_extraction_combinations"][0]["claims"] == 1


def test_failure_taxonomy_prefers_wrong_crop_over_missing_evidence():
    row = field("member_id", exact=False, layer="OCR")
    row["reason_codes"].append("WRONG_CROP_SUSPECTED")
    assert failure_category(row) == "WRONG_CROP"
