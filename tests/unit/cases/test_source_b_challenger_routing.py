from packages.ocr.source_b_routing import (
    ChallengerBudget,
    SourceBChallengeContext,
    route_to_ppocr_v5,
)


def context(**changes):
    values = {
        "source": "SOURCE_B", "document_family": "CMS1500",
        "current_claim_blocker": True, "crop_safety_status": "CROP_SAFE",
        "primary_resolved": False, "failure_reason": "PRIMARY_EMPTY",
    }
    values.update(changes)
    return SourceBChallengeContext(**values)


def test_only_safe_unresolved_source_b_blocker_crop_is_routed():
    assert route_to_ppocr_v5(context(), ChallengerBudget(10))[0]
    assert route_to_ppocr_v5(context(source="SOURCE_A"), ChallengerBudget(10))[1] == "SOURCE_NOT_ELIGIBLE"
    assert route_to_ppocr_v5(context(crop_safety_status="EMPTY_CROP"), ChallengerBudget(10))[1] == "UNSAFE_CROP"


def test_valid_primary_and_full_page_bypass_challenger():
    assert route_to_ppocr_v5(context(primary_resolved=True), ChallengerBudget(10))[1] == "VALID_PRIMARY_BYPASS"
    assert route_to_ppocr_v5(context(request_scope="FULL_PAGE"), ChallengerBudget(10))[1] == "FULL_PAGE_REJECTED"


def test_budget_never_exceeds_thirty_percent():
    budget = ChallengerBudget(10)
    results = [route_to_ppocr_v5(context(), budget)[0] for _ in range(6)]
    assert results == [True, True, True, False, False, False]
    assert budget.used == 3
