from evaluation.external_corpus_runner import aggregate_predictions


def _prediction(document_id: str, route: str, disposition: str, claim_disposition: str | None = None):
    return {
        "document_id": document_id,
        "route": route,
        "schema": route,
        "fields": {
            "member_id": {
                "value": "SECRET-MEMBER-ID",
                "decision": {"disposition": disposition},
            }
        },
        "claim_decision": ({"disposition": claim_disposition} if claim_disposition else None),
        "stage_seconds": {"ocr": 1.0, "evidence": 0.2},
        "wall_seconds": 2.0,
        "cpu_seconds": 1.5,
        "cloud_cost_usd": 0,
    }


def test_aggregate_is_truth_blind_and_does_not_emit_field_values():
    predictions = [
        _prediction("p1", "CMS1500", "AUTO_ACCEPTED", "STP_APPROVED"),
        _prediction("p2", "UB04", "HITL_REQUIRED", "REVIEW_REQUIRED"),
    ]
    lineage = {
        "p1": {"group": "Group A", "package_id": "pkg1"},
        "p2": {"group": "Group B", "package_id": "pkg2"},
    }
    result = aggregate_predictions(predictions, lineage)
    rendered = str(result)
    assert "SECRET-MEMBER-ID" not in rendered
    assert result["qualification"]["truth_blind"] is True
    assert result["qualification"]["accuracy_scored"] is False
    assert result["routing"]["distribution"] == {"CMS1500": 1, "UB04": 1}
    assert result["decision"]["decided_fields"] == 2
    assert result["decision"]["accepted_fields"] == 1
    assert result["claim"]["packages_with_review_signal"] == 1
    assert result["latency_seconds"]["p95"] == 2.0


def test_aggregate_reports_zero_field_pages_without_fabricating_decisions():
    predictions = [{
        "document_id": "p1",
        "route": "UNKNOWN_UNSTRUCTURED",
        "schema": "UNKNOWN_UNSTRUCTURED",
        "fields": {},
        "claim_decision": None,
        "stage_seconds": {},
        "wall_seconds": 0.5,
        "cpu_seconds": 0.4,
        "cloud_cost_usd": 0,
    }]
    lineage = {"p1": {"group": "Group C", "package_id": "pkg1"}}
    result = aggregate_predictions(predictions, lineage)
    assert result["extraction"]["pages_with_zero_fields"] == 1
    assert result["decision"]["decided_fields"] == 0
    assert result["decision"]["safe_coverage_signal"] is None
