import hashlib
import json
from pathlib import Path

import pytest

from evaluation.build_strict_identity_routing_review import build, score

SOURCE_MANIFEST_SHA = "a" * 64
CURRENT_POLICY_SHA = "c" * 64
PRECEDING_POLICY_SHA = "d" * 64


def _record(
    index: int,
    *,
    fixed: bool = False,
    conflict: bool = False,
    candidate: str | None = None,
    policy_sha: str = CURRENT_POLICY_SHA,
) -> dict:
    page_id = f"page-{index:03d}"
    sha = f"{index + 1:064x}"
    family = "CMS1500" if fixed else None
    return {
        "source_page_id": page_id,
        "source_page_sha256": sha,
        "package_id": f"package-{index % 3}",
        "source_asset_id": f"asset-{index}",
        "source_page_number": 1,
        "candidate_class": candidate or ("CMS1500" if fixed else "UNKNOWN"),
        "confidence_band": "HIGH" if fixed else "REVIEW_REQUIRED",
        "ocr_provenance": {"rendered_page_sha256": sha},
        "decision_provenance": {"decision_policy_sha256": policy_sha},
        "routing_result": {
            "router_nomination": "CMS1500" if fixed else "UNKNOWN_UNSTRUCTURED"
        },
        "production_chain": {
            "fixed_extractor_authorized": fixed,
            "verified_identity_family": family,
            "processing_route": "CMS_STANDARD_EXTRACTOR" if fixed else "SAFE_UNKNOWN",
            "decision_reason_codes": [],
        },
        "form_identity": {
            "localization_allowed": fixed,
            "family_eligibility": {},
            "conflicting_anchors": {"CMS1500": ["CONFLICT"] if conflict else []},
        },
    }


def _write(directory: Path, records: list[dict]) -> None:
    directory.mkdir(parents=True)
    for record in records:
        (directory / f"{record['source_page_id']}.json").write_text(
            json.dumps(record), "utf-8"
        )


def _setup(tmp_path: Path, records: list[dict], rapid_size: int | None = None):
    current = tmp_path / "current"
    preceding = tmp_path / "preceding"
    output = tmp_path / "output"
    _write(current, records)
    older = json.loads(json.dumps(records))
    for record in older:
        record["decision_provenance"]["decision_policy_sha256"] = PRECEDING_POLICY_SHA
    _write(preceding, older)
    result = build(
        current,
        preceding,
        output,
        rapid_size=rapid_size or len(records),
        expected_page_count=len(records),
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        blind_salt="fixed-test-salt",
    )
    return current, preceding, output, result


def _review(task: dict, role: str, label: str, *, ambiguity: bool = False) -> dict:
    suffix = "a" if role == "REVIEWER_A" else "b"
    return {
        "blind_task_id": task["blind_task_id"],
        "source_page_id": task["source_page_id"],
        "source_page_sha256": task["source_page_sha256"],
        "reviewed_label": label,
        "identity_ambiguity": ambiguity,
        "review_status": "COMPLETED",
        "reviewer_id": f"reviewer-{suffix}",
        "review_session_id": f"session-{task['source_page_id']}-{suffix}",
        "reviewer_role": role,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")


def _resign(queue: dict) -> None:
    queue.pop("queue_content_sha256", None)
    queue["queue_content_sha256"] = hashlib.sha256(
        json.dumps(queue, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_builds_blind_bound_queues_without_risk_order_leakage(tmp_path):
    records = [
        _record(0, fixed=True),
        _record(1, conflict=True),
        *[_record(index) for index in range(2, 12)],
    ]
    current, preceding, output, result = _setup(tmp_path, records, rapid_size=5)
    preceding_record = json.loads((preceding / "page-002.json").read_text("utf-8"))
    preceding_record["form_identity"]["localization_allowed"] = True
    (preceding / "page-002.json").write_text(json.dumps(preceding_record), "utf-8")
    result = build(
        current,
        preceding,
        output,
        rapid_size=5,
        expected_page_count=12,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        blind_salt="fixed-test-salt",
    )

    rapid = json.loads((output / "rapid_blind_queue.json").read_text("utf-8"))
    full = json.loads((output / "full_blind_queue.json").read_text("utf-8"))
    coordinator = json.loads((output / "coordinator_risk_manifest.json").read_text("utf-8"))
    provenance = json.loads((output / "trusted_label_provenance.json").read_text("utf-8"))

    assert result["rapid_pages"] == 5
    assert result["full_pages"] == 12
    assert result["full_double_review_pages"] == 12
    assert rapid["page_set_sha256"] == full["page_set_sha256"]
    assert rapid["source_manifest_sha256"] == SOURCE_MANIFEST_SHA
    assert rapid["current_decision_policy_sha256"] == CURRENT_POLICY_SHA
    assert {"page-000", "page-001", "page-002"} <= {
        row["source_page_id"] for row in rapid["records"]
    }
    assert all(
        "candidate_class" not in record
        and "router_nomination" not in record
        and "risk_reasons" not in record
        for record in full["records"]
    )
    priorities = [
        coordinator["records"][record["source_page_id"]]["priority"]
        for record in full["records"]
    ]
    assert priorities != sorted(priorities)
    assert coordinator["access"] == "COORDINATOR_ONLY_NOT_FOR_INITIAL_ANNOTATORS"
    assert provenance["admissible_exact_bindings"] is None
    assert result["status"] == "BLOCKED_LABEL_PROVENANCE_UNAVAILABLE"


def test_mandatory_risk_expands_rapid_target_instead_of_dropping_pages(tmp_path):
    records = [_record(index, fixed=True) for index in range(4)] + [_record(4)]
    _, _, _, result = _setup(tmp_path, records, rapid_size=2)
    assert result["rapid_target_pages"] == 2
    assert result["rapid_pages"] == 4


def test_trusted_label_provenance_is_derived_from_exact_bound_records(tmp_path):
    records = [_record(0), _record(1)]
    current = tmp_path / "current"
    preceding = tmp_path / "preceding"
    output = tmp_path / "output"
    _write(current, records)
    older = json.loads(json.dumps(records))
    for record in older:
        record["decision_provenance"]["decision_policy_sha256"] = PRECEDING_POLICY_SHA
    _write(preceding, older)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_page_id": records[0]["source_page_id"],
                        "source_page_sha256": records[0]["source_page_sha256"],
                        "label_authority": "INDEPENDENT_HUMAN_ADJUDICATION",
                    },
                    {
                        "source_page_id": records[1]["source_page_id"],
                        "source_page_sha256": records[1]["source_page_sha256"],
                        "label_authority": "MODEL_PREDICTION",
                    },
                ]
            }
        ),
        "utf-8",
    )
    build(
        current,
        preceding,
        output,
        expected_page_count=2,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        blind_salt="fixed-test-salt",
        provenance_path=provenance_path,
    )
    provenance = json.loads((output / "trusted_label_provenance.json").read_text("utf-8"))
    assert provenance["records_declared"] == 2
    assert provenance["admissible_exact_bindings"] == 1
    assert provenance["accuracy_scoring_authorized"] is True


def test_rejects_stale_sha_partial_page_set_and_mixed_policy(tmp_path):
    current, preceding = tmp_path / "current", tmp_path / "preceding"
    stale = _record(0)
    stale["ocr_provenance"]["rendered_page_sha256"] = "f" * 64
    _write(current, [stale])
    older = _record(0, policy_sha=PRECEDING_POLICY_SHA)
    _write(preceding, [older])
    with pytest.raises(ValueError, match="PAGE_SHA_MISMATCH"):
        build(
            current,
            preceding,
            tmp_path / "output",
            expected_page_count=1,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
        )

    current2, preceding2 = tmp_path / "current2", tmp_path / "preceding2"
    _write(current2, [_record(0), _record(1, policy_sha="e" * 64)])
    _write(
        preceding2,
        [_record(0, policy_sha=PRECEDING_POLICY_SHA), _record(1, policy_sha=PRECEDING_POLICY_SHA)],
    )
    with pytest.raises(ValueError, match="MIXED_DECISION_POLICY_SHA"):
        build(
            current2,
            preceding2,
            tmp_path / "output2",
            expected_page_count=2,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
        )

    current3, preceding3 = tmp_path / "current3", tmp_path / "preceding3"
    _write(current3, [_record(0)])
    _write(preceding3, [_record(0, policy_sha=PRECEDING_POLICY_SHA)])
    with pytest.raises(ValueError, match="PARTIAL_REPLAY"):
        build(
            current3,
            preceding3,
            tmp_path / "output3",
            expected_page_count=2,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
        )


def test_scoring_rejects_queue_tampering_and_duplicate_tasks(tmp_path):
    current, _, output, _ = _setup(tmp_path, [_record(0)])
    path = output / "rapid_blind_queue.json"
    queue = json.loads(path.read_text("utf-8"))
    queue["records"][0]["source_page_sha256"] = "f" * 64
    path.write_text(json.dumps(queue), "utf-8")
    with pytest.raises(ValueError, match="QUEUE_CONTENT_SHA_MISMATCH"):
        score(path, current, output / "reviews.jsonl", output / "adjudications.jsonl")

    queue = json.loads((output / "full_blind_queue.json").read_text("utf-8"))
    queue["records"].append(dict(queue["records"][0]))
    queue["records"][-1]["queue_position"] = 2
    _resign(queue)
    path.write_text(json.dumps(queue), "utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_BLIND_TASK_ID"):
        score(path, current, output / "reviews.jsonl", output / "adjudications.jsonl")


def test_scoring_blocks_with_null_metrics_until_reviews_complete(tmp_path):
    current, _, output, _ = _setup(tmp_path, [_record(0, fixed=True), _record(1)])
    result = score(
        output / "rapid_blind_queue.json",
        current,
        output / "reviews.jsonl",
        output / "adjudications.jsonl",
    )
    assert result["status"] == "BLOCKED_HUMAN_LABELS"
    assert result["progress"]["scorable_pages"] == 0
    assert result["progress"]["unscorable_pages"] == 2
    assert all(value is None for value in result["metrics"].values())


@pytest.mark.parametrize(
    ("label", "ambiguity"),
    [("CMS1500", False), ("UNKNOWN", False), ("OTHER_CLAIM_FORM", True)],
)
def test_dynamic_standard_unknown_or_ambiguous_label_requires_second_review(
    tmp_path, label, ambiguity
):
    current, _, output, _ = _setup(tmp_path, [_record(0)])
    task = json.loads((output / "rapid_blind_queue.json").read_text("utf-8"))["records"][0]
    assert task["required_independent_reviews"] == 1
    _write_jsonl(output / "reviews.jsonl", [_review(task, "REVIEWER_A", label, ambiguity=ambiguity)])
    result = score(
        output / "rapid_blind_queue.json",
        current,
        output / "reviews.jsonl",
        output / "adjudications.jsonl",
    )
    assert result["status"] == "BLOCKED_HUMAN_LABELS"
    assert result["progress"]["pending_reviews"] == 1


def test_metrics_use_risk_scope_nonstandard_denominator_and_production_review_state(tmp_path):
    records = [
        _record(0, fixed=True),
        _record(1, fixed=True),
        _record(2, candidate="OTHER_CLAIM_FORM"),
        _record(3, candidate="NON_CLAIM"),
    ]
    records[1]["candidate_class"] = "UB04"
    records[1]["routing_result"]["router_nomination"] = "UB04"
    records[1]["production_chain"]["verified_identity_family"] = "UB04"
    records[1]["production_chain"]["processing_route"] = "UB_STANDARD_EXTRACTOR"
    current, _, output, _ = _setup(tmp_path, records)
    queue = json.loads((output / "rapid_blind_queue.json").read_text("utf-8"))
    truth = {
        "page-000": "NON_CLAIM",
        "page-001": "UB04",
        "page-002": "OTHER_CLAIM_FORM",
        "page-003": "NON_CLAIM",
    }
    reviews = []
    for task in queue["records"]:
        reviews.append(_review(task, "REVIEWER_A", truth[task["source_page_id"]]))
        if task["required_independent_reviews"] == 2:
            reviews.append(_review(task, "REVIEWER_B", truth[task["source_page_id"]]))
    _write_jsonl(output / "reviews.jsonl", reviews)
    result = score(
        output / "rapid_blind_queue.json",
        current,
        output / "reviews.jsonl",
        output / "adjudications.jsonl",
    )
    metrics = result["metrics"]
    assert result["status"] == "ROUTING_PRELIMINARY_RISK_COHORT"
    assert result["metric_scope"]["routing_accuracy"] == (
        "RISK_ENRICHED_COHORT_NOT_A_CORPUS_ESTIMATE"
    )
    assert "corpus_overall_exact_accuracy" not in metrics
    assert metrics["rapid_risk_cohort_exact_accuracy"]["denominator"] == 4
    assert metrics["fixed_authorization_precision"]["numerator"] == 1
    assert metrics["fixed_authorization_precision"]["denominator"] == 2
    assert metrics["false_standard_authorization_rate"]["numerator"] == 1
    assert metrics["false_standard_authorization_rate"]["denominator"] == 3
    assert metrics["wrong_family_authorization_count"] == 0
    assert metrics["page_review_abstention_rate"]["numerator"] == 2


def test_disagreement_requires_independent_adjudication_and_rejects_duplicates(tmp_path):
    current, _, output, _ = _setup(tmp_path, [_record(0, fixed=True)])
    task = json.loads((output / "rapid_blind_queue.json").read_text("utf-8"))["records"][0]
    reviews = [
        _review(task, "REVIEWER_A", "CMS1500"),
        _review(task, "REVIEWER_B", "UB04"),
    ]
    _write_jsonl(output / "reviews.jsonl", reviews)
    blocked = score(
        output / "rapid_blind_queue.json",
        current,
        output / "reviews.jsonl",
        output / "adjudications.jsonl",
    )
    assert blocked["progress"]["pending_adjudications"] == 1
    adjudication = {
        "blind_task_id": task["blind_task_id"],
        "source_page_id": task["source_page_id"],
        "source_page_sha256": task["source_page_sha256"],
        "final_label": "CMS1500",
        "adjudication_status": "COMPLETED",
        "adjudicator_id": "adjudicator",
        "adjudication_session_id": "adjudication-session",
    }
    _write_jsonl(output / "adjudications.jsonl", [adjudication, adjudication])
    with pytest.raises(ValueError, match="DUPLICATE_ADJUDICATION_EVENT"):
        score(
            output / "rapid_blind_queue.json",
            current,
            output / "reviews.jsonl",
            output / "adjudications.jsonl",
        )
    _write_jsonl(output / "adjudications.jsonl", [adjudication])
    result = score(
        output / "rapid_blind_queue.json",
        current,
        output / "reviews.jsonl",
        output / "adjudications.jsonl",
    )
    assert result["status"] == "ROUTING_PRELIMINARY_RISK_COHORT"
    assert result["metrics"]["adjudication_count"] == 1


def test_duplicate_and_nonindependent_reviews_are_rejected(tmp_path):
    current, _, output, _ = _setup(tmp_path, [_record(0, fixed=True)])
    task = json.loads((output / "rapid_blind_queue.json").read_text("utf-8"))["records"][0]
    review = _review(task, "REVIEWER_A", "CMS1500")
    _write_jsonl(output / "reviews.jsonl", [review, review])
    with pytest.raises(ValueError, match="DUPLICATE_REVIEW_EVENT"):
        score(
            output / "rapid_blind_queue.json",
            current,
            output / "reviews.jsonl",
            output / "adjudications.jsonl",
        )

    second = _review(task, "REVIEWER_B", "CMS1500")
    second["reviewer_id"] = review["reviewer_id"]
    _write_jsonl(output / "reviews.jsonl", [review, second])
    with pytest.raises(ValueError, match="REVIEWS_NOT_INDEPENDENT"):
        score(
            output / "rapid_blind_queue.json",
            current,
            output / "reviews.jsonl",
            output / "adjudications.jsonl",
        )


def test_full_queue_requires_two_reviews_for_every_page(tmp_path):
    _, _, output, _ = _setup(tmp_path, [_record(0), _record(1), _record(2)])
    queue = json.loads((output / "full_blind_queue.json").read_text("utf-8"))
    assert all(task["required_independent_reviews"] == 2 for task in queue["records"])
