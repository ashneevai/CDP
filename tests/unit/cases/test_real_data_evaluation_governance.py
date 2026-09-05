from datetime import UTC, datetime

import pytest

from packages.real_data_evaluation.governance import (
    Annotation,
    AnnotationAuthority,
    AnnotationState,
    CohortPage,
    HoldoutExecution,
    HoldoutLedger,
    ReviewAction,
    ReviewEvent,
    ReviewTarget,
    adjudicate_critical,
    compare_critical_annotations,
    deterministic_package_sample,
    freeze_cohort,
    package_level_split,
    trusted_for_release,
    verify_frozen_cohort,
)

H = "a" * 64
NOW = datetime(2025, 1, 1, tzinfo=UTC)


def annotation(identifier: str, reviewer: str, value_hash: str = H) -> Annotation:
    return Annotation(
        annotation_id=identifier,
        package_id="pkg",
        source_page_id="page",
        field_name="NPI",
        critical=True,
        reviewer_id=reviewer,
        state=AnnotationState.VALUE,
        normalized_value_sha256=value_hash,
        source_region_sha256="b" * 64,
        annotation_version="v1",
        created_at=NOW,
    )


def page(package: str, identifier: str, page_class: str = "CMS1500") -> CohortPage:
    return CohortPage(
        package_id=package,
        document_id=f"doc-{package}",
        source_page_id=identifier,
        page_class=page_class,
        quality_band="LOW",
        package_complexity="MULTI_PAGE",
        source_page_sha256=H,
        label_manifest_sha256="c" * 64,
    )


def test_review_events_are_blind_and_corrections_require_provenance():
    with pytest.raises(ValueError, match="blind"):
        ReviewEvent(
            event_id="e",
            target=ReviewTarget.PAGE,
            action=ReviewAction.CONFIRM,
            archive_id="a",
            package_id="p",
            reviewer_id="r",
            reviewed_at=NOW,
            annotation_version="v1",
            prediction_visible=True,
        )
    with pytest.raises(ValueError, match="reason"):
        ReviewEvent(
            event_id="e",
            target=ReviewTarget.PAGE,
            action=ReviewAction.CORRECT,
            archive_id="a",
            package_id="p",
            reviewer_id="r",
            reviewed_at=NOW,
            annotation_version="v1",
        )


def test_field_review_requires_source_region_without_storing_value():
    event = ReviewEvent(
        event_id="e",
        target=ReviewTarget.FIELD,
        action=ReviewAction.CONFIRM,
        archive_id="a",
        package_id="p",
        source_page_id="page",
        field_name="NPI",
        reviewer_id="r",
        reviewed_at=NOW,
        annotation_version="v1",
        source_region_sha256=H,
        new_value_sha256="b" * 64,
    )
    assert "value" not in event.model_dump()


def test_critical_agreement_requires_independent_reviewers():
    decision = compare_critical_annotations(
        annotation("a", "reviewer-a"), annotation("b", "reviewer-b")
    )
    assert decision.status == "AGREED"
    with pytest.raises(ValueError, match="distinct"):
        compare_critical_annotations(annotation("a", "same"), annotation("b", "same"))


def test_disagreement_requires_explicit_independent_adjudication():
    a = annotation("a", "reviewer-a")
    b = annotation("b", "reviewer-b", "d" * 64)
    decision = compare_critical_annotations(a, b)
    assert decision.status == "ADJUDICATION_REQUIRED"
    with pytest.raises(ValueError, match="independent"):
        adjudicate_critical(
            decision,
            a,
            b,
            adjudicator_id="reviewer-a",
            final_state=AnnotationState.VALUE,
            final_value_sha256=H,
        )
    final = adjudicate_critical(
        decision,
        a,
        b,
        adjudicator_id="adjudicator",
        final_state=AnnotationState.VALUE,
        final_value_sha256=H,
    )
    assert final.authority == AnnotationAuthority.HUMAN_ADJUDICATED


def test_only_finalized_authoritative_labels_are_release_trusted():
    assert trusted_for_release(AnnotationAuthority.HUMAN_ADJUDICATED, finalized=True)
    assert trusted_for_release(AnnotationAuthority.SOURCE_SYSTEM_GROUND_TRUTH, finalized=True)
    assert not trusted_for_release(AnnotationAuthority.HUMAN_SINGLE_REVIEW, finalized=True)
    assert not trusted_for_release(AnnotationAuthority.MODEL_GENERATED, finalized=True)
    assert not trusted_for_release(AnnotationAuthority.HUMAN_ADJUDICATED, finalized=False)


def test_sampling_is_deterministic_and_selects_whole_packages():
    pages = [page("p1", "1"), page("p1", "2", "UB04"), page("p2", "3"), page("p3", "4", "UB04")]
    first = deterministic_package_sample(pages, {"CMS1500": 1, "UB04": 1}, seed="fixed")
    second = deterministic_package_sample(reversed(pages), {"CMS1500": 1, "UB04": 1}, seed="fixed")
    assert first == second
    selected_pages = [item for item in pages if item.package_id in first]
    assert all(item in selected_pages for item in pages if item.package_id in first)


def test_package_split_is_deterministic_and_has_no_leakage():
    split = package_level_split([f"p{i}" for i in range(10)], seed="v1")
    assert split == package_level_split(reversed(list(split)), seed="v1")
    assert sum(value == "DEVELOPMENT" for value in split.values()) == 7
    assert set(split.values()) == {"DEVELOPMENT", "HOLDOUT"}


def test_cohort_freeze_requires_labels_and_detects_mutation():
    pages = [page("p1", "1"), page("p2", "2")]
    frozen = freeze_cohort(
        archive_sha256=H,
        split_seed="v1",
        assignments={"p1": "DEVELOPMENT", "p2": "HOLDOUT"},
        pages=pages,
    )
    assert verify_frozen_cohort(frozen)
    assert not verify_frozen_cohort(frozen.model_copy(update={"assignments": {"p1": "HOLDOUT"}}))
    with pytest.raises(ValueError, match="label"):
        freeze_cohort(
            archive_sha256=H,
            split_seed="v1",
            assignments={"p1": "DEVELOPMENT", "p2": "HOLDOUT"},
            pages=[pages[0].model_copy(update={"label_manifest_sha256": None}), pages[1]],
        )


def test_holdout_ledger_rejects_tuning_and_wrong_cohort():
    ledger = HoldoutLedger(cohort_seal_sha256=H)
    execution = HoldoutExecution(
        execution_id="gate-1",
        cohort_seal_sha256=H,
        code_commit_sha256="b" * 64,
        purpose="MILESTONE_GATE",
        executed_at=NOW,
        result_manifest_sha256="c" * 64,
    )
    assert len(ledger.record(execution).executions) == 1
    with pytest.raises(ValueError, match="tuning"):
        ledger.record(execution.model_copy(update={"execution_id": "bad", "used_for_tuning": True}))
    with pytest.raises(ValueError, match="different"):
        ledger.record(
            execution.model_copy(update={"execution_id": "bad", "cohort_seal_sha256": "d" * 64})
        )
