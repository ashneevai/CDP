from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from packages.criticality import CriticalityLevel
from packages.domain.common import BoundingBox
from packages.evidence_decision import DecisionContext
from packages.hitl_reduction import (
    BlindReviewSubmission,
    ClaimRuntimeRecord,
    GovernedFieldLabel,
    HITLReductionInput,
    HITLReductionService,
    build_review_assignments,
    compile_review_submissions,
    verify_review_assignment,
)
from packages.hitl_reduction.contracts import (
    FieldRuntimeRecord,
    LabelAuthority,
    LabelDisposition,
    ReviewObservation,
)
from packages.ocr.contracts import OCRCandidate
from packages.ocr.provenance import EvidenceProvenance

PAGE_SHA = "a" * 64
CROP_SHA = "b" * 64
MANIFEST_SHA = "c" * 64


def _candidate(*, page_sha: str = PAGE_SHA, provenance: bool = True) -> OCRCandidate:
    box = BoundingBox(x0=1, y0=1, x1=20, y1=10, image_width=100, image_height=100)
    lineage = None
    if provenance:
        lineage = EvidenceProvenance(
            page_sha256=page_sha,
            crop_sha256=CROP_SHA,
            source_representation_id="render-v1",
            observation_id="observation-1",
            localization_id="canonical-patient-name",
            preprocessing_profile="recorded-canonical-field-crop-v1",
            preprocessing_sha256="d" * 64,
            engine_family="tesseract",
            engine_name="tesseract",
            engine_version="5",
            model_name="eng",
            model_version="5",
        )
    return OCRCandidate(
        value="Jane Doe",
        raw_value="Jane Doe",
        engine="tesseract",
        model_name="eng",
        model_version="5",
        preprocessing_variant="canonical",
        raw_confidence=0.99,
        calibrated_confidence=0.99,
        bounding_box=box,
        latency_ms=10,
        provenance=lineage,
    )


def _cohort(*, with_candidate: bool = False, hard_validation: bool = False):
    context = DecisionContext(
        field_id="field-1",
        field_name="patient_name",
        document_family="CMS1500",
        criticality=CriticalityLevel.C2,
        candidates=[_candidate()] if with_candidate else [],
        hard_validation_passed=hard_validation,
        registration_confidence=0.99,
    )
    field = FieldRuntimeRecord(
        field_instance_id="field-1",
        document_id="document-1",
        page_id="page-1",
        page_sha256=PAGE_SHA,
        crop_sha256=CROP_SHA,
        crop_reference="s3://blind-crops/field-1.png",
        source_segment="SOURCE_A",
        quality_segment="clean",
        latency_ms=12,
        decision_context=context,
    )
    claim = ClaimRuntimeRecord(
        claim_id="claim-1",
        document_family="CMS1500",
        source_segment="SOURCE_A",
        quality_segment="clean",
        fields=[field],
        enforce_configured_required_fields=False,
    )
    return HITLReductionInput(
        cohort_id="holdout-1",
        input_manifest_sha256=MANIFEST_SHA,
        holdout_frozen=True,
        holdout_independent=True,
        claims=[claim],
    )


@pytest.fixture(scope="module")
def service() -> HITLReductionService:
    return HITLReductionService()


def _prepared(service: HITLReductionService, **kwargs):
    return service.prepare(_cohort(**kwargs))


def _human_label(
    prepared: dict,
    *,
    reviews: list[ReviewObservation],
    final_value: str = "Jane Doe",
    adjudication: ReviewObservation | None = None,
    crop_sha: str = CROP_SHA,
) -> GovernedFieldLabel:
    sealed = prepared["sealed_predictions"]
    task = prepared["blind_review_queue"]["tasks"][0]
    return GovernedFieldLabel(
        field_instance_id="field-1",
        document_id="document-1",
        page_id="page-1",
        page_sha256=PAGE_SHA,
        crop_sha256=crop_sha,
        blind_task_id=task["blind_task_id"],
        prediction_seal_sha256=sealed["prediction_seal_sha256"],
        authority=LabelAuthority.HUMAN_ADJUDICATED,
        final_disposition=LabelDisposition.VALUE,
        final_value=final_value,
        reviews=reviews,
        adjudication=adjudication,
    )


def _after_seal(prepared: dict) -> datetime:
    sealed_at = datetime.fromisoformat(prepared["sealed_predictions"]["sealed_at"])
    return sealed_at + timedelta(seconds=1)


def test_prepare_seals_predictions_and_emits_prediction_free_blind_tasks(service):
    prepared = _prepared(service, with_candidate=True, hard_validation=True)
    task = prepared["blind_review_queue"]["tasks"][0]
    assert set(task) == {
        "blind_task_id",
        "field_instance_id",
        "field_name",
        "document_id",
        "page_id",
        "page_sha256",
        "crop_sha256",
        "crop_reference",
        "criticality",
        "required_independent_reviews",
        "adjudication_required_on_disagreement",
    }
    assert task["required_independent_reviews"] == 2
    assert prepared["current_metrics"]["accuracy"] is None
    assert len(prepared["sealed_predictions"]["prediction_seal_sha256"]) == 64


def test_score_rejects_prediction_tampering(service):
    prepared = _prepared(service)
    tampered = deepcopy(prepared["sealed_predictions"])
    tampered["claims"][0]["fields"][0]["runtime_decision"]["selected_value"] = "leak"
    with pytest.raises(ValueError, match="PREDICTION_SEAL_INVALID"):
        service.score(tampered, [])


def test_exact_label_lineage_mismatch_is_rejected(service):
    prepared = _prepared(service)
    reviewed_at = _after_seal(prepared)
    label = _human_label(
        prepared,
        crop_sha="e" * 64,
        reviews=[
            ReviewObservation(
                reviewer_id="reviewer-1",
                reviewed_at=reviewed_at,
                disposition=LabelDisposition.VALUE,
                value="Jane Doe",
            ),
            ReviewObservation(
                reviewer_id="reviewer-2",
                reviewed_at=reviewed_at,
                disposition=LabelDisposition.VALUE,
                value="Jane Doe",
            ),
        ],
    )
    with pytest.raises(ValueError, match="LABEL_LINEAGE_MISMATCH:field-1:crop_sha256"):
        service.score(prepared["sealed_predictions"], [label])


def test_critical_human_label_requires_two_independent_reviews(service):
    prepared = _prepared(service)
    label = _human_label(
        prepared,
        reviews=[
            ReviewObservation(
                reviewer_id="reviewer-1",
                reviewed_at=_after_seal(prepared),
                disposition=LabelDisposition.VALUE,
                value="Jane Doe",
            )
        ],
    )
    result = service.score(prepared["sealed_predictions"], [label])
    assert result["scored_metrics"]["eligible_labels"] == 0
    assert result["scored_metrics"]["status"] == "BLOCKED_HUMAN_LABELS"
    assert result["label_audit"]["records"][0]["reasons"] == ["INDEPENDENT_REVIEWS_INCOMPLETE"]
    assert result["qualification"]["decision"] == "NEEDS_MORE_DATA"


def test_disagreement_requires_third_independent_adjudicator(service):
    prepared = _prepared(service)
    reviewed_at = _after_seal(prepared)
    reviews = [
        ReviewObservation(
            reviewer_id="reviewer-1",
            reviewed_at=reviewed_at,
            disposition=LabelDisposition.VALUE,
            value="Jane Doe",
        ),
        ReviewObservation(
            reviewer_id="reviewer-2",
            reviewed_at=reviewed_at,
            disposition=LabelDisposition.VALUE,
            value="Janet Doe",
        ),
    ]
    blocked = service.score(
        prepared["sealed_predictions"],
        [_human_label(prepared, reviews=reviews)],
    )
    assert blocked["label_audit"]["records"][0]["reasons"] == ["DISAGREEMENT_REQUIRES_ADJUDICATION"]
    adjudication = ReviewObservation(
        reviewer_id="adjudicator-1",
        reviewed_at=reviewed_at + timedelta(seconds=1),
        disposition=LabelDisposition.VALUE,
        value="Janet Doe",
    )
    scored = service.score(
        prepared["sealed_predictions"],
        [
            _human_label(
                prepared,
                reviews=reviews,
                final_value="Janet Doe",
                adjudication=adjudication,
            )
        ],
    )
    assert scored["scored_metrics"]["eligible_labels"] == 1
    assert scored["scored_metrics"]["status"] == "BLOCKED_GATES"


def test_source_label_requires_complete_independent_snapshot_provenance(service):
    prepared = _prepared(service)
    sealed = prepared["sealed_predictions"]
    task = prepared["blind_review_queue"]["tasks"][0]
    label = GovernedFieldLabel(
        field_instance_id="field-1",
        document_id="document-1",
        page_id="page-1",
        page_sha256=PAGE_SHA,
        crop_sha256=CROP_SHA,
        blind_task_id=task["blind_task_id"],
        prediction_seal_sha256=sealed["prediction_seal_sha256"],
        authority=LabelAuthority.SOURCE_SYSTEM_GROUND_TRUTH,
        final_disposition=LabelDisposition.VALUE,
        final_value="Jane Doe",
    )
    result = service.score(sealed, [label])
    assert result["scored_metrics"]["eligible_labels"] == 0
    assert result["label_audit"]["records"][0]["reasons"] == [
        "SOURCE_PROVENANCE_INCOMPLETE",
        "SOURCE_NOT_INDEPENDENT_AND_NON_CIRCULAR",
    ]


def test_complete_toy_holdout_scores_but_cannot_bypass_sample_gate(service):
    prepared = _prepared(service)
    sealed = prepared["sealed_predictions"]
    task = prepared["blind_review_queue"]["tasks"][0]
    label = GovernedFieldLabel(
        field_instance_id="field-1",
        document_id="document-1",
        page_id="page-1",
        page_sha256=PAGE_SHA,
        crop_sha256=CROP_SHA,
        blind_task_id=task["blind_task_id"],
        prediction_seal_sha256=sealed["prediction_seal_sha256"],
        authority=LabelAuthority.SOURCE_SYSTEM_GROUND_TRUTH,
        final_disposition=LabelDisposition.VALUE,
        final_value="Jane Doe",
        source_system="payer-core",
        source_version="2026-09",
        source_snapshot_sha256="f" * 64,
        source_record_id="member-1",
        source_independent=True,
        source_non_circular=True,
    )
    result = service.score(sealed, [label])
    assert result["scored_metrics"]["label_coverage"] == 1
    assert result["scored_metrics"]["runtime"]["raw_accuracy"] == 0
    assert result["scored_metrics"]["status"] == "BLOCKED_GATES"
    assert result["qualification"]["gates"]["sample_size"] is False
    assert result["qualification"]["gates"]["measured_cost"] is False


def test_blocker_pareto_and_metrics_separate_recovery_from_human_review(service):
    prepared = _prepared(service, hard_validation=False)
    metrics = prepared["current_metrics"]["runtime"]
    assert metrics["field_hitl_count"] == 0
    assert metrics["machine_recovery_pending_count"] == 1
    assert metrics["claim_hitl_count"] == 1
    top = prepared["blocker_pareto"]["rows"][0]
    assert top["field_name"] == "patient_name"
    assert top["single_blocker_claim_unlocks"] == 1
    assert top["next_action"] == "CROP_RECOVERY"


def test_evaluation_only_route_never_gains_runtime_authority(service):
    prepared = _prepared(service, with_candidate=True, hard_validation=True)
    field = prepared["sealed_predictions"]["claims"][0]["fields"][0]
    assert field["runtime_decision"]["selected_value"] is None
    assert field["runtime_decision"]["route_mode"] == "runtime"
    assert any(
        reason.startswith("ROUTE_STATUS_REJECTED:")
        for reason in field["runtime_decision"]["reason_codes"]
    )
    assert field["evaluation_decision"]["selected_value"] == "Jane Doe"
    assert field["evaluation_decision"]["route_mode"] == "evaluation"


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        (_candidate(provenance=False), "CANDIDATE_PROVENANCE_REQUIRED"),
        (_candidate(page_sha="e" * 64), "CANDIDATE_PAGE_SHA256_MISMATCH"),
    ],
)
def test_candidate_lineage_is_mandatory(candidate, error):
    context = DecisionContext(
        field_id="field-1",
        field_name="patient_name",
        document_family="CMS1500",
        criticality=CriticalityLevel.C2,
        candidates=[candidate],
    )
    with pytest.raises(ValidationError, match=error):
        FieldRuntimeRecord(
            field_instance_id="field-1",
            document_id="document-1",
            page_id="page-1",
            page_sha256=PAGE_SHA,
            crop_sha256=CROP_SHA,
            crop_reference="s3://blind-crops/field-1.png",
            latency_ms=10,
            decision_context=context,
        )


def test_review_timestamp_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="REVIEW_TIMESTAMP_MUST_BE_TIMEZONE_AWARE"):
        ReviewObservation(
            reviewer_id="reviewer-1",
            reviewed_at=datetime.now(UTC).replace(tzinfo=None),
            disposition=LabelDisposition.VALUE,
            value="Jane Doe",
        )
    assert datetime.now(UTC).tzinfo is not None


def test_critical_review_assignments_are_isolated_and_reserve_adjudicator(service):
    queue = _prepared(service)["blind_review_queue"]
    result = build_review_assignments(queue, ["reviewer-a", "reviewer-b", "reviewer-c"])
    manifest = result["review_assignment_manifest"]
    task = manifest["tasks"][0]
    assert manifest["review_assignment_count"] == 2
    assert len(task["assigned_reviewer_ids"]) == 2
    assert len(task["eligible_adjudicator_ids"]) == 1
    assert set(task["assigned_reviewer_ids"]).isdisjoint(task["eligible_adjudicator_ids"])
    packs = [result[name] for name in sorted(result) if name.startswith("reviewer_")]
    for pack in packs:
        assert "eligible_adjudicator_ids" not in pack
        assert "assigned_reviewer_ids" not in pack
        for blind_task in pack["tasks"]:
            assert "runtime_decision" not in blind_task
            assert "evaluation_decision" not in blind_task
            assert "confidence" not in blind_task


def test_critical_review_assignment_requires_third_independent_person(service):
    queue = _prepared(service)["blind_review_queue"]
    with pytest.raises(ValueError, match="AT_LEAST_3_INDEPENDENT_REVIEWERS_REQUIRED"):
        build_review_assignments(queue, ["reviewer-a", "reviewer-b"])


def test_reviewer_aliases_do_not_satisfy_independence(service):
    queue = _prepared(service)["blind_review_queue"]
    with pytest.raises(ValueError, match="REVIEWER_IDENTITIES_NOT_INDEPENDENT"):
        build_review_assignments(queue, ["Reviewer-A", " reviewer-a ", "reviewer-c"])

    prepared = _prepared(service)
    reviewed_at = _after_seal(prepared)
    label = _human_label(
        prepared,
        reviews=[
            ReviewObservation(
                reviewer_id="Reviewer-A",
                reviewed_at=reviewed_at,
                disposition=LabelDisposition.VALUE,
                value="Jane Doe",
            ),
            ReviewObservation(
                reviewer_id=" reviewer-a ",
                reviewed_at=reviewed_at,
                disposition=LabelDisposition.VALUE,
                value="Jane Doe",
            ),
        ],
    )
    result = service.score(prepared["sealed_predictions"], [label])
    assert result["label_audit"]["records"][0]["reasons"] == [
        "REVIEWERS_NOT_INDEPENDENT"
    ]


def test_review_assignment_rejects_prediction_leakage(service):
    queue = deepcopy(_prepared(service)["blind_review_queue"])
    queue["tasks"][0]["selected_value"] = "Jane Doe"
    with pytest.raises(ValueError, match="BLIND_TASK_SCHEMA_OR_LEAKAGE_VIOLATION"):
        build_review_assignments(queue, ["reviewer-a", "reviewer-b", "reviewer-c"])


def test_review_assignment_manifest_is_tamper_evident(service):
    queue = _prepared(service)["blind_review_queue"]
    manifest = build_review_assignments(
        queue, ["reviewer-a", "reviewer-b", "reviewer-c"]
    )["review_assignment_manifest"]
    assert verify_review_assignment(manifest)["status"] == "VERIFIED"
    tampered = deepcopy(manifest)
    tampered["tasks"][0]["eligible_adjudicator_ids"] = []
    with pytest.raises(ValueError, match="REVIEW_ASSIGNMENT_SEAL_INVALID"):
        verify_review_assignment(tampered)


def _blind_submission(
    prepared: dict,
    assignments: dict,
    *,
    reviewer_id: str,
    value: str,
    seconds_after_seal: int = 1,
) -> BlindReviewSubmission:
    task = prepared["blind_review_queue"]["tasks"][0]
    manifest = assignments["review_assignment_manifest"]
    return BlindReviewSubmission(
        blind_task_id=task["blind_task_id"],
        prediction_seal_sha256=prepared["sealed_predictions"]["prediction_seal_sha256"],
        review_assignment_seal_sha256=manifest["review_assignment_seal_sha256"],
        reviewer_id=reviewer_id,
        reviewed_at=_after_seal(prepared) + timedelta(seconds=seconds_after_seal),
        disposition=LabelDisposition.VALUE,
        value=value,
    )


def test_review_compiler_emits_score_ready_governed_consensus(service):
    prepared = _prepared(service)
    assignments = build_review_assignments(
        prepared["blind_review_queue"], ["reviewer-a", "reviewer-b", "reviewer-c"]
    )
    manifest = assignments["review_assignment_manifest"]
    reviewers = manifest["tasks"][0]["assigned_reviewer_ids"]
    submissions = [
        _blind_submission(
            prepared,
            assignments,
            reviewer_id=reviewer,
            value="Jane Doe" if index == 0 else " JANE   DOE ",
        )
        for index, reviewer in enumerate(reviewers)
    ]
    compiled = compile_review_submissions(
        prepared["blind_review_queue"], manifest, submissions
    )
    assert compiled["review_progress"]["status"] == "READY_TO_SCORE"
    assert compiled["review_progress"]["completed_labels"] == 1
    labels = [GovernedFieldLabel.model_validate(row) for row in compiled["governed_labels"]]
    scored = service.score(prepared["sealed_predictions"], labels)
    assert scored["scored_metrics"]["eligible_labels"] == 1


def test_review_compiler_creates_blind_adjudication_queue_and_accepts_reserved_judge(service):
    prepared = _prepared(service)
    assignments = build_review_assignments(
        prepared["blind_review_queue"], ["reviewer-a", "reviewer-b", "reviewer-c"]
    )
    manifest = assignments["review_assignment_manifest"]
    task_assignment = manifest["tasks"][0]
    reviews = [
        _blind_submission(
            prepared,
            assignments,
            reviewer_id=reviewer,
            value=value,
        )
        for reviewer, value in zip(
            task_assignment["assigned_reviewer_ids"], ["Jane Doe", "Janet Doe"], strict=True
        )
    ]
    blocked = compile_review_submissions(
        prepared["blind_review_queue"], manifest, reviews
    )
    assert blocked["review_progress"]["status"] == "BLOCKED_PENDING_ADJUDICATION"
    adjudication_task = blocked["adjudication_queue"]["tasks"][0]
    assert "runtime_decision" not in adjudication_task
    assert "evaluation_decision" not in adjudication_task
    adjudication = _blind_submission(
        prepared,
        assignments,
        reviewer_id=task_assignment["eligible_adjudicator_ids"][0],
        value="Janet Doe",
        seconds_after_seal=2,
    )
    compiled = compile_review_submissions(
        prepared["blind_review_queue"], manifest, reviews, [adjudication]
    )
    assert compiled["review_progress"]["status"] == "READY_TO_SCORE"
    assert compiled["governed_labels"][0]["final_value"] == "Janet Doe"


def test_review_compiler_rejects_unassigned_reviewer(service):
    prepared = _prepared(service)
    assignments = build_review_assignments(
        prepared["blind_review_queue"], ["reviewer-a", "reviewer-b", "reviewer-c"]
    )
    manifest = assignments["review_assignment_manifest"]
    reserved = manifest["tasks"][0]["eligible_adjudicator_ids"][0]
    review = _blind_submission(
        prepared, assignments, reviewer_id=reserved, value="Jane Doe"
    )
    with pytest.raises(ValueError, match="REVIEWER_NOT_ASSIGNED_TO_TASK"):
        compile_review_submissions(prepared["blind_review_queue"], manifest, [review])


def test_review_compiler_rejects_premature_adjudication(service):
    prepared = _prepared(service)
    assignments = build_review_assignments(
        prepared["blind_review_queue"], ["reviewer-a", "reviewer-b", "reviewer-c"]
    )
    manifest = assignments["review_assignment_manifest"]
    adjudicator = manifest["tasks"][0]["eligible_adjudicator_ids"][0]
    adjudication = _blind_submission(
        prepared, assignments, reviewer_id=adjudicator, value="Jane Doe"
    )
    with pytest.raises(ValueError, match="ADJUDICATION_WITHOUT_COMPLETED_DISAGREEMENT"):
        compile_review_submissions(
            prepared["blind_review_queue"], manifest, [], [adjudication]
        )
