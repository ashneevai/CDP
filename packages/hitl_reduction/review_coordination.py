from __future__ import annotations

import json
import unicodedata
from hashlib import sha256
from typing import Any

from packages.evidence.normalization import normalize_agreement_value
from packages.hitl_reduction.contracts import (
    BlindReviewSubmission,
    GovernedFieldLabel,
    LabelAuthority,
    ReviewObservation,
)

BLIND_TASK_FIELDS = {
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


def canonical_reviewer_id(value: str) -> str:
    """Return the comparison identity used for independence checks."""
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _validate_queue(queue: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if queue.get("schema_version") != "hitl-reduction-blind-review-v1":
        raise ValueError("UNSUPPORTED_BLIND_REVIEW_QUEUE")
    seal = queue.get("prediction_seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        raise ValueError("PREDICTION_SEAL_REQUIRED")
    try:
        int(seal, 16)
    except ValueError as exc:
        raise ValueError("PREDICTION_SEAL_INVALID") from exc
    tasks = queue.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("BLIND_REVIEW_TASKS_REQUIRED")
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != BLIND_TASK_FIELDS:
            raise ValueError("BLIND_TASK_SCHEMA_OR_LEAKAGE_VIOLATION")
        task_id = task.get("blind_task_id")
        if not isinstance(task_id, str) or len(task_id) != 64:
            raise ValueError("BLIND_TASK_ID_INVALID")
        try:
            int(task_id, 16)
        except ValueError as exc:
            raise ValueError("BLIND_TASK_ID_INVALID") from exc
        if task_id in task_ids:
            raise ValueError("DUPLICATE_BLIND_TASK_ID")
        task_ids.add(task_id)
        required = task.get("required_independent_reviews")
        if required not in {1, 2}:
            raise ValueError("UNSUPPORTED_INDEPENDENT_REVIEW_COUNT")
    return seal, sorted(tasks, key=lambda item: item["blind_task_id"])


def build_review_assignments(
    blind_queue: dict[str, Any], reviewer_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Create isolated prediction-blind reviewer packs and a coordinator manifest."""
    seal, tasks = _validate_queue(blind_queue)
    reviewers = [value.strip() for value in reviewer_ids]
    if any(not value for value in reviewers):
        raise ValueError("REVIEWER_ID_REQUIRED")
    canonical = [canonical_reviewer_id(value) for value in reviewers]
    if len(canonical) != len(set(canonical)):
        raise ValueError("REVIEWER_IDENTITIES_NOT_INDEPENDENT")

    maximum_reviews = max(int(task["required_independent_reviews"]) for task in tasks)
    minimum_people = 3 if maximum_reviews == 2 else 1
    if len(reviewers) < minimum_people:
        raise ValueError(f"AT_LEAST_{minimum_people}_INDEPENDENT_REVIEWERS_REQUIRED")

    assignments: dict[str, list[dict[str, Any]]] = {reviewer: [] for reviewer in reviewers}
    coordinator_tasks: list[dict[str, Any]] = []
    for task in tasks:
        required = int(task["required_independent_reviews"])
        start = int(task["blind_task_id"][:16], 16) % len(reviewers)
        assigned = [reviewers[(start + offset) % len(reviewers)] for offset in range(required)]
        adjudicators = [reviewer for reviewer in reviewers if reviewer not in assigned]
        if required == 2 and not adjudicators:
            raise ValueError("INDEPENDENT_ADJUDICATOR_REQUIRED")
        for reviewer in assigned:
            assignments[reviewer].append(dict(task))
        coordinator_tasks.append(
            {
                "blind_task_id": task["blind_task_id"],
                "assigned_reviewer_ids": assigned,
                "eligible_adjudicator_ids": adjudicators,
            }
        )

    manifest_payload = {
        "schema_version": "hitl-reduction-review-assignment-v1",
        "prediction_seal_sha256": seal,
        "reviewer_count": len(reviewers),
        "task_count": len(tasks),
        "review_assignment_count": sum(len(rows) for rows in assignments.values()),
        "tasks": coordinator_tasks,
    }
    assignment_seal = _digest(manifest_payload)
    outputs: dict[str, dict[str, Any]] = {
        "review_assignment_manifest": {
            **manifest_payload,
            "review_assignment_seal_sha256": assignment_seal,
        }
    }
    for index, reviewer in enumerate(reviewers, 1):
        outputs[f"reviewer_{index:03d}"] = {
            "schema_version": "hitl-reduction-reviewer-pack-v1",
            "prediction_seal_sha256": seal,
            "review_assignment_seal_sha256": assignment_seal,
            "reviewer_id": reviewer,
            "tasks": assignments[reviewer],
        }
    return outputs


def verify_review_assignment(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when a coordinator manifest no longer matches its seal."""
    payload = dict(manifest)
    seal = payload.pop("review_assignment_seal_sha256", None)
    if payload.get("schema_version") != "hitl-reduction-review-assignment-v1":
        raise ValueError("UNSUPPORTED_REVIEW_ASSIGNMENT_MANIFEST")
    if not isinstance(seal, str) or _digest(payload) != seal:
        raise ValueError("REVIEW_ASSIGNMENT_SEAL_INVALID")
    return {
        "schema_version": "hitl-reduction-review-assignment-verification-v1",
        "status": "VERIFIED",
        "prediction_seal_sha256": payload["prediction_seal_sha256"],
        "review_assignment_seal_sha256": seal,
        "reviewer_count": payload["reviewer_count"],
        "task_count": payload["task_count"],
        "review_assignment_count": payload["review_assignment_count"],
    }


def _submission_key(field_name: str, submission: BlindReviewSubmission) -> tuple[str, str]:
    value = normalize_agreement_value(field_name, submission.value)
    return submission.disposition.value, value or ""


def _validate_assignment_binding(
    blind_queue: dict[str, Any], manifest: dict[str, Any]
) -> tuple[str, str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    prediction_seal, tasks = _validate_queue(blind_queue)
    verification = verify_review_assignment(manifest)
    if verification["prediction_seal_sha256"] != prediction_seal:
        raise ValueError("ASSIGNMENT_PREDICTION_SEAL_MISMATCH")
    task_by_id = {task["blind_task_id"]: task for task in tasks}
    assignment_by_id = {task["blind_task_id"]: task for task in manifest["tasks"]}
    if len(assignment_by_id) != len(manifest["tasks"]):
        raise ValueError("DUPLICATE_ASSIGNMENT_TASK_ID")
    if set(task_by_id) != set(assignment_by_id):
        raise ValueError("ASSIGNMENT_TASK_SET_MISMATCH")
    for task_id, assignment in assignment_by_id.items():
        assigned = assignment.get("assigned_reviewer_ids")
        adjudicators = assignment.get("eligible_adjudicator_ids")
        if not isinstance(assigned, list) or not isinstance(adjudicators, list):
            raise ValueError("ASSIGNMENT_REVIEWER_LIST_INVALID")
        if len(assigned) != task_by_id[task_id]["required_independent_reviews"]:
            raise ValueError("ASSIGNMENT_REVIEW_COUNT_MISMATCH")
        identities = [canonical_reviewer_id(value) for value in [*assigned, *adjudicators]]
        if any(not value for value in identities) or len(identities) != len(set(identities)):
            raise ValueError("ASSIGNMENT_IDENTITIES_NOT_INDEPENDENT")
        if len(assigned) == 2 and not adjudicators:
            raise ValueError("ASSIGNMENT_ADJUDICATOR_MISSING")
    return (
        prediction_seal,
        verification["review_assignment_seal_sha256"],
        task_by_id,
        assignment_by_id,
    )


def compile_review_submissions(
    blind_queue: dict[str, Any],
    manifest: dict[str, Any],
    reviews: list[BlindReviewSubmission],
    adjudications: list[BlindReviewSubmission] | None = None,
) -> dict[str, Any]:
    """Compile isolated reviews into governed labels without exposing model decisions."""
    prediction_seal, assignment_seal, tasks, assignments = _validate_assignment_binding(
        blind_queue, manifest
    )
    review_rows: dict[tuple[str, str], BlindReviewSubmission] = {}
    adjudication_rows: dict[str, BlindReviewSubmission] = {}

    for submission in reviews:
        task = tasks.get(submission.blind_task_id)
        if task is None:
            raise ValueError("REVIEW_TASK_NOT_ASSIGNED")
        if submission.prediction_seal_sha256 != prediction_seal:
            raise ValueError("REVIEW_PREDICTION_SEAL_MISMATCH")
        if submission.review_assignment_seal_sha256 != assignment_seal:
            raise ValueError("REVIEW_ASSIGNMENT_SEAL_MISMATCH")
        reviewer = canonical_reviewer_id(submission.reviewer_id)
        assigned = {
            canonical_reviewer_id(value)
            for value in assignments[submission.blind_task_id]["assigned_reviewer_ids"]
        }
        if reviewer not in assigned:
            raise ValueError("REVIEWER_NOT_ASSIGNED_TO_TASK")
        key = submission.blind_task_id, reviewer
        if key in review_rows:
            raise ValueError("DUPLICATE_REVIEW_SUBMISSION")
        review_rows[key] = submission

    for submission in adjudications or []:
        task = tasks.get(submission.blind_task_id)
        if task is None:
            raise ValueError("ADJUDICATION_TASK_NOT_ASSIGNED")
        if submission.prediction_seal_sha256 != prediction_seal:
            raise ValueError("ADJUDICATION_PREDICTION_SEAL_MISMATCH")
        if submission.review_assignment_seal_sha256 != assignment_seal:
            raise ValueError("ADJUDICATION_ASSIGNMENT_SEAL_MISMATCH")
        adjudicator = canonical_reviewer_id(submission.reviewer_id)
        eligible = {
            canonical_reviewer_id(value)
            for value in assignments[submission.blind_task_id]["eligible_adjudicator_ids"]
        }
        if adjudicator not in eligible:
            raise ValueError("ADJUDICATOR_NOT_ELIGIBLE_FOR_TASK")
        if submission.blind_task_id in adjudication_rows:
            raise ValueError("DUPLICATE_ADJUDICATION_SUBMISSION")
        adjudication_rows[submission.blind_task_id] = submission

    labels: list[dict[str, Any]] = []
    adjudication_queue: list[dict[str, Any]] = []
    used_adjudications: set[str] = set()
    pending_review_assignments = 0
    for task_id, task in sorted(tasks.items()):
        assignment = assignments[task_id]
        ordered_reviews = [
            review_rows.get((task_id, canonical_reviewer_id(reviewer)))
            for reviewer in assignment["assigned_reviewer_ids"]
        ]
        completed_reviews = [row for row in ordered_reviews if row is not None]
        if len(completed_reviews) != len(ordered_reviews):
            pending_review_assignments += len(ordered_reviews) - len(completed_reviews)
            continue
        review_keys = {_submission_key(task["field_name"], row) for row in completed_reviews}
        adjudication = None
        if len(review_keys) > 1:
            adjudication = adjudication_rows.get(task_id)
            if adjudication is None:
                adjudication_queue.append(
                    {
                        **task,
                        "review_options": [
                            {
                                "disposition": row.disposition.value,
                                "value": row.value,
                            }
                            for row in completed_reviews
                        ],
                    }
                )
                continue
            used_adjudications.add(task_id)
            final = adjudication
        else:
            if task_id in adjudication_rows:
                raise ValueError("ADJUDICATION_WITHOUT_DISAGREEMENT")
            final = completed_reviews[0]
        label = GovernedFieldLabel(
            field_instance_id=task["field_instance_id"],
            document_id=task["document_id"],
            page_id=task["page_id"],
            page_sha256=task["page_sha256"],
            crop_sha256=task["crop_sha256"],
            blind_task_id=task_id,
            prediction_seal_sha256=prediction_seal,
            authority=LabelAuthority.HUMAN_ADJUDICATED,
            final_disposition=final.disposition,
            final_value=final.value,
            reviews=[
                ReviewObservation(
                    reviewer_id=row.reviewer_id,
                    reviewed_at=row.reviewed_at,
                    disposition=row.disposition,
                    value=row.value,
                )
                for row in completed_reviews
            ],
            adjudication=(
                ReviewObservation(
                    reviewer_id=adjudication.reviewer_id,
                    reviewed_at=adjudication.reviewed_at,
                    disposition=adjudication.disposition,
                    value=adjudication.value,
                )
                if adjudication is not None
                else None
            ),
        )
        labels.append(label.model_dump(mode="json"))

    if set(adjudication_rows) != used_adjudications:
        raise ValueError("ADJUDICATION_WITHOUT_COMPLETED_DISAGREEMENT")

    pending_adjudications = len(adjudication_queue)
    completed = len(labels)
    if pending_review_assignments:
        status = "BLOCKED_PENDING_REVIEWS"
    elif pending_adjudications:
        status = "BLOCKED_PENDING_ADJUDICATION"
    elif completed == len(tasks):
        status = "READY_TO_SCORE"
    else:
        status = "BLOCKED_INCOMPLETE_LABELS"
    return {
        "governed_labels": labels,
        "adjudication_queue": {
            "schema_version": "hitl-reduction-adjudication-queue-v1",
            "prediction_seal_sha256": prediction_seal,
            "review_assignment_seal_sha256": assignment_seal,
            "tasks": adjudication_queue,
        },
        "review_progress": {
            "schema_version": "hitl-reduction-review-progress-v1",
            "status": status,
            "tasks": len(tasks),
            "completed_labels": completed,
            "pending_review_assignments": pending_review_assignments,
            "pending_adjudications": pending_adjudications,
            "submitted_adjudications": len(adjudication_rows),
        },
    }
