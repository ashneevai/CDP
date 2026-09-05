"""Fail-closed governance for blind real-data review and frozen cohorts.

This module carries references and hashes, not page images or clear-text field
values. It has no serving authority and does not import the inference pipeline.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from packages.domain.common import DomainModel

HEX64 = r"^[0-9a-f]{64}$"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class ReviewTarget(StrEnum):
    PAGE = "PAGE"
    DOCUMENT = "DOCUMENT"
    ATTACHMENT = "ATTACHMENT"
    FIELD = "FIELD"


class ReviewAction(StrEnum):
    CONFIRM = "CONFIRM"
    CORRECT = "CORRECT"
    UNKNOWN = "UNKNOWN"
    SPLIT_DOCUMENT = "SPLIT_DOCUMENT"
    MERGE_DOCUMENT = "MERGE_DOCUMENT"


class ReviewEvent(DomainModel):
    """Appendable review event containing identifiers and value hashes only."""

    event_id: str
    target: ReviewTarget
    action: ReviewAction
    archive_id: str
    package_id: str
    source_asset_id: str | None = None
    source_page_id: str | None = None
    document_id: str | None = None
    field_name: str | None = None
    reviewer_id: str
    reviewed_at: datetime
    annotation_version: str
    previous_value_sha256: str | None = Field(default=None, pattern=HEX64)
    new_value_sha256: str | None = Field(default=None, pattern=HEX64)
    correction_reason_code: str | None = None
    prediction_visible: bool = False
    source_region_sha256: str | None = Field(default=None, pattern=HEX64)

    @model_validator(mode="after")
    def enforce_blind_and_provenance(self):
        if self.prediction_visible:
            raise ValueError("initial review must be blind to CDP predictions")
        if self.action == ReviewAction.CORRECT and not self.correction_reason_code:
            raise ValueError("corrections require a reason code")
        if self.target == ReviewTarget.FIELD and not (
            self.field_name and self.source_region_sha256
        ):
            raise ValueError("field review requires field and source-region provenance")
        return self


class AnnotationState(StrEnum):
    VALUE = "VALUE"
    UNREADABLE = "UNREADABLE"
    NOT_PRESENT = "NOT_PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AnnotationAuthority(StrEnum):
    SOURCE_SYSTEM_GROUND_TRUTH = "SOURCE_SYSTEM_GROUND_TRUTH"
    HUMAN_ADJUDICATED = "HUMAN_ADJUDICATED"
    HUMAN_SINGLE_REVIEW = "HUMAN_SINGLE_REVIEW"
    SYNTHETIC = "SYNTHETIC"
    MODEL_GENERATED = "MODEL_GENERATED"
    UNKNOWN = "UNKNOWN"


TRUSTED_AUTHORITIES = {
    AnnotationAuthority.SOURCE_SYSTEM_GROUND_TRUTH,
    AnnotationAuthority.HUMAN_ADJUDICATED,
}


class Annotation(DomainModel):
    annotation_id: str
    package_id: str
    source_page_id: str
    field_name: str
    critical: bool
    reviewer_id: str
    state: AnnotationState
    normalized_value_sha256: str | None = Field(default=None, pattern=HEX64)
    source_region_sha256: str = Field(pattern=HEX64)
    annotation_version: str
    created_at: datetime
    prediction_visible: bool = False

    @model_validator(mode="after")
    def validate_annotation(self):
        if self.prediction_visible:
            raise ValueError("annotation must be collected in blind mode")
        if (self.state == AnnotationState.VALUE) != (self.normalized_value_sha256 is not None):
            raise ValueError("VALUE requires exactly one normalized value hash")
        return self

    @property
    def target_key(self) -> tuple[str, str, str]:
        return self.package_id, self.source_page_id, self.field_name

    @property
    def comparison_key(self) -> tuple[str, str | None]:
        return self.state.value, self.normalized_value_sha256


class CriticalAnnotationDecision(DomainModel):
    status: str = Field(pattern=r"^(AGREED|ADJUDICATION_REQUIRED|FINAL)$")
    annotation_a_id: str
    annotation_b_id: str
    agreement: bool
    final_state: AnnotationState | None = None
    final_value_sha256: str | None = Field(default=None, pattern=HEX64)
    authority: AnnotationAuthority | None = None
    adjudicator_id: str | None = None
    adjudicated_at: datetime | None = None


def compare_critical_annotations(a: Annotation, b: Annotation) -> CriticalAnnotationDecision:
    if not (a.critical and b.critical):
        raise ValueError("dual annotation is required only through this critical-field path")
    if a.target_key != b.target_key:
        raise ValueError("annotations do not address the same field observation")
    if a.reviewer_id == b.reviewer_id:
        raise ValueError("critical annotations require two distinct reviewers")
    agreement = a.comparison_key == b.comparison_key
    return CriticalAnnotationDecision(
        status="AGREED" if agreement else "ADJUDICATION_REQUIRED",
        annotation_a_id=a.annotation_id,
        annotation_b_id=b.annotation_id,
        agreement=agreement,
    )


def adjudicate_critical(
    decision: CriticalAnnotationDecision,
    a: Annotation,
    b: Annotation,
    *,
    adjudicator_id: str,
    final_state: AnnotationState,
    final_value_sha256: str | None = None,
) -> CriticalAnnotationDecision:
    if decision.status != "ADJUDICATION_REQUIRED":
        raise ValueError("only disagreements may enter adjudication")
    if adjudicator_id in {a.reviewer_id, b.reviewer_id}:
        raise ValueError("adjudicator must be independent of both annotators")
    if (final_state == AnnotationState.VALUE) != (final_value_sha256 is not None):
        raise ValueError("adjudicated VALUE requires exactly one value hash")
    return CriticalAnnotationDecision(
        status="FINAL",
        annotation_a_id=a.annotation_id,
        annotation_b_id=b.annotation_id,
        agreement=False,
        final_state=final_state,
        final_value_sha256=final_value_sha256,
        authority=AnnotationAuthority.HUMAN_ADJUDICATED,
        adjudicator_id=adjudicator_id,
        adjudicated_at=datetime.now(UTC),
    )


def trusted_for_release(authority: AnnotationAuthority, *, finalized: bool) -> bool:
    return finalized and authority in TRUSTED_AUTHORITIES


def finalize_agreement(
    decision: CriticalAnnotationDecision, a: Annotation, b: Annotation
) -> CriticalAnnotationDecision:
    """Finalize matching independent annotations without selecting one reviewer."""
    if decision.status != "AGREED" or not decision.agreement:
        raise ValueError("only agreed dual annotations can bypass adjudication")
    if a.comparison_key != b.comparison_key:
        raise ValueError("annotations changed after comparison")
    return CriticalAnnotationDecision(
        status="FINAL",
        annotation_a_id=a.annotation_id,
        annotation_b_id=b.annotation_id,
        agreement=True,
        final_state=a.state,
        final_value_sha256=a.normalized_value_sha256,
        authority=AnnotationAuthority.HUMAN_ADJUDICATED,
        adjudicated_at=datetime.now(UTC),
    )


class CohortPage(DomainModel):
    package_id: str
    document_id: str
    source_page_id: str
    page_class: str
    quality_band: str
    package_complexity: str
    source_page_sha256: str = Field(pattern=HEX64)
    label_manifest_sha256: str | None = Field(default=None, pattern=HEX64)


def deterministic_package_sample(
    pages: Iterable[CohortPage],
    target_pages_by_class: dict[str, int],
    *,
    seed: str,
) -> tuple[str, ...]:
    """Select complete packages, ordered reproducibly, without page cherry-picking."""
    grouped: dict[str, list[CohortPage]] = {}
    for page in pages:
        grouped.setdefault(page.package_id, []).append(page)
    order = sorted(grouped, key=lambda package: (_digest([seed, package]), package))
    selected: list[str] = []
    counts: Counter[str] = Counter()
    targets = {name: max(0, count) for name, count in target_pages_by_class.items()}
    for package in order:
        contribution = Counter(page.page_class for page in grouped[package])
        if any(counts[name] < target and contribution[name] for name, target in targets.items()):
            selected.append(package)
            counts.update(contribution)
        if all(counts[name] >= target for name, target in targets.items()):
            break
    return tuple(selected)


def package_level_split(
    package_ids: Iterable[str], *, seed: str, development_fraction: float = 0.7
) -> dict[str, str]:
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    packages = sorted(set(package_ids), key=lambda item: (_digest([seed, item]), item))
    if len(packages) < 2:
        raise ValueError("at least two packages are required for a leakage-safe split")
    development_count = min(len(packages) - 1, max(1, round(len(packages) * development_fraction)))
    return {
        package: "DEVELOPMENT" if index < development_count else "HOLDOUT"
        for index, package in enumerate(packages)
    }


def assert_no_package_leakage(assignments: Iterable[tuple[str, str]]) -> None:
    """Reject manifests that place a package in multiple cohorts."""
    observed: dict[str, str] = {}
    for package_id, split in assignments:
        if split not in {"DEVELOPMENT", "HOLDOUT"}:
            raise ValueError(f"unknown split assignment: {split}")
        previous = observed.setdefault(package_id, split)
        if previous != split:
            raise ValueError(f"package leakage detected: {package_id}")


class FrozenCohort(DomainModel):
    schema_version: str = "real-eval-cohort-v1"
    archive_sha256: str = Field(pattern=HEX64)
    split_seed: str
    assignments: dict[str, str]
    package_manifest_sha256: str = Field(pattern=HEX64)
    page_manifest_sha256: str = Field(pattern=HEX64)
    label_manifest_sha256: str = Field(pattern=HEX64)
    frozen_at: datetime
    seal_sha256: str = Field(pattern=HEX64)


def freeze_cohort(
    *,
    archive_sha256: str,
    split_seed: str,
    assignments: dict[str, str],
    pages: Iterable[CohortPage],
) -> FrozenCohort:
    if set(assignments.values()) - {"DEVELOPMENT", "HOLDOUT"}:
        raise ValueError("unknown split assignment")
    page_list = sorted(
        (page.model_dump(mode="json") for page in pages), key=lambda x: x["source_page_id"]
    )
    page_packages = {page["package_id"] for page in page_list}
    if page_packages != set(assignments):
        raise ValueError("every and only selected page packages must be assigned")
    if any(page["label_manifest_sha256"] is None for page in page_list):
        raise ValueError("trusted label manifests must exist before cohort freeze")
    payload = {
        "schema_version": "real-eval-cohort-v1",
        "archive_sha256": archive_sha256,
        "split_seed": split_seed,
        "assignments": dict(sorted(assignments.items())),
        "package_manifest_sha256": _digest(sorted(assignments)),
        "page_manifest_sha256": _digest(page_list),
        "label_manifest_sha256": _digest(
            sorted((page["source_page_id"], page["label_manifest_sha256"]) for page in page_list)
        ),
        "frozen_at": datetime.now(UTC),
    }
    draft = FrozenCohort(**payload, seal_sha256="0" * 64)
    seal = _digest(draft.model_dump(mode="json", exclude={"seal_sha256"}))
    return draft.model_copy(update={"seal_sha256": seal})


def verify_frozen_cohort(cohort: FrozenCohort) -> bool:
    payload = cohort.model_dump(mode="json", exclude={"seal_sha256"})
    return _digest(payload) == cohort.seal_sha256


class HoldoutExecution(DomainModel):
    execution_id: str
    cohort_seal_sha256: str = Field(pattern=HEX64)
    code_commit_sha256: str = Field(pattern=HEX64)
    purpose: str = Field(pattern=r"^(MILESTONE_GATE|FINAL_GATE)$")
    executed_at: datetime
    result_manifest_sha256: str = Field(pattern=HEX64)
    used_for_tuning: bool = False


class HoldoutLedger(DomainModel):
    cohort_seal_sha256: str = Field(pattern=HEX64)
    executions: tuple[HoldoutExecution, ...] = ()

    def record(self, execution: HoldoutExecution) -> HoldoutLedger:
        if execution.cohort_seal_sha256 != self.cohort_seal_sha256:
            raise ValueError("execution targets a different frozen cohort")
        if execution.used_for_tuning:
            raise ValueError("holdout results must never be used for iterative tuning")
        if any(item.execution_id == execution.execution_id for item in self.executions):
            raise ValueError("holdout execution IDs are immutable and unique")
        return self.model_copy(update={"executions": (*self.executions, execution)})
