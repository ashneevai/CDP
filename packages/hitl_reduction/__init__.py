"""Governed, leakage-resistant HITL reduction evaluation."""

from packages.hitl_reduction.contracts import (
    BlindReviewSubmission,
    ClaimRuntimeRecord,
    FieldRuntimeRecord,
    GovernedFieldLabel,
    HITLReductionInput,
    LabelAuthority,
    LabelDisposition,
    OperationalEvidence,
    ReviewObservation,
)
from packages.hitl_reduction.review_coordination import (
    build_review_assignments,
    compile_review_submissions,
    verify_review_assignment,
)
from packages.hitl_reduction.service import HITLReductionService

__all__ = [
    "BlindReviewSubmission",
    "ClaimRuntimeRecord",
    "FieldRuntimeRecord",
    "GovernedFieldLabel",
    "HITLReductionInput",
    "HITLReductionService",
    "LabelAuthority",
    "LabelDisposition",
    "OperationalEvidence",
    "ReviewObservation",
    "build_review_assignments",
    "compile_review_submissions",
    "verify_review_assignment",
]
