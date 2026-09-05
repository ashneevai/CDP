from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from typing import Any

from packages.claim_decision import ClaimDecisionContext
from packages.evidence.models import EvidenceClass
from packages.evidence.normalization import normalize_agreement_value
from packages.evidence_decision import EvidenceDecisionService, FieldDisposition, NextAction
from packages.hitl_reduction.contracts import (
    ClaimRuntimeRecord,
    GovernedFieldLabel,
    HITLReductionInput,
    LabelAuthority,
    LabelDisposition,
    OperationalEvidence,
    ReviewObservation,
)
from packages.hitl_reduction.review_coordination import canonical_reviewer_id
from packages.production_readiness_gate import (
    ProductionReadinessGate,
    ReadinessDecision,
    ReadinessEvidence,
)
from packages.route_registry import RouteLifecycle
from packages.route_registry.promotion import RoutePromotionEvidence, RoutePromotionGate
from packages.runtime_profile.decision_factory import (
    DecisionServiceBundle,
    DecisionServiceFactory,
)

ACCEPTED_DISPOSITIONS = {
    FieldDisposition.AUTO_ACCEPTED.value,
    FieldDisposition.REFERENCE_CONFIRMED.value,
    FieldDisposition.HUMAN_CONFIRMED.value,
}
TERMINAL_HUMAN_DISPOSITIONS = {
    FieldDisposition.HUMAN_REVIEW_REQUIRED.value,
    FieldDisposition.INSUFFICIENT_EVIDENCE.value,
}
INELIGIBLE_TRUTH = {LabelDisposition.UNREADABLE, LabelDisposition.NOT_APPLICABLE}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


def _evaluation_service(bundle: DecisionServiceBundle) -> EvidenceDecisionService:
    identity = {
        key: value
        for key, value in bundle.profile.decision_identity().items()
        if key != "claim_policy_hash"
    }
    identity.update(
        evidence_policy_version=bundle.evidence_decision.evidence_policy.version,
        route_registry_version=bundle.route_registry.version,
        field_policy_version=bundle.field_policy.version,
        route_mode="evaluation",
    )
    return EvidenceDecisionService(
        reconciler=bundle.evidence_decision.reconciler,
        evidence_policy=bundle.evidence_decision.evidence_policy,
        field_policy=bundle.field_policy,
        route_mode="evaluation",
        route_registry=bundle.route_registry,
        configuration_identity=identity,
    )


def _claim_context(claim: ClaimRuntimeRecord, field_decisions: list) -> ClaimDecisionContext:
    return ClaimDecisionContext(
        claim_id=claim.claim_id,
        document_family=claim.document_family,
        field_decisions=field_decisions,
        claim_evidence=claim.claim_evidence,
        contradictions=claim.contradictions,
        document_integrity_valid=claim.document_integrity_valid,
        template_integrity_valid=claim.template_integrity_valid,
        registration_integrity_valid=claim.registration_integrity_valid,
        process_integrity_valid=claim.process_integrity_valid,
        structural_consistency_valid=claim.structural_consistency_valid,
        dependent_field_groups=claim.dependent_field_groups,
        enforce_configured_required_fields=claim.enforce_configured_required_fields,
    )


def _operational_metrics(claims: list[dict], decision_key: str) -> dict[str, Any]:
    fields = [field for claim in claims for field in claim["fields"]]
    total_fields = len(fields)
    total_claims = len(claims)
    accepted = sum(field[decision_key]["disposition"] in ACCEPTED_DISPOSITIONS for field in fields)
    terminal_hitl = sum(
        field[decision_key]["disposition"] in TERMINAL_HUMAN_DISPOSITIONS
        or field[decision_key]["next_action"] == NextAction.HUMAN_REVIEW.value
        for field in fields
    )
    machine_recovery = sum(
        field[decision_key]["disposition"] not in ACCEPTED_DISPOSITIONS
        and field[decision_key]["next_action"]
        not in {NextAction.HUMAN_REVIEW.value, NextAction.NONE.value}
        for field in fields
    )
    claim_key = (
        "runtime_claim_decision"
        if decision_key == "runtime_decision"
        else ("evaluation_claim_decision")
    )
    stp_count = sum(claim[claim_key]["stp_eligible"] for claim in claims)
    segments: dict[str, list[bool]] = defaultdict(list)
    for claim in claims:
        segment = f"{claim['source_segment']}|{claim['quality_segment']}"
        segments[segment].append(bool(claim[claim_key]["stp_eligible"]))
    segment_metrics = {
        segment: {
            "claims": len(values),
            "claim_hitl": 1 - sum(values) / len(values),
        }
        for segment, values in sorted(segments.items())
    }
    return {
        "fields": total_fields,
        "claims": total_claims,
        "safe_field_coverage": accepted / total_fields if total_fields else None,
        "field_hitl": terminal_hitl / total_fields if total_fields else None,
        "field_hitl_count": terminal_hitl,
        "machine_recovery_pending": machine_recovery / total_fields if total_fields else None,
        "machine_recovery_pending_count": machine_recovery,
        "unresolved_fields": total_fields - accepted,
        "claim_stp": stp_count / total_claims if total_claims else None,
        "claim_hitl": 1 - stp_count / total_claims if total_claims else None,
        "claim_hitl_count": total_claims - stp_count,
        "segments": segment_metrics,
    }


def _blocker_pareto(claims: list[dict]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for claim in claims:
        blockers = claim["runtime_claim_decision"]["blocking_unresolved_fields"]
        fields_by_name = {field["field_name"]: field for field in claim["fields"]}
        for blocker in blockers:
            field = fields_by_name.get(blocker)
            if field is None:
                key = (
                    claim["document_family"],
                    blocker,
                    claim["source_segment"],
                    claim["quality_segment"],
                    "MISSING_DECISION",
                    "HUMAN_REVIEW",
                    "REQUIRED_FIELD_DECISION_MISSING",
                )
            else:
                decision = field["runtime_decision"]
                key = (
                    claim["document_family"],
                    blocker,
                    field["source_segment"],
                    field["quality_segment"],
                    decision["disposition"],
                    decision["next_action"],
                    (decision["reason_codes"] or ["UNSPECIFIED"])[0],
                )
            row = grouped.setdefault(
                key,
                {
                    "document_family": key[0],
                    "field_name": key[1],
                    "source_segment": key[2],
                    "quality_segment": key[3],
                    "disposition": key[4],
                    "next_action": key[5],
                    "primary_reason": key[6],
                    "claims_affected": 0,
                    "single_blocker_claim_unlocks": 0,
                    "claim_ids": [],
                },
            )
            row["claims_affected"] += 1
            row["claim_ids"].append(claim["claim_id"])
            if len(blockers) == 1:
                row["single_blocker_claim_unlocks"] += 1
    rows = sorted(
        grouped.values(),
        key=lambda row: (
            -row["single_blocker_claim_unlocks"],
            -row["claims_affected"],
            row["document_family"],
            row["field_name"],
        ),
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["claim_ids"] = sorted(row["claim_ids"])
    return {"rows": rows, "total_blocker_groups": len(rows)}


def _blind_task(seal: str, field: dict) -> str:
    return _digest(
        {
            "prediction_seal_sha256": seal,
            "field_instance_id": field["field_instance_id"],
            "page_sha256": field["page_sha256"],
            "crop_sha256": field["crop_sha256"],
        }
    )


def _review_key(field_name: str, review: ReviewObservation) -> tuple[str, str]:
    value = normalize_agreement_value(field_name, review.value)
    return review.disposition.value, value or ""


def _final_key(field_name: str, label: GovernedFieldLabel) -> tuple[str, str]:
    value = normalize_agreement_value(field_name, label.final_value)
    return label.final_disposition.value, value or ""


def _validate_label(
    label: GovernedFieldLabel,
    field: dict,
    seal: str,
    sealed_at: datetime,
) -> list[str]:
    reasons: list[str] = []
    expected = {
        "document_id": field["document_id"],
        "page_id": field["page_id"],
        "page_sha256": field["page_sha256"],
        "crop_sha256": field["crop_sha256"],
        "prediction_seal_sha256": seal,
        "blind_task_id": _blind_task(seal, field),
    }
    for attribute, value in expected.items():
        if getattr(label, attribute) != value:
            raise ValueError(f"LABEL_LINEAGE_MISMATCH:{label.field_instance_id}:{attribute}")
    if label.final_disposition in INELIGIBLE_TRUTH:
        reasons.append("FINAL_LABEL_NOT_SCORABLE")
    if label.derived_from_cdp:
        reasons.append("CIRCULAR_LABEL_DERIVED_FROM_CDP")
    if label.authority is LabelAuthority.SOURCE_SYSTEM_GROUND_TRUTH:
        required = (
            label.source_system,
            label.source_version,
            label.source_snapshot_sha256,
            label.source_record_id,
        )
        if not all(required):
            reasons.append("SOURCE_PROVENANCE_INCOMPLETE")
        if not label.source_independent or not label.source_non_circular:
            reasons.append("SOURCE_NOT_INDEPENDENT_AND_NON_CIRCULAR")
    else:
        reviewer_ids = [review.reviewer_id for review in label.reviews]
        canonical_reviewer_ids = [canonical_reviewer_id(value) for value in reviewer_ids]
        required_reviews = 2 if field["criticality"] in {"C2", "C3"} else 1
        if len(reviewer_ids) < required_reviews:
            reasons.append("INDEPENDENT_REVIEWS_INCOMPLETE")
        if len(canonical_reviewer_ids) != len(set(canonical_reviewer_ids)):
            reasons.append("REVIEWERS_NOT_INDEPENDENT")
        if any(review.reviewed_at <= sealed_at for review in label.reviews):
            reasons.append("REVIEW_PRECEDES_PREDICTION_SEAL")
        review_keys = {
            _review_key(field["field_name"], review) for review in label.reviews
        }
        if len(review_keys) > 1:
            adjudication = label.adjudication
            if adjudication is None:
                reasons.append("DISAGREEMENT_REQUIRES_ADJUDICATION")
            elif canonical_reviewer_id(adjudication.reviewer_id) in set(
                canonical_reviewer_ids
            ):
                reasons.append("ADJUDICATOR_NOT_INDEPENDENT")
            elif adjudication.reviewed_at <= sealed_at:
                reasons.append("ADJUDICATION_PRECEDES_PREDICTION_SEAL")
            elif adjudication.reviewed_at <= max(review.reviewed_at for review in label.reviews):
                reasons.append("ADJUDICATION_PRECEDES_REVIEWS")
            elif _review_key(field["field_name"], adjudication) != _final_key(
                field["field_name"], label
            ):
                reasons.append("FINAL_LABEL_DOES_NOT_MATCH_ADJUDICATION")
        elif review_keys and next(iter(review_keys)) != _final_key(field["field_name"], label):
            reasons.append("FINAL_LABEL_DOES_NOT_MATCH_REVIEW_CONSENSUS")
    return reasons


def _truth_value(label: GovernedFieldLabel) -> str | None:
    return label.final_value if label.final_disposition is LabelDisposition.VALUE else None


def _scored_metrics(
    fields: list[dict],
    eligible_labels: dict[str, GovernedFieldLabel],
    decision_key: str,
) -> dict[str, Any]:
    scored = 0
    correct = 0
    critical_scored = 0
    critical_correct = 0
    accepted = 0
    accepted_correct = 0
    accepted_critical = 0
    accepted_critical_correct = 0
    false_accepts = 0
    critical_false_accepts = 0
    for field in fields:
        label = eligible_labels.get(field["field_instance_id"])
        if label is None:
            continue
        decision = field[decision_key]
        truth = normalize_agreement_value(field["field_name"], _truth_value(label))
        prediction = normalize_agreement_value(field["field_name"], decision["selected_value"])
        is_correct = prediction == truth
        is_critical = field["criticality"] in {"C2", "C3"}
        is_accepted = decision["disposition"] in ACCEPTED_DISPOSITIONS
        scored += 1
        correct += is_correct
        if is_critical:
            critical_scored += 1
            critical_correct += is_correct
        if is_accepted:
            accepted += 1
            accepted_correct += is_correct
            false_accepts += not is_correct
            if is_critical:
                accepted_critical += 1
                accepted_critical_correct += is_correct
                critical_false_accepts += not is_correct
    return {
        "scored_fields": scored,
        "raw_accuracy": correct / scored if scored else None,
        "critical_scored_fields": critical_scored,
        "critical_accuracy": (critical_correct / critical_scored if critical_scored else None),
        "accepted_fields": accepted,
        "accepted_precision": accepted_correct / accepted if accepted else None,
        "false_accept_count": false_accepts,
        "false_accept_rate": false_accepts / accepted if accepted else None,
        "accepted_critical_fields": accepted_critical,
        "critical_accepted_precision": (
            accepted_critical_correct / accepted_critical if accepted_critical else None
        ),
        "critical_false_accept_count": critical_false_accepts,
    }


class HITLReductionService:
    """Two-phase HITL evaluation that never exposes truth to decision execution."""

    def __init__(
        self,
        bundle: DecisionServiceBundle | None = None,
        readiness_gate: ProductionReadinessGate | None = None,
        route_gate: RoutePromotionGate | None = None,
    ) -> None:
        self.bundle = bundle or DecisionServiceFactory.from_profile()
        self.evaluation_decision = _evaluation_service(self.bundle)
        self.readiness_gate = readiness_gate or ProductionReadinessGate.load()
        self.route_gate = route_gate or RoutePromotionGate.load()

    def prepare(self, cohort: HITLReductionInput) -> dict[str, Any]:
        sealed_at = _utcnow()
        claims: list[dict] = []
        for claim_record in cohort.claims:
            runtime_decisions = [
                self.bundle.evidence_decision.decide(field.decision_context)
                for field in claim_record.fields
            ]
            evaluation_decisions = [
                self.evaluation_decision.decide(field.decision_context)
                for field in claim_record.fields
            ]
            runtime_claim = self.bundle.claim_decision.decide(
                _claim_context(claim_record, runtime_decisions)
            )
            evaluation_claim = self.bundle.claim_decision.decide(
                _claim_context(claim_record, evaluation_decisions)
            )
            fields = []
            for record, runtime, evaluation in zip(
                claim_record.fields, runtime_decisions, evaluation_decisions, strict=True
            ):
                fields.append(
                    {
                        "field_instance_id": record.field_instance_id,
                        "field_name": record.decision_context.field_name,
                        "document_id": record.document_id,
                        "page_id": record.page_id,
                        "page_sha256": record.page_sha256,
                        "crop_sha256": record.crop_sha256,
                        "crop_reference": record.crop_reference,
                        "source_segment": record.source_segment,
                        "quality_segment": record.quality_segment,
                        "criticality": record.decision_context.criticality.value,
                        "latency_ms": record.latency_ms,
                        "cost_spent_usd": record.decision_context.cost_spent_usd,
                        "runtime_decision": runtime.model_dump(mode="json"),
                        "evaluation_decision": evaluation.model_dump(mode="json"),
                    }
                )
            claims.append(
                {
                    "claim_id": claim_record.claim_id,
                    "document_family": claim_record.document_family,
                    "source_segment": claim_record.source_segment,
                    "quality_segment": claim_record.quality_segment,
                    "fields": fields,
                    "runtime_claim_decision": runtime_claim.model_dump(mode="json"),
                    "evaluation_claim_decision": evaluation_claim.model_dump(mode="json"),
                }
            )
        predictions = {
            "schema_version": "hitl-reduction-sealed-predictions-v1",
            "sealed_at": sealed_at.isoformat(),
            "cohort_id": cohort.cohort_id,
            "input_manifest_sha256": cohort.input_manifest_sha256,
            "holdout_frozen": cohort.holdout_frozen,
            "holdout_independent": cohort.holdout_independent,
            "runtime_profile": self.bundle.profile.model_dump(mode="json"),
            "operational_evidence": cohort.operational_evidence.model_dump(mode="json"),
            "claims": claims,
        }
        seal = _digest(predictions)
        predictions["prediction_seal_sha256"] = seal
        queue = []
        for prepared_claim in claims:
            for field in prepared_claim["fields"]:
                queue.append(
                    {
                        "blind_task_id": _blind_task(seal, field),
                        "field_instance_id": field["field_instance_id"],
                        "field_name": field["field_name"],
                        "document_id": field["document_id"],
                        "page_id": field["page_id"],
                        "page_sha256": field["page_sha256"],
                        "crop_sha256": field["crop_sha256"],
                        "crop_reference": field["crop_reference"],
                        "criticality": field["criticality"],
                        "required_independent_reviews": (
                            2 if field["criticality"] in {"C2", "C3"} else 1
                        ),
                        "adjudication_required_on_disagreement": True,
                    }
                )
        queue.sort(key=lambda item: _digest({"seal": seal, "task": item["blind_task_id"]}))
        return {
            "sealed_predictions": predictions,
            "blind_review_queue": {
                "schema_version": "hitl-reduction-blind-review-v1",
                "prediction_seal_sha256": seal,
                "tasks": queue,
            },
            "blocker_pareto": {
                "schema_version": "hitl-reduction-blocker-pareto-v1",
                "prediction_seal_sha256": seal,
                **_blocker_pareto(claims),
            },
            "current_metrics": {
                "schema_version": "hitl-reduction-current-metrics-v1",
                "prediction_seal_sha256": seal,
                "status": "AWAITING_TRUSTED_LABELS",
                "runtime": _operational_metrics(claims, "runtime_decision"),
                "evaluation_shadow": _operational_metrics(claims, "evaluation_decision"),
                "accuracy": None,
            },
        }

    def score(
        self,
        sealed_predictions: dict[str, Any],
        labels: Iterable[GovernedFieldLabel],
    ) -> dict[str, Any]:
        payload = deepcopy(sealed_predictions)
        seal = payload.pop("prediction_seal_sha256", None)
        if not isinstance(seal, str) or _digest(payload) != seal:
            raise ValueError("PREDICTION_SEAL_INVALID")
        sealed_at = datetime.fromisoformat(str(payload["sealed_at"]))
        claims = payload["claims"]
        fields = [field for claim in claims for field in claim["fields"]]
        by_field = {field["field_instance_id"]: field for field in fields}
        eligible: dict[str, GovernedFieldLabel] = {}
        audit = []
        seen: set[str] = set()
        for label in labels:
            if label.field_instance_id in seen:
                raise ValueError(f"DUPLICATE_LABEL:{label.field_instance_id}")
            seen.add(label.field_instance_id)
            field = by_field.get(label.field_instance_id)
            if field is None:
                raise ValueError(f"LABEL_FIELD_NOT_IN_SEALED_COHORT:{label.field_instance_id}")
            reasons = _validate_label(label, field, seal, sealed_at)
            if not reasons:
                eligible[label.field_instance_id] = label
            audit.append(
                {
                    "field_instance_id": label.field_instance_id,
                    "eligible": not reasons,
                    "reasons": reasons,
                }
            )
        runtime_scored = _scored_metrics(fields, eligible, "runtime_decision")
        evaluation_scored = _scored_metrics(fields, eligible, "evaluation_decision")
        runtime_operational = _operational_metrics(claims, "runtime_decision")
        evaluation_operational = _operational_metrics(claims, "evaluation_decision")
        coverage_complete = len(eligible) == len(fields)
        label_coverage = len(eligible) / len(fields) if fields else 0
        readiness = self._readiness(
            payload,
            fields,
            eligible,
            runtime_scored,
            runtime_operational,
            coverage_complete,
        )
        status = "BLOCKED_HUMAN_LABELS"
        if coverage_complete:
            if readiness.decision is ReadinessDecision.REJECT:
                status = "REJECTED_SAFETY"
            elif readiness.decision is ReadinessDecision.PROMOTE_TO_PRODUCTION:
                status = "READY_FOR_PRODUCTION"
            elif readiness.decision is ReadinessDecision.PROMOTE_TO_SHADOW:
                status = "READY_FOR_SHADOW"
            else:
                status = "BLOCKED_GATES"
        return {
            "scored_metrics": {
                "schema_version": "hitl-reduction-scored-metrics-v1",
                "prediction_seal_sha256": seal,
                "status": status,
                "label_coverage": label_coverage,
                "eligible_labels": len(eligible),
                "required_labels": len(fields),
                "runtime": {**runtime_operational, **runtime_scored},
                "evaluation_shadow": {
                    **evaluation_operational,
                    **evaluation_scored,
                },
            },
            "qualification": readiness.model_dump(mode="json"),
            "route_promotion_candidates": self._route_promotions(
                payload, fields, eligible, coverage_complete
            ),
            "label_audit": {
                "schema_version": "hitl-reduction-label-audit-v1",
                "prediction_seal_sha256": seal,
                "records": sorted(audit, key=lambda item: item["field_instance_id"]),
            },
        }

    def _readiness(
        self,
        payload: dict,
        fields: list[dict],
        eligible: dict[str, GovernedFieldLabel],
        scored: dict,
        operational_metrics: dict,
        coverage_complete: bool,
    ):
        operational = OperationalEvidence.model_validate(payload["operational_evidence"])
        documents = {
            field["document_id"] for field in fields if field["field_instance_id"] in eligible
        }
        segments = operational_metrics["segments"].values()
        maximum_segment_hitl = max((segment["claim_hitl"] for segment in segments), default=None)
        observed_p95 = _percentile_95([field["latency_ms"] for field in fields])
        p95_latency = operational.p95_latency_ms
        if p95_latency is None:
            p95_latency = observed_p95
        truth = scored if coverage_complete else defaultdict(lambda: None)
        evidence = ReadinessEvidence(
            holdout_frozen=bool(payload["holdout_frozen"] and coverage_complete),
            holdout_independent=bool(payload["holdout_independent"] and coverage_complete),
            holdout_documents=len(documents),
            holdout_fields=len(eligible),
            full_suite_passed=operational.full_suite_passed,
            overall_raw_accuracy=truth["raw_accuracy"],
            critical_accuracy=truth["critical_accuracy"],
            total_false_accept_rate=truth["false_accept_rate"],
            critical_false_accept_count=truth["critical_false_accept_count"],
            safe_field_coverage=operational_metrics["safe_field_coverage"],
            accepted_precision=truth["accepted_precision"],
            claim_stp=operational_metrics["claim_stp"],
            claim_hitl=operational_metrics["claim_hitl"],
            claim_hitl_count=operational_metrics["claim_hitl_count"],
            accepted_critical_field_decisions=(
                scored["accepted_critical_fields"] if coverage_complete else 0
            ),
            critical_accepted_precision=truth["critical_accepted_precision"],
            wrong_crop_recall=operational.wrong_crop_recall,
            maximum_segment_claim_hitl=maximum_segment_hitl,
            p95_latency_ms=p95_latency,
            cost_per_document_usd=operational.cost_per_document_usd,
            runtime_parity_passed=operational.runtime_parity_passed,
            route_governance_passed=operational.route_governance_passed,
            security_passed=operational.security_passed,
            database_and_events_passed=operational.database_and_events_passed,
            load_and_keda_passed=operational.load_and_keda_passed,
            shadow_validation_passed=operational.shadow_validation_passed,
            failure_injection_passed=operational.failure_injection_passed,
            release_commit_sha=operational.release_commit_sha,
            evidence_bundle_sha256=operational.evidence_bundle_sha256,
            approvals=operational.approvals,
        )
        return self.readiness_gate.evaluate(evidence)

    def _route_promotions(
        self,
        payload: dict,
        fields: list[dict],
        eligible: dict[str, GovernedFieldLabel],
        coverage_complete: bool,
    ) -> dict[str, Any]:
        operational = OperationalEvidence.model_validate(payload["operational_evidence"])
        rows = []
        for route in self.bundle.route_registry.routes:
            if route.status not in {RouteLifecycle.EVALUATION_ONLY, RouteLifecycle.SHADOW}:
                continue
            route_fields = [
                field
                for field in fields
                if (field["evaluation_decision"].get("evidence_bundle") or {}).get("route_id")
                == route.route_id
                and field["field_instance_id"] in eligible
            ]
            correct = 0
            agreements = 0
            correct_agreements = 0
            critical_false_agreements = 0
            for field in route_fields:
                label = eligible[field["field_instance_id"]]
                truth = normalize_agreement_value(field["field_name"], _truth_value(label))
                decision = field["evaluation_decision"]
                prediction = normalize_agreement_value(
                    field["field_name"], decision["selected_value"]
                )
                correct += prediction == truth
                independent_values = {
                    normalize_agreement_value(field["field_name"], item.get("value"))
                    for item in (decision.get("evidence_bundle") or {}).get("evidence_items", [])
                    if item.get("evidence_class") == EvidenceClass.E2.value
                    and item.get("independent")
                }
                agreement = bool(prediction and prediction in independent_values)
                agreements += agreement
                correct_agreements += agreement and prediction == truth
                critical_false_agreements += bool(
                    agreement and prediction != truth and field["criticality"] in {"C2", "C3"}
                )
            evidence = RoutePromotionEvidence(
                route_id=route.route_id,
                current_status=route.status,
                independent_holdout_frozen=bool(
                    payload["holdout_frozen"]
                    and payload["holdout_independent"]
                    and coverage_complete
                ),
                holdout_samples=len(route_fields),
                holdout_accuracy=correct / len(route_fields) if route_fields else None,
                agreement_precision=(correct_agreements / agreements if agreements else None),
                critical_false_agreements=(critical_false_agreements if agreements else None),
                mean_latency_ms=(
                    sum(field["latency_ms"] for field in route_fields) / len(route_fields)
                    if route_fields
                    else None
                ),
                cost_per_call_usd=operational.route_cost_per_call_usd.get(route.route_id),
                runtime_shadow_samples=operational.route_shadow_samples.get(route.route_id, 0),
                operational_reliability=operational.route_operational_reliability.get(
                    route.route_id
                ),
            )
            result = self.route_gate.evaluate(evidence)
            rows.append(
                {
                    "route_id": route.route_id,
                    "current_status": route.status.value,
                    "evidence": evidence.model_dump(mode="json"),
                    "recommendation": result.model_dump(mode="json"),
                    "configuration_changed": False,
                }
            )
        return {
            "schema_version": "hitl-reduction-route-candidates-v1",
            "prediction_seal_sha256": sealed_predictions_seal(payload),
            "routes": rows,
        }


def sealed_predictions_seal(payload_without_seal: dict[str, Any]) -> str:
    """Return the verified canonical seal for an already stripped payload."""
    return _digest(payload_without_seal)
