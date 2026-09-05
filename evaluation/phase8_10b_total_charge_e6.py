"""One-variable total-charge E6 reconciliation experiment for Phase 8.10B."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.phase8_7_stp import _service_lines
from packages.candidate_reconciliation import EvidenceReconciler
from packages.claim_decision import ClaimDecisionContext
from packages.claim_evidence import ClaimEvidenceBuilder
from packages.evidence import StructuralLocalizationEvidence
from packages.evidence.name_agreement import compare_patient_names
from packages.evidence_decision import (
    DecisionContext,
    EvidenceDecisionService,
    FieldDecision,
    FieldDisposition,
)
from packages.route_registry import RouteDefinition, RouteLifecycle, RouteRegistry
from packages.runtime_profile import DecisionServiceFactory

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "evaluation_results/phase8_10"
OUTPUT = ROOT / "evaluation_results/phase8_10b/total_charge_e6_experiment.json"
REPORT = ROOT / "docs/CDP_PHASE8_10B_TOTAL_CHARGE_E6_EXPERIMENT.md"
SOURCES = ("source_a", "source_b", "source_c")
ACCEPTED = {FieldDisposition.AUTO_ACCEPTED, FieldDisposition.REFERENCE_CONFIRMED}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _context(row: dict, cross_field_evidence: set[str], field_policy) -> DecisionContext:
    policy = field_policy.for_field(row["family"], row["field_name"])
    return DecisionContext(
        field_id=f"{row['document_id']}:{row['field_name']}",
        field_name=row["field_name"],
        document_family=row["family"],
        criticality=policy.criticality,
        required=policy.required,
        blocks_stp=policy.blocks_stp,
        requires_review_when_unresolved=policy.requires_review_when_unresolved,
        candidates=row["candidates"],
        deterministic_evidence=set(row["deterministic_validation"]["evidence"]),
        deterministic_evidence_version=row["deterministic_validation"]["version"],
        hard_validation_passed=row["deterministic_validation"]["passed"],
        structural_localization=StructuralLocalizationEvidence.model_validate(
            row["localization_evidence"]
        ),
        wrong_crop_suspected=row["wrong_crop_suspected"],
        cross_field_evidence=cross_field_evidence,
    )


def _correct(row: dict, decision: FieldDecision) -> bool:
    if (decision.selected_value or "").strip().casefold() == str(row["truth"] or "").strip().casefold():
        return True
    return row["field_name"] in {"patient_name", "insured_name", "provider_name"} and (
        compare_patient_names(decision.selected_value, row["truth"]).agrees
    )


def _metrics(rows: list[dict], decisions: dict[tuple[str, str], FieldDecision]) -> dict:
    joined = [(row, decisions[(row["document_id"], row["field_name"])]) for row in rows]
    accepted = [(row, decision) for row, decision in joined if decision.disposition in ACCEPTED]
    target = [(row, decision) for row, decision in joined if row["field_name"] == "total_charge"]
    target_accepted = [(row, decision) for row, decision in target if decision.disposition in ACCEPTED]
    return {
        "fields": len(joined),
        "accepted_fields": len(accepted),
        "accepted_precision": sum(_correct(row, decision) for row, decision in accepted) / max(1, len(accepted)),
        "critical_false_accepts": sum(
            not _correct(row, decision) and row["criticality"] == "C1"
            for row, decision in accepted
        ),
        "field_hitl": 1 - len(accepted) / max(1, len(joined)),
        "correct_but_reviewed": sum(
            _correct(row, decision) and decision.disposition not in ACCEPTED
            for row, decision in joined
        ),
        "total_charge": {
            "fields": len(target),
            "correct_candidates": sum(_correct(row, decision) for row, decision in target),
            "accepted": len(target_accepted),
            "accepted_correct": sum(_correct(row, decision) for row, decision in target_accepted),
            "false_accepts": sum(not _correct(row, decision) for row, decision in target_accepted),
            "correct_but_reviewed": sum(
                _correct(row, decision) and decision.disposition not in ACCEPTED
                for row, decision in target
            ),
            "dispositions": dict(Counter(decision.disposition.value for _, decision in target)),
            "reason_codes": dict(Counter(
                reason for _, decision in target for reason in decision.reason_codes
            )),
        },
    }


def _decision_projection(decision: FieldDecision) -> dict:
    return {
        "selected_value": decision.selected_value,
        "disposition": decision.disposition,
        "calibrated_probability": decision.calibrated_probability,
        "reason_codes": decision.reason_codes,
        "available_evidence": decision.available_evidence,
        "missing_evidence": decision.missing_evidence,
        "next_action": decision.next_action,
    }


def run(
    *,
    write_outputs: bool = True,
    candidate_financial_authority: bool = False,
    input_root: Path = INPUT,
    service_lines_by_source: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict:
    decision_bundle = DecisionServiceFactory.from_profile()
    treatment_service = decision_bundle.evidence_decision
    if candidate_financial_authority:
        identity = dict(treatment_service.configuration_identity)
        identity["runtime_profile_id"] = "phase8.14-financial-e6-candidate"
        candidate_route = RouteDefinition(
            route_id="UB04.total_charge.rapidocr.paddleocr.phase8.14",
            field="total_charge", form="UB04", primary_engine="rapidocr",
            confirmation_engine="paddleocr",
            preprocessing_profile="recorded-canonical-field-crop-v1",
            policy_version="evidence-policy-v4-dependency-aware",
            benchmark_dataset="phase8.10b-financial-e6",
            sample_count=60, standalone_accuracy=None, agreement_precision=None,
            false_agreement_count=0, mean_latency_ms=None, cost_per_call_usd=0,
            cost_status="LOCAL_CPU", status=RouteLifecycle.EVALUATION_ONLY,
        )
        candidate_registry = RouteRegistry(
            version="phase8.14-financial-e6-candidate",
            routes=[*decision_bundle.route_registry.routes, candidate_route],
        )
        treatment_service = EvidenceDecisionService(
            reconciler=EvidenceReconciler(
                calibration=decision_bundle.evidence_decision.reconciler.calibration,
                allow_authoritative_financial_e6=True,
            ),
            evidence_policy=decision_bundle.evidence_decision.evidence_policy,
            field_policy=decision_bundle.field_policy,
            route_registry=candidate_registry,
            route_mode="evaluation",
            configuration_identity=identity,
        )
    evidence_builder = ClaimEvidenceBuilder.load()
    all_rows: list[dict] = []
    treatment_e6: dict[tuple[str, str], set[str]] = {}

    for source in SOURCES:
        source_rows = _read_jsonl(input_root / source / "policy_replay_input.jsonl")
        all_rows.extend(source_rows)
        rows_by_document: dict[str, list[dict]] = defaultdict(list)
        for row in source_rows:
            rows_by_document[row["document_id"]].append(row)
        services = (
            service_lines_by_source.get(source, {})
            if service_lines_by_source is not None
            else _service_lines(input_root / source)
        )
        for document_id, document_rows in rows_by_document.items():
            values = {row["field_name"]: row["final_value"] for row in document_rows}
            claim_evidence = evidence_builder.build(
                claim_id=document_id,
                document_family=document_rows[0]["family"],
                claim_values=values,
                service_lines=services.get(document_id, []),
            )
            treatment_e6[(document_id, "total_charge")] = claim_evidence.evidence_types_for(
                "total_charge"
            )

    baseline: dict[tuple[str, str], FieldDecision] = {}
    treatment: dict[tuple[str, str], FieldDecision] = {}
    for row in all_rows:
        key = (row["document_id"], row["field_name"])
        original_cross = set(row["cross_field_evidence"])
        baseline[key] = decision_bundle.evidence_decision.decide(
            _context(row, original_cross, decision_bundle.field_policy)
        )
        experiment_cross = original_cross
        if row["field_name"] == "total_charge":
            experiment_cross = {
                item for item in original_cross if item != "CLAIM_TOTAL_RECONCILED"
            } | treatment_e6[key]
        service = treatment_service if row["field_name"] == "total_charge" else decision_bundle.evidence_decision
        treatment[key] = service.decide(
            _context(row, experiment_cross, decision_bundle.field_policy)
        )

    non_target_changes = [
        {"document_id": row["document_id"], "field": row["field_name"]}
        for row in all_rows
        if row["field_name"] != "total_charge"
        and _decision_projection(baseline[(row["document_id"], row["field_name"])])
        != _decision_projection(treatment[(row["document_id"], row["field_name"])])
    ]

    def claims(decisions: dict[tuple[str, str], FieldDecision]) -> list:
        grouped: dict[str, list[FieldDecision]] = defaultdict(list)
        families = {}
        for row in all_rows:
            grouped[row["document_id"]].append(decisions[(row["document_id"], row["field_name"])])
            families[row["document_id"]] = row["family"]
        return [
            decision_bundle.claim_decision.decide(
                ClaimDecisionContext(
                    claim_id=claim_id,
                    document_family=families[claim_id],
                    field_decisions=field_decisions,
                    policy_id=decision_bundle.claim_decision.policy_id,
                    policy_version=decision_bundle.claim_decision.policy_version,
                )
            )
            for claim_id, field_decisions in sorted(grouped.items())
        ]

    baseline_metrics = _metrics(all_rows, baseline)
    treatment_metrics = _metrics(all_rows, treatment)
    baseline_claims, treatment_claims = claims(baseline), claims(treatment)
    reconciled = sum(
        "CLAIM_TOTAL_CONFIRMED" in values for values in treatment_e6.values()
    )
    promoted = (
        treatment_metrics["total_charge"]["accepted_correct"]
        > baseline_metrics["total_charge"]["accepted_correct"]
        and treatment_metrics["total_charge"]["false_accepts"] == 0
        and treatment_metrics["critical_false_accepts"]
        == baseline_metrics["critical_false_accepts"]
        and not non_target_changes
    )
    result = {
        "experiment": "total_charge_claim_total_e6_name_alignment",
        "candidate_financial_authority": candidate_financial_authority,
        "runtime_profile_id": decision_bundle.profile.decision_identity()["runtime_profile_id"],
        "one_code_change": "CLAIM_TOTAL_RECONCILED -> CLAIM_TOTAL_CONFIRMED",
        "reconciled_claims": reconciled,
        "baseline": baseline_metrics,
        "treatment": treatment_metrics,
        "correct_but_reviewed_reduction": (
            treatment_metrics["total_charge"]["accepted_correct"]
            - baseline_metrics["total_charge"]["accepted_correct"]
        ),
        "non_total_charge_decision_changes": non_target_changes,
        "baseline_claim_stp": sum(claim.stp_eligible for claim in baseline_claims) / len(baseline_claims),
        "treatment_claim_stp": sum(claim.stp_eligible for claim in treatment_claims) / len(treatment_claims),
        "policy_changed": False,
        "ocr_changed": False,
        "localization_changed": False,
        "ub_reconstruction_changed": False,
        "decision": "PROMOTE" if promoted else "REVERT",
    }
    if not write_outputs:
        return result
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    REPORT.write_text(
        "# Phase 8.10B total-charge E6 experiment\n\n"
        "Exactly one behavior changed: the existing truth-blind claim-total reconciliation fact now uses the evidence name already allowed by the frozen policy.\n\n"
        f"- Reconciled claims: {reconciled}\n"
        f"- Total-charge correct-but-reviewed: {baseline_metrics['total_charge']['correct_but_reviewed']} -> {treatment_metrics['total_charge']['correct_but_reviewed']}\n"
        f"- Reduction: {result['correct_but_reviewed_reduction']}\n"
        f"- Total-charge false accepts: {baseline_metrics['total_charge']['false_accepts']} -> {treatment_metrics['total_charge']['false_accepts']}\n"
        f"- Overall accepted precision: {baseline_metrics['accepted_precision']:.2%} -> {treatment_metrics['accepted_precision']:.2%}\n"
        f"- Critical false accepts: {baseline_metrics['critical_false_accepts']} -> {treatment_metrics['critical_false_accepts']}\n"
        f"- Non-total-charge decision changes: {len(non_target_changes)}\n"
        f"- Claim STP: {result['baseline_claim_stp']:.2%} -> {result['treatment_claim_stp']:.2%}\n"
        f"- Decision: **{result['decision']}**\n\n"
        "OCR, localization, policy, UB reconstruction, HITL policy, and STP policy were unchanged.\n",
        "utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
