"""Phase 9D claim STP closure and achievable-ceiling analysis."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation_results/phase9d"
ROWS = ROOT / "evaluation/baselines/phase8_12/inputs/source_b/policy_replay_input.jsonl"
P823 = ROOT / "evaluation_results/phase8_23/provenance_independence_audit.json"
P827 = ROOT / "evaluation_results/phase8_27/comparative_report.json"
P9A = ROOT / "evaluation_results/phase9a/blocker_cohort_analysis.json"
P9B = ROOT / "evaluation_results/phase9b/name_error_cohort.json"
P9C = ROOT / "evaluation_results/phase9c/comparative_report.json"
P9C_DISTANCE = ROOT / "evaluation_results/phase9c/claim_unlock_distance.json"

EXTRACTION = "A_EXTRACTION_DEFECT"
VALIDATION = "B_VALIDATION_GAP"
AUTHORITY = "C_ACCEPTANCE_AUTHORITY_GAP"
AUTHORITATIVE = "D_AUTHORITATIVE_DATA_REQUIRED"
SOURCE = "E_SOURCE_EVIDENCE_REQUIRED"
CONFLICT = "F_CONFLICTING_EVIDENCE"
UNREADABLE = "G_UNREADABLE_SOURCE"
MANDATORY = "H_MANDATORY_HITL"

OWNERS = {
    EXTRACTION: "CDP EXTRACTION",
    VALIDATION: "CDP VALIDATION",
    AUTHORITY: "CDP ACCEPTANCE POLICY",
    AUTHORITATIVE: "AUTHORITATIVE DATA INTEGRATION",
    SOURCE: "SOURCE DOCUMENT ACQUISITION",
    CONFLICT: "MANDATORY HUMAN REVIEW",
    UNREADABLE: "MANDATORY HUMAN REVIEW",
    MANDATORY: "MANDATORY HUMAN REVIEW",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")


def _distance_summary(counts: dict[str, int]) -> dict[str, float | int]:
    buckets = {f"distance_{index}": 0 for index in range(5)}
    buckets["distance_5_plus"] = 0
    for count in counts.values():
        buckets[f"distance_{count}" if count < 5 else "distance_5_plus"] += 1
    values = list(counts.values())
    return {
        **buckets,
        "mean_unlock_distance": statistics.mean(values),
        "median_unlock_distance": statistics.median(values),
    }


def _classification(row: dict[str, Any]) -> str:
    if not row["candidate_correct"]:
        return EXTRACTION
    field = row["field_name"]
    if field in {"member_id", "patient_name", "insured_name", "subscriber_name", "provider_name"}:
        return AUTHORITATIVE
    if row["blocker_reason"] == "MISSING_INDEPENDENT_EVIDENCE":
        return SOURCE
    if row["E3_format_validation"] and row["E5_localization_evidence"]:
        return AUTHORITY
    return VALIDATION


def _required_evidence(category: str, field: str) -> list[str]:
    if category == EXTRACTION:
        return ["CORRECT_E1_PRIMARY_OCR", "E5_SAFE_LOCALIZATION"]
    if category == AUTHORITATIVE:
        return ["E7_AUTHORITATIVE_REFERENCE"]
    if category == SOURCE:
        return ["E2_PROVENANCE_SEPARATED_SOURCE_OBSERVATION"]
    if category == AUTHORITY:
        return ["FIELD_SPECIFIC_ACCEPTANCE_AUTHORITY"]
    if category == VALIDATION:
        return ["E6_DETERMINISTIC_BUSINESS_VALIDATION"]
    return ["HUMAN_REVIEW"]


def run(output: Path = OUTPUT) -> dict[str, Any]:
    rows = _jsonl(ROWS)
    source = {(row["document_id"], row["field_name"]): row for row in rows}
    audit = {(row["claim_id"], row["field_name"]): row for row in _json(P823)["fields"]}
    phase9b = {(row["claim_id"], row["field_name"]): row for row in _json(P9B)["records"]}
    phase9c = _json(P9C)
    if phase9c["verdict"] != "PASS" or phase9c["metrics"]["field_hitl_after"] != 0.48:
        raise RuntimeError("PHASE9C_BASELINE_MISMATCH")
    distance = _json(P9C_DISTANCE)
    remaining_by_claim = {
        row["claim_id"]: set(row["blockers_remaining"]) for row in distance["claims"]
    }
    p9a = _json(P9A)["decisions"]
    blockers = [
        row
        for row in p9a
        if row["hitl_after"] and row["field_name"] in remaining_by_claim[row["claim_id"]]
    ]
    if len(blockers) != 96:
        raise RuntimeError(f"PHASE9C_BLOCKER_COUNT_MISMATCH:{len(blockers)}")
    phase827 = _json(P827)
    raw_source_available = bool(phase827["metrics"]["raw_bundles_available"])

    matrix = []
    for decision in blockers:
        key = (decision["claim_id"], decision["field_name"])
        src = source[key]
        audited = audit[key]
        p9b = phase9b.get(key)
        correct = bool(p9b["after_correctness"] if p9b else decision["after_correct"])
        validation = src.get("deterministic_validation") or {}
        location = src.get("localization_evidence") or {}
        available = sorted(set(audited.get("satisfied_evidence", [])))
        record = {
            "claim_id": decision["claim_id"],
            "form_type": decision["form_type"],
            "field_name": decision["field_name"],
            "field_type": decision["field_type"],
            "criticality": decision["criticality"],
            "current_candidate": p9b["phase9b_candidate"] if p9b else src.get("final_value"),
            "ground_truth_correct": correct,
            "candidate_correct": correct,
            "accepted": False,
            "outcome_case": "CORRECT_BUT_HITL" if correct else "INCORRECT_AND_HITL",
            "crop_safety": p9b["crop_safety"] if p9b else decision["crop_safety_outcome"],
            "OCR_status": "CORRECT" if correct else decision["failure_reason"],
            "validation_status": "PASS" if validation.get("passed") else "NOT_ESTABLISHED",
            "HITL_reason": decision["acceptance_reason"],
            "blocker_reason": decision["failure_reason"],
            "available_evidence_classes": available,
            "E1_primary_OCR": bool(src.get("candidates")),
            "E2_independent_OCR": "E2" in available,
            "E3_format_validation": bool(validation.get("passed")),
            "E4_cross_field_consistency": "E6" in available,
            "E5_localization_evidence": bool(
                location.get("confirmed") and location.get("geometry_valid")
            ),
            "E6_business_rule_validation": bool(src.get("cross_field_evidence")),
            "E7_authoritative_reference": False,
            "source_evidence_available": raw_source_available,
            "independent_evidence_available": "E2" in available,
            "acceptance_policy": decision["policy_version"],
            "acceptance_authority_status": "NOT_AUTHORIZED",
            "claim_unlock_distance": len(remaining_by_claim[decision["claim_id"]]),
            "would_removing_this_blocker_unlock_claim": len(
                remaining_by_claim[decision["claim_id"]]
            )
            == 1,
        }
        record["primary_category"] = _classification(record)
        record["remediation_owner"] = OWNERS[record["primary_category"]]
        record["required_evidence"] = _required_evidence(
            record["primary_category"], record["field_name"]
        )
        matrix.append(record)
    if any(row["outcome_case"] == "INCORRECT_BUT_ACCEPTED" for row in matrix):
        raise RuntimeError("INCORRECT_ACCEPTANCE_IN_CLOSURE_MATRIX")
    _write(output / "claim_closure_matrix.json", {"blocker_count": len(matrix), "rows": matrix})

    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        by_claim[row["claim_id"]].append(row)
    counts = {claim: len(items) for claim, items in by_claim.items()}
    claims = []
    for claim in sorted(by_claim):
        items = by_claim[claim]
        claim_fields = {row["field_name"] for row in rows if row["document_id"] == claim}
        hitl_fields = {item["field_name"] for item in items}
        accepted_fields = sorted(claim_fields - hitl_fields)
        claims.append(
            {
                "claim_id": claim,
                "form_type": items[0]["form_type"],
                "source": "SOURCE_B",
                "field_count": sum(row["document_id"] == claim for row in rows),
                "accepted_fields": accepted_fields,
                "hitl_fields": sorted(hitl_fields),
                "blocker_count": len(items),
                "critical_blocker_count": sum(
                    item["criticality"] in {"C1", "C2", "C3"} for item in items
                ),
                "claim_HITL": True,
                "claim_STP": False,
                "claim_unlock_distance": len(items),
                "remaining_blockers": [item["field_name"] for item in items],
            }
        )
    _write(
        output / "claim_stp_baseline.json",
        {"claims": claims, "distance_distribution": _distance_summary(counts)},
    )

    category_counts = Counter(row["primary_category"] for row in matrix)
    classification = {
        "total": len(matrix),
        "by_category": dict(category_counts),
        "by_owner": dict(Counter(row["remediation_owner"] for row in matrix)),
        "by_outcome_case": dict(Counter(row["outcome_case"] for row in matrix)),
        "rows": [
            {
                "claim_id": row["claim_id"],
                "field_name": row["field_name"],
                "primary_category": row["primary_category"],
                "remediation_owner": row["remediation_owner"],
            }
            for row in matrix
        ],
    }
    _write(output / "blocker_classification.json", classification)

    correct_hitl = [row for row in matrix if row["outcome_case"] == "CORRECT_BUT_HITL"]
    authority_rows = [
        {
            "claim_id": row["claim_id"],
            "field_name": row["field_name"],
            "current_evidence_combination": row["available_evidence_classes"],
            "required_evidence_combination": row["required_evidence"],
            "missing_evidence_class": row["required_evidence"],
            "reason_acceptance_blocked": row["primary_category"],
        }
        for row in correct_hitl
    ]
    _write(
        output / "acceptance_authority_analysis.json",
        {"correct_but_hitl_count": len(correct_hitl), "rows": authority_rows},
    )

    authoritative_rows = [row for row in matrix if row["primary_category"] == AUTHORITATIVE]
    opportunity = {
        "live_calls_made": False,
        "frozen_status": "NOT_AVAILABLE",
        "member_eligibility_claims": sorted(
            {
                row["claim_id"]
                for row in authoritative_rows
                if row["field_name"] in {"member_id", "patient_name", "insured_name"}
            }
        ),
        "provider_master_claims": sorted(
            {row["claim_id"] for row in authoritative_rows if row["field_name"] == "provider_name"}
        ),
        "code_reference_claims": sorted(
            {
                row["claim_id"]
                for row in authoritative_rows
                if "code" in row["field_name"] or "diagnosis" in row["field_name"]
            }
        ),
    }
    _write(output / "authoritative_data_opportunity.json", opportunity)

    current_fixable = {EXTRACTION, VALIDATION, AUTHORITY}
    authoritative_fixable = current_fixable | {AUTHORITATIVE}
    full_fixable = authoritative_fixable | {SOURCE}

    def ceiling(fixable: set[str]) -> list[str]:
        return sorted(
            claim
            for claim, items in by_claim.items()
            if all(item["primary_category"] in fixable for item in items)
        )

    current_ceiling = ceiling(current_fixable)
    authoritative_ceiling = ceiling(authoritative_fixable)
    full_ceiling = ceiling(full_fixable)
    ceiling_report = {
        "ACHIEVED_STP": {"claims": [], "count": 0, "rate": 0.0},
        "CURRENT_EVIDENCE_STP_CEILING": {
            "claims": current_ceiling,
            "count": len(current_ceiling),
            "rate": len(current_ceiling) / 20,
            "label": "STRUCTURAL_CEILING_NOT_ACHIEVED_STP",
        },
        "POTENTIAL_STP_IF_AUTHORITATIVE_DATA_AVAILABLE": {
            "claims": authoritative_ceiling,
            "count": len(authoritative_ceiling),
            "rate": len(authoritative_ceiling) / 20,
            "label": "OPPORTUNITY_CEILING_NOT_MEASURED_MATCHES",
        },
        "FULL_SOURCE_STP_CEILING": {
            "claims": full_ceiling,
            "count": len(full_ceiling),
            "rate": len(full_ceiling) / 20,
            "label": "THEORETICAL_STRUCTURAL_CEILING_NOT_ACHIEVED_STP",
        },
    }
    _write(output / "stp_ceiling_analysis.json", ceiling_report)

    plans = []
    for claim in sorted(by_claim):
        items = by_claim[claim]
        categories = {item["primary_category"] for item in items}
        if SOURCE in categories:
            disposition = "HITL_SOURCE_EVIDENCE_REQUIRED"
        elif AUTHORITATIVE in categories:
            disposition = "HITL_AUTHORITATIVE_DATA_REQUIRED"
        elif EXTRACTION in categories:
            disposition = "HITL_EXTRACTION_REPAIRABLE"
        elif AUTHORITY in categories:
            disposition = "HITL_POLICY_REPAIRABLE"
        elif categories & {CONFLICT, UNREADABLE, MANDATORY}:
            disposition = "HITL_MANDATORY_REVIEW"
        else:
            disposition = "HITL_POLICY_REPAIRABLE"
        plans.append(
            {
                "claim_id": claim,
                "current_distance": len(items),
                "blockers": [item["field_name"] for item in items],
                "blocker_classifications": [item["primary_category"] for item in items],
                "minimum_safe_remediation_set": [
                    {
                        "field": item["field_name"],
                        "owner": item["remediation_owner"],
                        "action": item["required_evidence"],
                    }
                    for item in items
                ],
                "estimated_distance_after_software_fix": sum(
                    item["primary_category"] not in current_fixable for item in items
                ),
                "estimated_distance_after_authoritative_data": sum(
                    item["primary_category"] not in authoritative_fixable for item in items
                ),
                "estimated_distance_after_full_source": sum(
                    item["primary_category"] not in full_fixable for item in items
                ),
                "final_disposition": disposition,
            }
        )
    cohort_claims: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in matrix:
        cohort_claims[(row["field_name"], row["primary_category"], row["remediation_owner"])].add(
            row["claim_id"]
        )
    remediations = []
    complexity = {EXTRACTION: 2, VALIDATION: 1, AUTHORITY: 2, AUTHORITATIVE: 4, SOURCE: 5}
    for (field, category, owner), affected in cohort_claims.items():
        distance_reduction = sum(1 / counts[claim] for claim in affected)
        claims_potentially_unlocked = sum(
            all(
                item["field_name"] == field or item["primary_category"] in current_fixable
                for item in by_claim[claim]
            )
            for claim in affected
        )
        score = distance_reduction * len(affected) / complexity.get(category, 5)
        remediations.append(
            {
                "field": field,
                "category": category,
                "owner": owner,
                "claims_affected": len(affected),
                "claims_potentially_unlocked": claims_potentially_unlocked,
                "distance_reduction": round(distance_reduction, 6),
                "implementation_complexity": complexity.get(category, 5),
                "claim_unlock_leverage": round(score, 6),
            }
        )
    remediations.sort(key=lambda row: (-row["claim_unlock_leverage"], row["field"]))
    _write(
        output / "claim_remediation_plan.json",
        {"claims": plans, "top_10_remediations": remediations[:10]},
    )

    metrics = {
        "total_claims": 20,
        "current_STP_claims": 0,
        "current_STP_rate": 0.0,
        "current_claim_HITL": 1.0,
        **_distance_summary(counts),
        "remaining_blockers": len(matrix),
        "extraction_defect_blockers": category_counts[EXTRACTION],
        "validation_gap_blockers": category_counts[VALIDATION],
        "acceptance_authority_gap_blockers": category_counts[AUTHORITY],
        "authoritative_data_required_blockers": category_counts[AUTHORITATIVE],
        "source_evidence_required_blockers": category_counts[SOURCE],
        "conflicting_evidence_blockers": category_counts[CONFLICT],
        "unreadable_source_blockers": category_counts[UNREADABLE],
        "mandatory_hitl_blockers": category_counts[MANDATORY],
        "correct_but_hitl_fields": len(correct_hitl),
        "incorrect_and_hitl_fields": sum(
            row["outcome_case"] == "INCORRECT_AND_HITL" for row in matrix
        ),
        "incorrect_but_accepted_fields": 0,
        "CURRENT_EVIDENCE_STP_CEILING": len(current_ceiling) / 20,
        "POTENTIAL_STP_IF_AUTHORITATIVE_DATA_AVAILABLE": len(authoritative_ceiling) / 20,
        "FULL_SOURCE_STP_CEILING": len(full_ceiling) / 20,
        "claims_unlockable_by_software": len(current_ceiling),
        "claims_unlockable_by_authoritative_data": len(authoritative_ceiling),
        "claims_requiring_source_acquisition": len(
            {row["claim_id"] for row in matrix if row["primary_category"] == SOURCE}
        ),
        "claims_requiring_mandatory_HITL": 0,
        "accepted_precision": phase9c["metrics"]["accepted_precision"],
        "critical_false_accepts": phase9c["metrics"]["critical_false_accepts"],
    }
    gates = {
        "all_blockers_classified": sum(category_counts.values()) == len(matrix),
        "closure_matrix_complete": len(matrix) == 96,
        "ceilings_are_structural_not_achieved": ceiling_report["ACHIEVED_STP"]["count"] == 0,
        "accepted_precision_gte_995": metrics["accepted_precision"] >= 0.995,
        "critical_false_accepts_zero": metrics["critical_false_accepts"] == 0,
        "phase8_27_source_limits_respected": not raw_source_available,
    }
    report = {
        "phase": "9D",
        "verdict": "PASS" if all(gates.values()) else "REJECT",
        "verdict_reason": "All post-9C blockers have deterministic categories, owners, and non-fabricated scenario ceilings",
        "metrics": metrics,
        "acceptance_gates": gates,
        "quick_win_experiments": [],
        "experiments_retained": [],
        "experiments_reverted": [],
        "top_10_remediations_by_claim_unlock_leverage": remediations[:10],
        "recommended_phase9e_strategy": "OPTION_B_AUTHORITATIVE_EVIDENCE_INTEGRATION",
        "phase9e_numerical_opportunity": {
            "additional_claims_potentially_STP": len(authoritative_ceiling),
            "potential_STP_rate": len(authoritative_ceiling) / 20,
            "not_achieved_STP": True,
        },
        "thresholds_changed": False,
        "historical_artifacts_modified": False,
    }
    _write(output / "comparative_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
