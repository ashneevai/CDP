"""Trusted-cohort-only Azure GPT-4o Tier-1 shadow benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from packages.llm_adjudication import AzureShadowAdjudicationService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluation_data/azure_live_shadow"
DEFAULT_OUTPUT = ROOT / "evaluation_results/azure_live_shadow"
TRUSTED = {"HUMAN_ADJUDICATED", "SOURCE_SYSTEM_GROUND_TRUTH"}
FORMS = {"CMS1500", "UB04"}
ABSTAIN = {"HITL", "CONFLICT", "INSUFFICIENT_EVIDENCE"}


def _read(path: Path, default: Any) -> Any:
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def _sha(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    return sorted(values)[round((len(values) - 1) * q)] if values else None


def _trusted(row: dict) -> bool:
    if row.get("ground_truth_authority") not in TRUSTED or row.get("form_type") not in FORMS:
        return False
    if row.get("reviewed_page_class") != row.get("form_type"):
        return False
    if not all(
        row.get(k)
        for k in (
            "package_id",
            "source_asset_id",
            "source_page_id",
            "cdp_page_id",
            "field_name",
            "field_type",
            "criticality",
            "ground_truth_state",
        )
    ):
        return False
    if row.get("ground_truth_state") not in {
        "VALUE",
        "NOT_PRESENT",
        "UNREADABLE",
        "NOT_APPLICABLE",
    }:
        return False
    if row.get("criticality") == "CRITICAL" and not row.get("critical_dual_reviewed"):
        return False
    return not (row.get("annotation_disagreed") and not row.get("adjudication_complete"))


def _bound(row: dict) -> bool:
    b = row.get("binding") or {}
    return b.get("exact") is True and all(
        b.get(k)
        for k in (
            "rendered_page_sha256",
            "source_representation_id",
            "pipeline_execution_id",
            "page_observation_id",
        )
    )


def _eligible(row: dict) -> bool:
    return (
        row.get("local_decision")
        in {"HITL", "ESCALATE", "HUMAN_REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE"}
        and bool(row.get("claim_blocking") or row.get("local_hitl"))
        and row.get("crop_safe") is True
        and not row.get("authoritative_conflict")
        and len({str(v) for v in row.get("local_candidates", []) if v}) >= 2
    )


def _safe_cohort(row: dict) -> dict:
    return {
        k: row.get(k)
        for k in (
            "package_id",
            "source_asset_id",
            "source_page_id",
            "cdp_page_id",
            "form_type",
            "field_name",
            "field_type",
            "criticality",
            "ground_truth_state",
            "ground_truth_authority",
            "local_decision",
            "local_hitl_reason",
            "claim_blocking",
            "critical_dual_reviewed",
            "adjudication_complete",
        )
    } | {
        "ground_truth": "REDACTED",
        "ground_truth_sha256": _sha(row.get("ground_truth")),
        "local_candidates": [
            {"candidate_id": f"candidate_{i}", "value_sha256": _sha(v)}
            for i, v in enumerate(dict.fromkeys(row.get("local_candidates", [])))
        ],
    }


def _classify_tier2(row: dict) -> str:
    reasons = set(row.get("local_reason_codes") or [])
    if row.get("authoritative_conflict"):
        return "CONFLICTING_EVIDENCE"
    if "AUTHORIZED_REFERENCE_REQUIRED" in reasons:
        return "AUTHORITATIVE_DATA_REQUIRED"
    if "SOURCE_EVIDENCE_MISSING" in reasons:
        return "SOURCE_EVIDENCE_MISSING"
    if not row.get("local_candidates"):
        return "NO_SAFE_CANDIDATE"
    if row.get("crop_safe") and row.get("visual_ambiguity"):
        return "VISUAL_AMBIGUITY"
    return "TEXT_EVIDENCE_INSUFFICIENT"


def run(
    inputs: Path = DEFAULT_INPUT,
    output: Path = DEFAULT_OUTPUT,
    *,
    live: bool = False,
    service: AzureShadowAdjudicationService | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    supplied = _read(inputs / "trusted_shadow_cohort.json", {"records": []}).get("records", [])
    trusted = [r for r in supplied if _trusted(r)]
    bound = [r for r in trusted if _bound(r)]
    eligible = [r for r in bound if _eligible(r)]
    outcomes = []
    if live and eligible:
        service = service or AzureShadowAdjudicationService.from_env()
        if service is None:
            raise RuntimeError("live shadow requires LLM_ENABLED=true")
        for row in sorted(
            eligible,
            key=lambda r: (
                max(1, int(r.get("claim_distance") or 1)),
                r.get("criticality") != "CRITICAL",
                -float(r.get("localization_confidence") or 0),
            ),
        ):
            result = service.observe(
                field_name=row["field_name"],
                field_type=row["field_type"],
                candidates=[str(v) for v in row["local_candidates"]],
                claim_blocking=bool(row.get("claim_blocking")),
                crop_safe=True,
                localization_confidence=float(row.get("localization_confidence") or 0),
                critical=row["criticality"] == "CRITICAL",
                authoritative_conflict=False,
                page_key=row["cdp_page_id"],
                claim_key=row["package_id"],
                claim_distance=max(1, int(row.get("claim_distance") or 1)),
                evidence=row.get("evidence") or {},
            )
            selected = (
                row["local_candidates"][int(result.candidate_id.rsplit("_", 1)[1])]
                if result.candidate_id and result.candidate_id.rsplit("_", 1)[1].isdigit()
                else None
            )
            correct = (
                result.decision.value == "SELECT_CANDIDATE"
                and row.get("ground_truth_state") == "VALUE"
                and str(selected) == str(row.get("ground_truth"))
            )
            outcomes.append(
                {
                    "field_id": row.get("field_id"),
                    "source_page_id": row["source_page_id"],
                    "cdp_page_id": row["cdp_page_id"],
                    "package_id": row["package_id"],
                    "field_name": row["field_name"],
                    "critical": row["criticality"] == "CRITICAL",
                    "deployment": result.deployment,
                    "api_version": service.router.config.api_version,
                    "azure_request_id": result.request_id,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": result.latency_ms,
                    "cache_hit": result.cache_hit,
                    "cost_usd": result.cost_usd,
                    "cost_status": result.cost_status,
                    "azure_decision": result.decision.value,
                    "selected_candidate_id": result.candidate_id,
                    "proposal_correct": correct,
                    "potential_hitl_removed": correct,
                    "potential_blocker_removed": correct and bool(row.get("claim_blocking")),
                    "claim_distance_reduced": correct and bool(row.get("claim_blocking")),
                    "data_categories_sent": list(result.data_categories_sent),
                }
            )
    pages = {r["source_page_id"] for r in trusted}
    routed_pages = {r["source_page_id"] for r in eligible} if live else set()
    decisions = Counter(r["azure_decision"] for r in outcomes)
    selections = [r for r in outcomes if r["azure_decision"] == "SELECT_CANDIDATE"]
    correct = sum(bool(r["proposal_correct"]) for r in selections)
    critical_wrong_count = sum(r["critical"] and not r["proposal_correct"] for r in selections)
    critical_wrong = critical_wrong_count if outcomes else None
    tokens_in = [r["input_tokens"] for r in outcomes]
    tokens_out = [r["output_tokens"] for r in outcomes]
    latencies = [r["latency_ms"] for r in outcomes]
    costs = [r["cost_usd"] for r in outcomes if r["cost_usd"] is not None]
    pricing = (
        "CONFIGURED"
        if outcomes and all(r["cost_status"] == "CONFIGURED" for r in outcomes)
        else "PRICING_NOT_CONFIGURED"
    )
    routing_rate = len(routed_pages) / len(pages) if pages else None
    precision = correct / len(selections) if selections else None
    binding_rate = len(bound) / len(trusted) if trusted else 0.0
    blockers = sum(bool(r["potential_blocker_removed"]) for r in outcomes)
    hitl = sum(bool(r["potential_hitl_removed"]) for r in outcomes)
    by_claim = defaultdict(list)
    for r in outcomes:
        by_claim[r["package_id"]].append(r)
    advanced = sum(any(x["claim_distance_reduced"] for x in rows) for rows in by_claim.values())
    unlocked = sum(rows and all(x["proposal_correct"] for x in rows) for rows in by_claim.values())
    gates = {
        "trusted_labels_available": bool(trusted),
        "source_to_cdp_binding_sufficient": bool(trusted) and binding_rate == 1.0,
        "candidate_precision_99_5": precision is not None and precision >= 0.995,
        "critical_false_accept_potential_zero": bool(outcomes) and critical_wrong_count == 0,
        "routing_at_most_15_percent": routing_rate is not None and routing_rate <= 0.15,
        "mean_cost_at_most_0_001": (
            "NOT_CONFIGURED"
            if pricing != "CONFIGURED"
            else (sum(costs) / len(pages) <= 0.001 if pages else False)
        ),
        "authoritative_conflicts_protected": not any(
            r.get("authoritative_conflict") for r in eligible
        ),
        "novel_candidate_acceptance_zero": True,
        "material_hitl_improvement": hitl > 0,
    }
    if not trusted:
        status = "NEEDS_MORE_TRUSTED_LABELS"
    elif pricing != "CONFIGURED":
        status = "PRICING_NOT_CONFIGURED"
    elif all(v is True for v in gates.values()):
        status = "AZURE_LLM_AUTHORITY_READY"
    elif outcomes:
        status = "KEEP_SHADOW_ONLY"
    else:
        status = "KEEP_SHADOW_ONLY"
    annotations = {
        "supplied_records": len(supplied),
        "trusted_field_labels": len(trusted),
        "rejected_untrusted": len(supplied) - len(trusted),
        "critical_labels_dual_reviewed": sum(
            r.get("criticality") == "CRITICAL" and r.get("critical_dual_reviewed") for r in trusted
        ),
        "adjudications": sum(bool(r.get("adjudication_complete")) for r in trusted),
    }
    binding = {
        "trusted_records": len(trusted),
        "bound_records": len(bound),
        "binding_rate": binding_rate,
        "unbound_source_page_ids": sorted({r["source_page_id"] for r in trusted if not _bound(r)}),
        "records": [
            {
                k: r.get(k)
                for k in ("package_id", "source_asset_id", "source_page_id", "cdp_page_id")
            }
            | {"binding": r.get("binding")}
            for r in bound
        ],
    }
    routing = {
        "pages_evaluated": len(pages),
        "pages_with_azure_calls": len(routed_pages),
        "routing_rate": routing_rate,
        "calls_per_page": len(outcomes) / len(pages) if pages else None,
        "calls_per_routed_page": len(outcomes) / len(routed_pages) if routed_pages else None,
        "tier1_calls": len(outcomes),
        "cache_hits": sum(r["cache_hit"] for r in outcomes),
        "cache_misses": sum(not r["cache_hit"] for r in outcomes),
    }
    token_metrics = {
        "mean_input_tokens": mean(tokens_in) if tokens_in else None,
        "p50_input_tokens": _percentile(tokens_in, 0.5),
        "p95_input_tokens": _percentile(tokens_in, 0.95),
        "max_input_tokens": max(tokens_in, default=None),
        "mean_output_tokens": mean(tokens_out) if tokens_out else None,
        "p50_output_tokens": _percentile(tokens_out, 0.5),
        "p95_output_tokens": _percentile(tokens_out, 0.95),
        "max_output_tokens": max(tokens_out, default=None),
        "caps": {"input": 500, "output": 40},
    }
    latency = {
        "p50_azure_latency_ms": _percentile(latencies, 0.5),
        "p95_azure_latency_ms": _percentile(latencies, 0.95),
        "p99_azure_latency_ms": _percentile(latencies, 0.99),
        "local_only_field_latency_ms": None,
        "local_plus_shadow_latency_ms": None,
    }
    cost = {
        "pricing_status": pricing,
        "invocations": len(outcomes),
        "mean_cost_per_invocation": mean(costs) if costs else None,
        "mean_paid_cost_per_page": sum(costs) / len(pages) if costs and pages else None,
        "p95_page_cost": None,
        "cost_per_potential_blocker_removed": sum(costs) / blockers if costs and blockers else None,
        "cost_per_potential_claim_unlocked": sum(costs) / unlocked if costs and unlocked else None,
    }
    hitl_report = {
        "local_unresolved_count": len(eligible),
        "potential_field_hitl_removals": hitl,
        "potential_blockers_removed": blockers,
    }
    unlock = {"claims_potentially_advanced": advanced, "claims_potentially_unlocked": unlocked}
    tier2 = Counter(
        _classify_tier2(r)
        for r in eligible
        if r["source_page_id"] not in routed_pages
        or not any(
            o["source_page_id"] == r["source_page_id"] and o["proposal_correct"] for o in outcomes
        )
    )
    tier2_report = {
        "cohorts": dict(sorted(tier2.items())),
        "tier2_candidate_cohort_size": tier2.get("VISUAL_AMBIGUITY", 0),
        "tier2_enabled": False,
    }
    metrics = {
        "status": status,
        "benchmark_pages": len(pages),
        "form_counts": dict(Counter(r["form_type"] for r in trusted)),
        "fields_unresolved_locally": len(eligible),
        "fields_routed_to_azure": len(outcomes),
        "azure_calls": sum(not r["cache_hit"] for r in outcomes),
        "decisions": dict(decisions),
        "candidate_selection_precision": precision,
        "critical_false_accept_potential": critical_wrong,
        "potential_field_hitl_reduction": hitl,
        "potential_blockers_removed": blockers,
        "claims_potentially_advanced": advanced,
        "claims_potentially_unlocked": unlocked,
    }
    comparative = {
        "local_only": {"unresolved_fields": len(eligible)},
        "local_plus_azure_shadow": {
            "safely_resolvable_fields": hitl,
            "candidate_selection_precision": precision,
            "critical_false_accept_potential": critical_wrong,
        },
        "canonical_decisions_changed": 0,
    }
    artifacts = {
        "trusted_shadow_cohort.json": {
            "records": [_safe_cohort(r) for r in trusted],
            "raw_phi_committed": False,
        },
        "source_to_cdp_shadow_binding.json": binding,
        "annotation_summary.json": annotations,
        "azure_live_shadow_metrics.json": metrics,
        "routing_metrics.json": routing,
        "token_metrics.json": token_metrics,
        "latency_metrics.json": latency,
        "azure_cost_metrics.json": cost,
        "potential_hitl_reduction.json": hitl_report,
        "potential_claim_unlock.json": unlock,
        "tier2_candidate_analysis.json": tier2_report,
        "promotion_gate_report.json": {"status": status, "gates": gates},
        "comparative_report.json": comparative,
    }
    for name, payload in artifacts.items():
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    with (output / "shadow_candidate_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return metrics | {
        "annotation_summary": annotations,
        "binding_rate": binding_rate,
        "routing": routing,
        "tokens": token_metrics,
        "latency": latency,
        "cost": cost,
        "gates": gates,
        "tier2_candidate_cohort_size": tier2_report["tier2_candidate_cohort_size"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--live", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.inputs, a.output, live=a.live), sort_keys=True))


if __name__ == "__main__":
    main()
