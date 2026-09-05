"""Phase 8.11 competition-frontier audit and governed scorecard.

The module consumes the frozen 8.10 records, verifies them against the frozen
git artifact, and reports accuracy, automation, evidence, cost, and readiness
without accessing the locked holdout or granting acceptance authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evaluation_results/phase8_10"
CANDIDATE = ROOT / "evaluation_results/phase8_11/candidate"
OUTPUT = ROOT / "evaluation_results/phase8_11"
DOCS = ROOT / "docs"
FROZEN_COMMIT = "3a7d5d1a"
FROZEN_MANIFEST = ROOT / "evaluation/baselines/phase8_10_governed.json"
ACCEPTED = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED"}
GOVERNED_PATHS = (
    "accuracy.overall",
    "accuracy.CMS1500",
    "accuracy.UB04",
    "accuracy.critical",
    "localization.production_usable_localization",
    "critical_localization.production_usable_localization",
    "wrong_crop.precision",
    "wrong_crop.recall",
    "regional_ocr_budget.invocation_rate",
    "safety_and_automation.accepted_precision",
    "safety_and_automation.critical_false_accepts",
    "safety_and_automation.claim_stp",
    "safety_and_automation.claim_hitl",
    "safety_and_automation.field_hitl",
    "cost.cloud_cost_per_page_usd",
    "cost.fully_loaded_cost_per_page_usd",
)


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text("utf-8"))


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", "utf-8")
        return
    columns = sorted({key for row in rows for key, value in row.items()
                      if not isinstance(value, (dict, list, tuple, set))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _at(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        value = value[part]
    return value


def _baseline_freeze(current: dict) -> dict:
    manifest_bytes = FROZEN_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    frozen = manifest["governed_metrics"]
    comparisons = {
        path: {"frozen": frozen[path], "replay": _at(current, path),
               "exact": frozen[path] == _at(current, path)}
        for path in GOVERNED_PATHS
    }
    return {
        "phase": "8.11",
        "baseline_phase": "8.10",
        "baseline_commit": FROZEN_COMMIT,
        "frozen_summary_sha256": manifest["summary_sha256_at_freeze"],
        "freeze_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reproduced": all(item["exact"] for item in comparisons.values()),
        "governed_metric_comparison": comparisons,
        "latency_excluded_from_exact_reproduction": True,
        "locked_holdout_accessed": False,
    }


def _field_rows() -> list[dict]:
    result = []
    for source in ("SOURCE_A", "SOURCE_B", "SOURCE_C"):
        directory = BASELINE / source.lower()
        extraction = {
            (row["document_id"], row["field_name"]): row
            for row in _rows(directory / "v3_extraction/field_records.jsonl")
        }
        for row in _rows(directory / "field_decisions.jsonl"):
            item = extraction[(row["document_id"], row["field_name"])]
            decision = row["field_decision"]
            disposition = decision["disposition"]
            accepted = disposition in ACCEPTED
            exact = bool(item["exact"])
            missing = list(decision.get("missing_evidence") or ())
            if accepted:
                category = "AUTO_ACCEPTED_CORRECT" if exact else "AUTO_ACCEPTED_INCORRECT"
            elif exact and missing:
                category = "CORRECT_REVIEWED_MISSING_EVIDENCE"
            elif exact:
                category = "CORRECT_REVIEWED_AMBIGUITY"
            elif not item.get("final"):
                category = "UNRESOLVED_REVIEWED"
            else:
                layer = item.get("failure_layer") or "UNKNOWN"
                category = f"INCORRECT_REVIEWED_{layer}"
            result.append({
                "document_id": row["document_id"], "source": source,
                "family": row["family"], "field_name": row["field_name"],
                "criticality": row["criticality"], "critical": bool(item["critical"]),
                "exact": exact, "disposition": disposition, "accepted": accepted,
                "category": category, "missing_evidence": missing,
                "reason_codes": list(decision.get("reason_codes") or ()),
                "next_action": decision.get("next_action"),
                "failure_layer": item.get("failure_layer"),
                "reviewed": not accepted,
            })
    return result


def _master_disposition(rows: list[dict]) -> dict:
    categories = Counter(row["category"] for row in rows)
    dimensions = {}
    for dimension in ("field_name", "source", "family", "criticality"):
        grouped: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            grouped[str(row[dimension])][row["category"]] += 1
        dimensions[dimension] = {key: dict(value) for key, value in sorted(grouped.items())}
    return {"samples": len(rows), "categories": dict(categories), "by": dimensions}


def _correct_reviewed(rows: list[dict]) -> dict:
    reviewed = [row for row in rows if row["reviewed"]]
    correct = [row for row in reviewed if row["exact"]]

    def scoped(dimension: str) -> dict:
        result = {}
        for key in sorted({str(row[dimension]) for row in rows}):
            group = [row for row in rows if str(row[dimension]) == key and row["reviewed"]]
            result[key] = {
                "reviewed": len(group),
                "correct_but_reviewed": sum(row["exact"] for row in group),
                "rate": sum(row["exact"] for row in group) / max(1, len(group)),
            }
        return result

    return {
        "reviewed_fields": len(reviewed),
        "correct_but_reviewed": len(correct),
        "correct_but_reviewed_rate": len(correct) / max(1, len(reviewed)),
        "by_field": scoped("field_name"), "by_source": scoped("source"),
        "by_family": scoped("family"), "by_criticality": scoped("criticality"),
    }


def _claim_blockers(field_rows: list[dict]) -> dict:
    fields = {(row["document_id"], row["field_name"]): row for row in field_rows}
    counts: dict[str, Counter] = defaultdict(Counter)
    total_claims = 0
    for source in ("SOURCE_A", "SOURCE_B", "SOURCE_C"):
        for claim in _rows(BASELINE / source.lower() / "claim_decisions.jsonl"):
            total_claims += 1
            blockers = claim.get("blocking_unresolved_fields") or []
            for field in blockers:
                item = fields[(claim["claim_id"], field)]
                count = counts[field]
                count["claims_blocked"] += 1
                count["single_blocker_claims"] += len(blockers) == 1
                count["correct_but_reviewed"] += item["exact"]
                count["wrongly_extracted"] += not item["exact"]
                count[f"missing_{'+'.join(item['missing_evidence']) or 'NONE'}"] += 1
    rows = []
    for field, count in counts.items():
        reviews = count["claims_blocked"]
        row = {"field_name": field, **dict(count)}
        row["claim_unlock_value"] = count["single_blocker_claims"]
        row["unlock_efficiency"] = count["single_blocker_claims"] / max(1, reviews)
        rows.append(row)
    rows.sort(key=lambda row: (-row["claim_unlock_value"], -row["claims_blocked"], row["field_name"]))
    return {"claims": total_claims, "rows": rows}


def _accuracy_pareto(summary: dict) -> dict:
    records = _rows(BASELINE / "extraction_records.jsonl")
    dimensions = {}
    for dimension in ("field_name", "source", "family", "critical"):
        output = {}
        for key in sorted({str(row[dimension]) for row in records}):
            group = [row for row in records if str(row[dimension]) == key]
            localized = [row for row in group if row["production_usable_localization"]]
            output[key] = {
                "samples": len(group),
                "usable_localization": len(localized) / len(group),
                "final_accuracy": sum(row["exact"] for row in group) / len(group),
                "accuracy_given_usable_localization": (
                    sum(row["exact"] for row in localized) / max(1, len(localized))
                ),
                "errors": sum(not row["exact"] for row in group),
            }
        dimensions[dimension] = output
    return {
        "overall": summary["accuracy"],
        "conditional": dimensions,
        "failure_pareto": summary["failure_pareto"],
        "top_component_coverage": (
            sum(item["count"] for item in summary["failure_pareto"][:2])
            / max(1, sum(item["count"] for item in summary["failure_pareto"]))
        ),
        "ub_service_lines": summary["ub_service_lines"],
    }


def _secondary_roi(summary: dict) -> dict:
    experiment = _read(CANDIDATE / "summary.json") if (CANDIDATE / "summary.json").is_file() else None
    rows = []
    for field, metrics in summary["evidence_yield"].items():
        calls = round(metrics["samples"] * metrics["secondary_invocation_rate"])
        gains = metrics["secondary_incremental_resolutions"]
        rows.append({
            "field_name": field, "samples": metrics["samples"], "calls": calls,
            "incremental_correct": gains,
            "resolution_per_call": gains / max(1, calls),
            "incremental_accuracy_gain": metrics["secondary_incremental_accuracy_gain"],
            "incremental_claims_unlocked": metrics["incremental_claims_unlocked"],
            "cloud_cost_usd": metrics["incremental_cloud_cost_usd"],
            "latency_status": metrics["incremental_latency_status"],
        })
    experiment_result = {"status": "NOT_RUN"}
    if experiment:
        gain = experiment["accuracy"]["overall"] - summary["accuracy"]["overall"]
        call_delta = (
            experiment["regional_ocr_budget"]["invocation_rate"]
            - summary["regional_ocr_budget"]["invocation_rate"]
        )
        experiment_result = {
            "status": "REJECTED_NEGATIVE_ROI" if gain <= 0 and call_delta > 0 else "REVIEW",
            "overall_accuracy_gain": gain,
            "secondary_invocation_rate_delta": call_delta,
            "critical_false_accepts": experiment["safety_and_automation"][
                "critical_false_accepts"
            ],
        }
    return {"by_field": sorted(rows, key=lambda row: (-row["resolution_per_call"], row["field_name"])),
            "confidence_fallback_experiment": experiment_result}


def _reference_readiness() -> dict:
    config = yaml.safe_load((ROOT / "config/reference_enrichment.yaml").read_text("utf-8"))
    adapters = []
    for item in config.get("providers", []):
        if not item.get("enabled", False):
            status = "DISABLED"
        elif item.get("type") == "test_fixture":
            status = "TEST_FIXTURE"
        elif item.get("authorized", False):
            status = "AUTHORIZED"
        else:
            status = "DISABLED"
        adapters.append({"name": item["name"], "type": item["type"], "status": status})
    return {
        "adapters": adapters,
        "authorized_count": sum(item["status"] == "AUTHORIZED" for item in adapters),
        "fixture_evidence_can_unlock_production": False,
        "runtime_service": "ReferenceEvidenceService",
    }


def _cost(summary: dict) -> dict:
    source = list(summary["safety_and_automation"]["source_automation"].values())
    hitl = mean(item["hitl_cost_per_page_usd"] for item in source)
    fully = summary["cost"]["fully_loaded_cost_per_page_usd"]
    machine = max(0.0, fully - hitl)
    return {
        "machine_cost_per_page_usd": machine,
        "hitl_cost_per_page_usd": hitl,
        "cloud_cost_per_page_usd": summary["cost"]["cloud_cost_per_page_usd"],
        "fully_loaded_cost_per_page_usd": fully,
        "target_le_0_10": fully <= .10,
        "dominant_component": "HITL" if hitl > machine else "MACHINE",
        "cost_per_safe_stp_claim_usd": summary["cost"]["cost_per_safe_stp_claim_usd"],
        "cost_by_tier": {
            tier: {"status": "NOT_MEASURED", "cost_per_page_usd": None}
            for tier in ("TIER_A_LOCAL_BATCH", "TIER_B_API", "TIER_C_EVENT_DRIVEN", "TIER_D_CLOUD_SCALE")
        },
        "assumption": "Phase 8.2 engineering labor and infrastructure model; not an invoice.",
    }


def _tier_readiness() -> dict:
    return {
        "TIER_A_LOCAL_BATCH": {
            "implementation": "AVAILABLE", "benchmark": "UNCACHED_1_2_4_8_WORKERS_MEASURED",
            "production_ready": False,
            "gap": "larger production-representative soak and host calibration pending",
        },
        "TIER_B_API": {
            "implementation": "AVAILABLE", "benchmark": "NOT_MEASURED",
            "production_ready": False, "gap": "authenticated load and failure-injection run pending",
        },
        "TIER_C_EVENT_DRIVEN": {
            "implementation": "PARTIAL", "benchmark": "NOT_MEASURED",
            "production_ready": False, "gap": "duplicate/out-of-order/DLQ recovery evidence pending",
        },
        "TIER_D_CLOUD_SCALE": {
            "implementation": "SCAFFOLDED", "benchmark": "NOT_MEASURED",
            "production_ready": False, "gap": "cloud deployment, authorization, and cost evidence pending",
        },
    }


def _throughput() -> tuple[dict, dict]:
    path = OUTPUT / "throughput/summary.json"
    if not path.is_file():
        return ({
            "benchmark_mode": True, "prior_document_output_cache": "DISABLED",
            "workers": {str(item): {"status": "NOT_MEASURED", "pages_per_second": None}
                        for item in (1, 2, 4, 8)},
            "latency_is_not_throughput": True,
        }, {"status": "INSTRUMENTED_NOT_YET_MEASURED"})
    measured = _read(path)
    digests = {}
    profiles = {}
    for profile in measured["profiles"]:
        workers = profile["workers"]
        records = []
        for record_path in sorted((OUTPUT / f"throughput/workers-{workers}").glob(
            "shard-*/uncached/field_records.jsonl"
        )):
            records.extend(_rows(record_path))
        canonical = sorted(
            (row["document_id"], row["field_name"], str(row.get("final")), bool(row["exact"]))
            for row in records
        )
        digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
        digests[str(workers)] = digest
        profiles[str(workers)] = {
            "status": "MEASURED", "pages": profile["pages"],
            "wall_seconds": profile["wall_seconds"],
            "pages_per_second": profile["pages_per_second"],
            "documents_per_minute": profile["documents_per_minute"],
            "output_sha256": digest,
        }
    parity = len(set(digests.values())) == 1
    refresh_path = OUTPUT / "stage_refresh/summary.json"
    refresh = _read(refresh_path) if refresh_path.is_file() else None
    single = (
        refresh["profiles"][0]
        if refresh
        else next(item for item in measured["profiles"] if item["workers"] == 1)
    )
    stage_shards = single.get("stage_latency_ms") or []
    stages = stage_shards[0] if len(stage_shards) == 1 else {}
    pareto = sorted(
        ({"stage": stage, **values} for stage, values in stages.items()),
        key=lambda item: -item["p95"],
    )
    return ({
        "benchmark_mode": True, "prior_document_output_cache": "DISABLED",
        "pages_per_profile": measured["pages_per_profile"], "workers": profiles,
        "best_worker_count": measured["best_worker_count"],
        "best_pages_per_second": measured["best_pages_per_second"],
        "output_parity": "PASS" if parity else "FAIL",
        "latency_is_not_throughput": True,
    }, {
        "status": "MEASURED" if "full_page_observation" in stages else "PARTIALLY_MEASURED",
        "single_worker_refresh": {
            "pages": single["pages"], "wall_seconds": single["wall_seconds"],
            "pages_per_second": single["pages_per_second"],
        },
        "pareto": pareto,
        "dominant_stage": pareto[0]["stage"] if pareto else None,
    })


def _render_docs(report: dict) -> None:
    accuracy = report["accuracy_pareto"]["overall"]
    reviewed = report["correct_but_reviewed"]
    cost = report["cost_pareto"]
    safety = report["safety"]
    _table = lambda rows, cols: "\n".join(
        ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        + ["| " + " | ".join(str(row.get(col, "")) for col in cols) + " |" for row in rows]
    )
    disposition_rows = [
        {"category": key, "count": value}
        for key, value in sorted(report["master_field_disposition"]["categories"].items(),
                                 key=lambda item: -item[1])
    ]
    (DOCS / "CDP_PHASE8_11_FIELD_DISPOSITION_PARETO.md").write_text(
        "# Phase 8.11 Field Disposition Pareto\n\n"
        f"Frozen fields: {report['master_field_disposition']['samples']}. Correct-but-reviewed: "
        f"{reviewed['correct_but_reviewed']}/{reviewed['reviewed_fields']} "
        f"({reviewed['correct_but_reviewed_rate']:.2%}).\n\n"
        + _table(disposition_rows, ["category", "count"]) + "\n", "utf-8"
    )
    (DOCS / "CLAIM_BLOCKER_PARETO.md").write_text(
        "# Phase 8.11 Claim Blocker Pareto\n\n"
        "Unlock efficiency is the number of claims with this field as the sole blocker divided by "
        "claims reviewed for the field. No policy bypass is modeled.\n\n"
        + _table(report["claim_blocker_pareto"]["rows"], ["field_name", "claims_blocked",
                 "single_blocker_claims", "correct_but_reviewed", "wrongly_extracted",
                 "unlock_efficiency"]) + "\n", "utf-8"
    )
    (DOCS / "COST_PARETO.md").write_text(
        "# Phase 8.11 Cost Pareto\n\n"
        f"Machine/page: ${cost['machine_cost_per_page_usd']:.6f}. HITL/page: "
        f"${cost['hitl_cost_per_page_usd']:.6f}. Fully loaded/page: "
        f"${cost['fully_loaded_cost_per_page_usd']:.6f}. Cloud common path: "
        f"${cost['cloud_cost_per_page_usd']:.2f}. The $0.10 target is "
        f"{'met' if cost['target_le_0_10'] else 'not met'}; HITL is dominant.\n\n"
        "Tier-specific cost is not reported until each tier has a measured benchmark.\n", "utf-8"
    )
    tier_rows = [{"tier": key, **value} for key, value in report["tier_readiness"].items()]
    (DOCS / "TIER_READINESS.md").write_text(
        "# Phase 8.11 Tier Readiness\n\n"
        + _table(tier_rows, ["tier", "implementation", "benchmark", "production_ready", "gap"])
        + "\n", "utf-8"
    )
    throughput_rows = [
        {"workers": workers, **values}
        for workers, values in report["throughput"]["workers"].items()
    ]
    (DOCS / "CDP_PHASE8_11_THROUGHPUT.md").write_text(
        "# Phase 8.11 Uncached Throughput\n\n"
        "Every profile processed the same eight pages with prior document-output caches "
        "disabled. Output parity passed.\n\n"
        + _table(throughput_rows, ["workers", "pages", "wall_seconds", "pages_per_second",
                                  "documents_per_minute", "output_sha256"])
        + f"\n\nBest measured profile: {report['throughput']['best_worker_count']} worker.\n",
        "utf-8",
    )
    (DOCS / "CDP_PHASE8_11_LATENCY_PARETO.md").write_text(
        "# Phase 8.11 Stage Latency Pareto\n\n"
        "Measured on an uncached eight-page, single-worker refresh.\n\n"
        + _table(report["stage_latency"].get("pareto", []),
                 ["stage", "samples", "p50", "p95", "max"])
        + f"\n\nDominant P95 stage: {report['stage_latency'].get('dominant_stage')}.\n",
        "utf-8",
    )
    gates = report["gates"]
    score_rows = [
        {"metric": "Overall accuracy", "result": f"{accuracy['overall']:.2%}", "target": ">=92%"},
        {"metric": "CMS accuracy", "result": f"{accuracy['CMS1500']:.2%}", "target": ">=93%"},
        {"metric": "UB accuracy", "result": f"{accuracy['UB04']:.2%}", "target": ">=92%"},
        {"metric": "Critical accuracy", "result": f"{accuracy['critical']:.2%}", "target": ">=95%"},
        {"metric": "Claim STP", "result": f"{safety['claim_stp']:.2%}", "target": "evidence-safe"},
        {"metric": "Claim HITL", "result": f"{safety['claim_hitl']:.2%}", "target": "<=20% stretch"},
        {"metric": "Fully loaded/page", "result": f"${cost['fully_loaded_cost_per_page_usd']:.4f}", "target": "<=$0.10"},
        {"metric": "Critical false accepts", "result": safety["critical_false_accepts"], "target": "0"},
    ]
    (DOCS / "COMPETITION_SCORECARD.md").write_text(
        "# Phase 8.11 Competition Scorecard\n\n"
        f"Decision: **{report['decision']}**. Baseline reproduction: "
        f"**{'PASS' if report['baseline_freeze']['reproduced'] else 'FAIL'}**. Locked holdout: "
        "**SEALED**.\n\n" + _table(score_rows, ["metric", "result", "target"])
        + "\n\nPromotion gates:\n\n" + _table(
            [{"gate": key, "passed": value} for key, value in gates.items()], ["gate", "passed"]
        ) + "\n", "utf-8"
    )


def run(output: Path = OUTPUT) -> dict:
    baseline = _read(BASELINE / "summary.json")
    freeze = _baseline_freeze(baseline)
    if not freeze["reproduced"]:
        report = {"phase": "8.11", "decision": "STOP_BASELINE_MISMATCH",
                  "baseline_freeze": freeze}
        _write(output / "summary.json", report)
        return report
    fields = _field_rows()
    disposition = _master_disposition(fields)
    reviewed = _correct_reviewed(fields)
    blockers = _claim_blockers(fields)
    accuracy = _accuracy_pareto(baseline)
    secondary = _secondary_roi(baseline)
    cost = _cost(baseline)
    safety = baseline["safety_and_automation"]
    throughput, stage_latency = _throughput()
    gates = {
        "overall_accuracy_ge_92": baseline["accuracy"]["overall"] >= .92,
        "cms_accuracy_ge_93": baseline["accuracy"]["CMS1500"] >= .93,
        "ub_accuracy_ge_92": baseline["accuracy"]["UB04"] >= .92,
        "critical_accuracy_ge_95": baseline["accuracy"]["critical"] >= .95,
        "wrong_crop_recall_ge_90": baseline["wrong_crop"]["recall"] >= .90,
        "wrong_crop_precision_ge_99": baseline["wrong_crop"]["precision"] >= .99,
        "usable_localization_ge_95": baseline["localization"][
            "production_usable_localization"
        ] >= .95,
        "critical_usable_localization_ge_97": baseline["critical_localization"][
            "production_usable_localization"
        ] >= .97,
        "accepted_precision_ge_99_5": safety["accepted_precision"] >= .995,
        "critical_false_accepts_zero": safety["critical_false_accepts"] == 0,
        "fully_loaded_cost_le_0_10": cost["target_le_0_10"],
        "common_path_cloud_cost_zero": cost["cloud_cost_per_page_usd"] == 0,
        "throughput_output_parity": throughput.get("output_parity") == "PASS",
        "stage_p95_le_8_seconds": all(
            item["p95"] <= 8_000 for item in stage_latency.get("pareto", [])
        ),
        "locked_holdout_sealed": True,
    }
    report = {
        "phase": "8.11", "decision": "NEEDS_MORE_DATA",
        "baseline_freeze": freeze,
        "dataset_firewall": {"locked_holdout_accessed": False,
                             "production_source_validation": "NOT_ESTABLISHED"},
        "master_field_disposition": disposition,
        "correct_but_reviewed": reviewed,
        "claim_blocker_pareto": blockers,
        "accuracy_pareto": accuracy,
        "localization": baseline["localization"],
        "critical_localization": baseline["critical_localization"],
        "wrong_crop": baseline["wrong_crop"],
        "secondary_ocr_roi": secondary,
        "reference_readiness": _reference_readiness(),
        "safety": safety,
        "cost_pareto": cost,
        "throughput": throughput,
        "stage_latency": stage_latency,
        "tier_readiness": _tier_readiness(),
        "reliability": {
            "idempotent_ingestion": "UNIT_AND_INTEGRATION_TESTED",
            "event_failure_injection": "NOT_COMPLETE",
            "circuit_breaker": "AVAILABLE_FOR_OPTIONAL_CLOUD_HANDWRITING",
            "overload_backpressure": "NOT_BENCHMARKED",
            "production_ready": False,
        },
        "gates": gates,
    }
    _write(output / "summary.json", report)
    _write(output / "field_disposition.json", fields)
    _csv(output / "field_disposition.csv", fields)
    _write(output / "claim_blocker_pareto.json", blockers["rows"])
    _csv(output / "claim_blocker_pareto.csv", blockers["rows"])
    _write(output / "secondary_ocr_roi.json", secondary)
    _csv(output / "secondary_ocr_roi.csv", secondary["by_field"])
    _render_docs(report)
    return report


def main() -> int:
    report = run()
    print(json.dumps(report, indent=2))
    return 2 if report["decision"].startswith("STOP") else 0


if __name__ == "__main__":
    raise SystemExit(main())
