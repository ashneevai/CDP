"""Materialize reviewed truth plus exact CDP lineage into a runtime-only cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runs/azure_shadow"
OUTPUT = ROOT / "evaluation_data/azure_live_shadow/trusted_shadow_cohort.json"
GOVERNED = ROOT / "evaluation_results/azure_live_shadow"
CRITICAL = {
    "member_id",
    "subscriber_id",
    "NPI",
    "patient_DOB",
    "service_date",
    "total_charge",
    "principal_diagnosis",
}
STAGES = (
    "SOURCE_PRESENT",
    "DISCOVERED",
    "INGESTED",
    "CLASSIFIED",
    "PAGE_CREATED",
    "OBSERVED",
    "LOCALIZED",
    "OCR_EXECUTED",
    "CANDIDATE_GENERATED",
    "VALIDATED",
    "HITL_OR_ACCEPTED",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def _latest(rows: list[dict], key):
    out = {}
    for row in rows:
        out[key(row)] = row
    return out


def _norm(state: str, value: object) -> tuple[str, str]:
    return state, " ".join(str(value or "").upper().split())


def materialize(run_dir: Path = RUN, output: Path = OUTPUT, governed: Path = GOVERNED) -> dict:
    pages = _latest(_rows(run_dir / "page_reviews.jsonl"), lambda r: r["source_page_id"])
    annotations = _latest(
        _rows(run_dir / "annotations.jsonl"),
        lambda r: (r["source_page_id"], r["field_name"], r["annotator_role"]),
    )
    adjudications = _latest(
        _rows(run_dir / "adjudications.jsonl"), lambda r: (r["source_page_id"], r["field_name"])
    )
    executions = _latest(_rows(run_dir / "cdp_executions.jsonl"), lambda r: r["source_page_id"])
    records = []
    agreements = disagreements = trusted = 0
    for page_id, page in pages.items():
        if page.get("reviewed_class") not in {"CMS1500", "UB04"} or page.get("action") in {
            "SKIP",
            "UNKNOWN",
        }:
            continue
        execution = executions.get(page_id)
        if not execution:
            continue
        for field in execution.get("fields", []):
            name = field["field_name"]
            a = annotations.get((page_id, name, "ANNOTATOR_A"))
            b = annotations.get((page_id, name, "ANNOTATOR_B"))
            final = None
            authority = None
            dual = False
            adjudicated = False
            if name in CRITICAL:
                if not a or not b or a["annotator_id"] == b["annotator_id"]:
                    continue
                dual = True
                if _norm(a["state"], a.get("value")) == _norm(b["state"], b.get("value")):
                    final = a
                    authority = "HUMAN_ADJUDICATED"
                    agreements += 1
                else:
                    disagreements += 1
                    final = adjudications.get((page_id, name))
                    if final:
                        authority = "HUMAN_ADJUDICATED"
                        adjudicated = True
            else:
                final = a or b
                if final:
                    authority = "HUMAN_SINGLE_REVIEW"
            if not final:
                continue
            binding = execution.get("binding") or {}
            record = {
                "package_id": page["package_id"],
                "source_asset_id": page["source_asset_id"],
                "source_page_id": page_id,
                "cdp_page_id": execution.get("cdp_page_id"),
                "field_id": field.get("field_id"),
                "form_type": page["reviewed_class"],
                "reviewed_page_class": page["reviewed_class"],
                "source_quality_band": page["reviewed_quality_band"],
                "field_name": name,
                "field_type": field.get("field_type", "TEXT"),
                "criticality": "CRITICAL" if name in CRITICAL else "NON_CRITICAL",
                "ground_truth": final.get("final_value", final.get("value")),
                "ground_truth_state": final.get("final_state", final.get("state")),
                "ground_truth_authority": authority,
                "critical_dual_reviewed": dual,
                "annotation_disagreed": (dual and final is None) or adjudicated,
                "adjudication_complete": adjudicated,
                "local_candidates": field.get("local_candidates", []),
                "local_decision": field.get("local_decision"),
                "local_hitl": field.get("local_hitl", False),
                "local_hitl_reason": field.get("local_hitl_reason"),
                "claim_blocking": field.get("claim_blocking", False),
                "crop_safe": field.get("crop_safe", False),
                "localization_confidence": field.get("localization_confidence", 0),
                "authoritative_conflict": field.get("authoritative_conflict", False),
                "claim_distance": field.get("claim_distance", 1),
                "local_reason_codes": field.get("local_reason_codes", []),
                "visual_ambiguity": field.get("visual_ambiguity", False),
                "evidence": field.get("evidence", {}),
                "binding": binding,
            }
            records.append(record)
            trusted += authority == "HUMAN_ADJUDICATED"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", "utf-8")
    governed.mkdir(parents=True, exist_ok=True)
    claim_pages = [
        r
        for r in pages.values()
        if r.get("reviewed_class") in {"CMS1500", "UB04"}
        and r.get("action") not in {"SKIP", "UNKNOWN"}
    ]
    queue = (
        json.loads((governed / "review_cohort_candidates.json").read_text("utf-8"))
        if (governed / "review_cohort_candidates.json").exists()
        else {"selected_pages": 0}
    )
    review_progress = {
        "queue_size": queue.get("selected_pages", 0),
        "pages_reviewed": sum(r.get("action") not in {"SKIP"} for r in pages.values()),
        "cms1500_confirmed": sum(r.get("reviewed_class") == "CMS1500" for r in claim_pages),
        "ub04_confirmed": sum(r.get("reviewed_class") == "UB04" for r in claim_pages),
        "unknown": sum(r.get("reviewed_class") == "UNKNOWN" for r in pages.values()),
        "document_boundaries_confirmed": sum(
            str(r.get("boundary_action", "")).startswith("CONFIRM_") for r in pages.values()
        ),
        "remaining": max(0, queue.get("selected_pages", 0) - len(pages)),
    }
    annotation_summary = {
        "fields_annotated": len(annotations),
        "critical_fields_dual_reviewed": sum(
            1
            for key in {(p, f) for p, f, _ in annotations}
            if (key[0], key[1], "ANNOTATOR_A") in annotations
            and (key[0], key[1], "ANNOTATOR_B") in annotations
            and key[1] in CRITICAL
        ),
        "agreements": agreements,
        "disagreements": disagreements,
        "adjudications": len(adjudications),
        "trusted_labels": trusted,
        "runtime_values_committed": False,
    }
    drops = Counter()
    coverage = []
    for page in claim_pages:
        execution = executions.get(page["source_page_id"])
        present = set(execution.get("stages", [])) if execution else set()
        missing = [stage for stage in STAGES if stage not in present]
        for stage in missing:
            drops[stage] += 1
        coverage.append(
            {
                "source_page_id": page["source_page_id"],
                "pipeline_execution_id": execution.get("execution_id") if execution else None,
                "completed_stages": sorted(present),
                "missing_stages": missing,
            }
        )
    binding = {
        "benchmark_pages": len(claim_pages),
        "exact_cdp_bindings": sum(
            bool((executions.get(p["source_page_id"]) or {}).get("binding", {}).get("exact"))
            for p in claim_pages
        ),
        "binding_rate": sum(
            bool((executions.get(p["source_page_id"]) or {}).get("binding", {}).get("exact"))
            for p in claim_pages
        )
        / len(claim_pages)
        if claim_pages
        else 0.0,
    }
    (governed / "review_progress.json").write_text(
        json.dumps(review_progress, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    (governed / "annotation_summary.json").write_text(
        json.dumps(annotation_summary, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    (governed / "adjudication_summary.json").write_text(
        json.dumps(
            {
                "agreements": agreements,
                "disagreements": disagreements,
                "completed": len(adjudications),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    (governed / "pipeline_coverage.json").write_text(
        json.dumps(
            {"stages": list(STAGES), "drops": dict(drops), "records": coverage},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    (governed / "source_to_cdp_shadow_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    return {
        "review": review_progress,
        "annotation": annotation_summary,
        "binding": binding,
        "trusted_cohort_records": len(records),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=RUN)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--governed", type=Path, default=GOVERNED)
    a = p.parse_args()
    print(json.dumps(materialize(a.run_dir, a.output, a.governed), sort_keys=True))


if __name__ == "__main__":
    main()
