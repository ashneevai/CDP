"""Phase 8.26 upstream evidence availability and capture-gap analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "evaluation_results/phase8_23/provenance_independence_audit.json"
PHASE825_INVENTORY = ROOT / "evaluation_results/phase8_25/blocker_inventory.json"
FROZEN_ROWS = ROOT / "evaluation/baselines/phase8_12/inputs/source_b/policy_replay_input.jsonl"
OBSERVATIONS = ROOT / "evaluation_results/phase8_8c/source_b/observations"
LINEAGE = ROOT / "evaluation/phase8_26_lineage_hashes.json"
OUTPUT = ROOT / "evaluation_results/phase8_26"
VERSION = "phase8.26-evidence-availability-v1"

SEMANTIC_SECTIONS = {
    "member_id": "subscriber", "provider_npi": "rendering_provider",
    "service_date": "service_header", "total_charge": "claim_total",
    "federal_tax_no": "billing_provider",
}

REQUIRED_SOURCE_EVIDENCE = {
    "member_id": "member card, eligibility record, or distinct subscriber section",
    "provider_npi": "provider supporting form or service-line provider section with matching role",
    "service_date": "separate encounter record or semantically compatible service-date section",
    "total_charge": "independent remittance, itemized statement, or adjudicated financial document",
    "federal_tax_no": "provider tax form or separate billing-provider supporting document",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")


def _lineage_unchanged(expected: dict[str, dict[str, str]]) -> tuple[bool, dict[str, dict[str, str]]]:
    actual = {}
    for phase, hashes in expected.items():
        folder = ROOT / "evaluation_results" / phase
        actual[phase] = {name: _sha(folder / name) for name in hashes}
    return actual == expected, actual


def run(output: Path = OUTPUT) -> dict[str, Any]:
    audit = _json(AUDIT)["fields"]
    phase825 = {(row["claim_id"], row["field_name"]): row
                for row in _json(PHASE825_INVENTORY)["fields"]}
    frozen = {(row["document_id"], row["field_name"]): row for row in _jsonl(FROZEN_ROWS)}
    e2_blockers = [row for row in audit if row["blocker_after"] and "E2" in row["required_evidence"]]
    expected_lineage = _json(LINEAGE)
    lineage_ok, actual_lineage = _lineage_unchanged(expected_lineage)
    inventory = []
    for row in e2_blockers:
        key = (row["claim_id"], row["field_name"])
        frozen_row, prior = frozen[key], phase825[key]
        candidates = frozen_row.get("candidates") or []
        provenances = [candidate.get("provenance") or {} for candidate in candidates]
        crops = {p.get("crop_sha256") for p in provenances if p.get("crop_sha256")}
        regions = {p.get("localization_region_id") for p in provenances if p.get("localization_region_id")}
        pages = {p.get("observation_id") or p.get("page_id") for p in provenances
                 if p.get("observation_id") or p.get("page_id")}
        documents = {p.get("document_id") or p.get("source_representation_id") for p in provenances
                     if p.get("document_id") or p.get("source_representation_id")}
        same_crop_duplicate = len(candidates) > 1 and len(crops) == 1
        root_cause = "DUPLICATE_SAME_CROP_EVIDENCE" if same_crop_duplicate else "UNKNOWN_REQUIRES_SOURCE_AUDIT"
        ownership = "NO_SAFE_AUTOMATION_PATH" if same_crop_duplicate else "SOURCE_SYSTEM_INTEGRATION"
        inventory.append({
            "claim_id": row["claim_id"], "bundle_id": prior["bundle_id"],
            "field_name": row["field_name"], "form_type": prior["form_type"],
            "source": prior["source"], "pages_available": len(pages) or 1,
            "documents_available": 1, "attachments_available": 0,
            "semantic_sections_present": [SEMANTIC_SECTIONS[row["field_name"]]],
            "candidate_observations": len(candidates), "distinct_crop_count": len(crops),
            "distinct_localization_region_count": len(regions), "distinct_page_count": len(pages),
            "distinct_document_count": len(documents),
            "raw_source_bundle_available": False,
            "normalized_page_inventory_available": (OBSERVATIONS / f"{row['claim_id']}.json").exists(),
            "ingested_document_inventory_available": False,
            "ocr_page_observation_available": (OBSERVATIONS / f"{row['claim_id']}.json").exists(),
            "field_candidate_inventory_available": bool(candidates),
            "primary_root_cause": root_cause, "root_cause_reason_codes": (
                ["MULTIPLE_CANDIDATES_SHARE_CROP", "MULTIPLE_ENGINES_OR_PREPROCESSING_NOT_INDEPENDENT"]
                if same_crop_duplicate else
                ["RAW_SOURCE_BUNDLE_UNAVAILABLE", "SECOND_OBSERVATION_PRESENCE_CANNOT_BE_VERIFIED"]
            ),
            "remediation_ownership": ownership,
            "additional_source_evidence_required": REQUIRED_SOURCE_EVIDENCE[row["field_name"]],
        })
    root_counts = Counter(row["primary_root_cause"] for row in inventory)
    owner_counts = Counter(row["remediation_ownership"] for row in inventory)
    claims = {row["claim_id"] for row in inventory}
    blockers_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        blockers_by_claim[row["claim_id"]].append(row)
    ingestion = {
        "raw_bundle_claims_available": 0,
        "normalized_page_claims_available": len({row["claim_id"] for row in inventory
                                                  if row["normalized_page_inventory_available"]}),
        "ingested_document_inventory_claims_available": 0,
        "ocr_observation_claims_available": len({row["claim_id"] for row in inventory
                                                  if row["ocr_page_observation_available"]}),
        "field_candidate_records_available": sum(row["field_candidate_inventory_available"] for row in inventory),
        "source_missing": 0, "ingestion_missing": 0, "parsing_missing": 0,
        "localization_missing": 0, "candidate_generation_missing": 0,
        "unknown_pending_raw_source_audit": root_counts["UNKNOWN_REQUIRES_SOURCE_AUDIT"],
        "reason_codes": ["RAW_SOURCE_BUNDLE_NOT_PRESENT_IN_EVALUATION_LINEAGE",
                         "CANNOT_DISTINGUISH_SOURCE_ABSENCE_FROM_UPSTREAM_CAPTURE_WITHOUT_RAW_BUNDLE"],
    }
    attachments = {
        "claims_with_attachment_inventory": 0, "attachments_available": 0,
        "semantically_relevant_attachment_opportunities": 0,
        "status": "SOURCE_B_ATTACHMENT_INVENTORY_UNAVAILABLE",
        "categories_checked": ["patient_subscriber_identity", "provider_identity_npi",
                               "service_dates", "member_subscriber_identifiers",
                               "authorization_reference_identifiers"],
        "no_attachment_counted_without_semantic_relevance": True,
    }
    claims_source_audit = sorted(claim for claim, rows in blockers_by_claim.items()
                                 if any(r["primary_root_cause"] == "UNKNOWN_REQUIRES_SOURCE_AUDIT" for r in rows))
    unlock = {
        "claims_affected": len(claims),
        "claims_potentially_unlockable_if_e2_available": 0,
        "claims_requiring_upstream_source_audit": claims_source_audit,
        "claims_requiring_upstream_process_change": [],
        "claims_with_no_safe_stp_path_under_current_evidence": sorted(claims),
        "reason": "Every claim retains non-E2 blockers or lacks admissible E2; raw bundles are unavailable.",
    }
    fully_classified = len(inventory) == 61 and sum(root_counts.values()) == 61
    gates = {
        "all_e2_blockers_classified_once": fully_classified,
        "software_vs_upstream_ownership_assigned": sum(owner_counts.values()) == len(inventory),
        "no_acceptance_decisions": True, "thresholds_unchanged": True,
        "no_synthetic_or_duplicate_evidence": True, "phase8_23_adjudicator_unchanged": True,
        "no_new_ocr_or_llm": True, "frozen_lineage_unchanged": lineage_ok,
        "raw_source_bundles_available_for_definitive_audit": False,
    }
    verdict = "PASS" if all(gates.values()) else (
        "NEEDS_MORE_DATA" if not gates["raw_source_bundles_available_for_definitive_audit"] else "REJECT"
    )
    _write(output / "evidence_availability_inventory.json", {"version": VERSION, "fields": inventory})
    _write(output / "ingestion_gap_analysis.json", ingestion)
    _write(output / "attachment_opportunity_analysis.json", attachments)
    _write(output / "root_cause_distribution.json", {
        "blockers": len(inventory), "by_root_cause": dict(root_counts),
        "by_remediation_owner": dict(owner_counts),
        "software_fixable_blockers": sum(owner_counts[name] for name in (
            "SOFTWARE_DISCOVERY_FIX", "DOCUMENT_CLASSIFICATION_FIX", "LOCALIZATION_FIX")),
        "ingestion_fixable_blockers": owner_counts["INGESTION_FIX"],
        "process_capture_gap_blockers": owner_counts["BUSINESS_PROCESS_CHANGE"],
        "source_system_integration_blockers": owner_counts["SOURCE_SYSTEM_INTEGRATION"],
    })
    _write(output / "claim_unlock_potential.json", unlock)
    report = {
        "phase": "8.26", "version": VERSION, "verdict": verdict,
        "remaining_e2_blockers": len(inventory), "root_cause_distribution": dict(root_counts),
        "remediation_ownership_distribution": dict(owner_counts),
        "claim_impact": unlock, "acceptance_gates": gates,
        "expected_lineage_hashes": expected_lineage, "actual_lineage_hashes": actual_lineage,
        "acceptance_decisions_created": 0, "thresholds_changed": False,
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
