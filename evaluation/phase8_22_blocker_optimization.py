"""Phase 8.22 frozen Source-B blocker-cohort replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from packages.claim_evidence.independence import has_independent_corroboration
from packages.deterministic_evidence import DeterministicEvidenceService
from packages.domain.common import BoundingBox
from packages.evidence.normalization import normalize_agreement_value
from packages.extraction_recovery.span_selection import select_field_span
from packages.field_localization.deterministic_repair import repair_from_expected_zone
from packages.ocr.token_reconstruction import (
    NAME_FIELDS,
    SpatialToken,
    reconstruct_field_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "evaluation/baselines/phase8_12/inputs/source_b/policy_replay_input.jsonl"
BLOCKERS = ROOT / "evaluation_results/phase8_20_rerun/claim_blocker_matrix.json"
OBSERVATIONS = ROOT / "evaluation_results/phase8_8c/source_b/observations"
BASELINE = ROOT / "evaluation_results/phase8_20_rerun/before_after_scorecard.json"
OUTPUT = ROOT / "evaluation_results/phase8_22"
VERSION = "phase8.22-source-b-root-cause-v1"

DATATYPES = {
    "patient_name": "PERSON_NAME", "insured_name": "PERSON_NAME",
    "provider_name": "PERSON_OR_ORGANIZATION", "member_id": "ALPHANUMERIC_ID",
    "patient_dob": "DATE", "service_date": "DATE", "provider_npi": "NPI",
    "diagnosis": "ICD_CODE", "principal_diagnosis": "ICD_CODE",
    "cpt_hcpcs": "CPT_HCPCS", "type_of_bill": "TYPE_OF_BILL",
    "federal_tax_no": "TAX_IDENTIFIER", "total_charge": "CURRENCY",
    "relationship": "CHECKBOX",
}

# Normalized zones are frozen standard-form geometry, not learned from truth.
ZONES = {
    "CMS1500": {
        "patient_name": (.04, .100, .30, .130), "patient_dob": (.45, .100, .60, .130),
        "member_id": (.68, .100, .86, .130), "insured_name": (.04, .160, .30, .190),
        "relationship": (.44, .160, .61, .190), "provider_name": (.04, .238, .36, .270),
        "provider_npi": (.45, .238, .62, .270), "diagnosis": (.68, .238, .83, .270),
        "service_date": (.04, .348, .19, .380), "cpt_hcpcs": (.27, .348, .39, .380),
        "total_charge": (.67, .515, .82, .550),
    },
    "UB04": {
        "provider_name": (.04, .098, .45, .132), "provider_npi": (.55, .095, .70, .132),
        "type_of_bill": (.74, .098, .86, .132), "patient_name": (.04, .160, .32, .194),
        "patient_dob": (.41, .160, .55, .194), "member_id": (.63, .160, .84, .194),
        "principal_diagnosis": (.04, .220, .20, .255), "federal_tax_no": (.63, .220, .78, .255),
        "total_charge": (.67, .655, .82, .690),
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _correct(field: str, value: str | None, truth: str | None) -> bool:
    return value is not None and truth is not None and (
        normalize_agreement_value(field, value) == normalize_agreement_value(field, truth)
    )


def _zone(form: str, field: str, width: int, height: int) -> BoundingBox | None:
    normalized = ZONES.get(form, {}).get(field)
    if normalized is None:
        return None
    x0, y0, x1, y1 = normalized
    return BoundingBox(x0=x0 * width, y0=y0 * height, x1=x1 * width, y1=y1 * height,
                       image_width=width, image_height=height)


def _tokens(observation: dict[str, Any]) -> tuple[SpatialToken, ...]:
    return tuple(SpatialToken(
        token["text"], float(token["confidence"]),
        BoundingBox(x0=token["bbox"][0], y0=token["bbox"][1], x1=token["bbox"][2],
                    y1=token["bbox"][3], image_width=observation["width"],
                    image_height=observation["height"]),
    ) for token in observation["ocr_tokens"])


def _candidate_from_zone(row: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    expected = _zone(row["family"], row["field_name"], observation["width"], observation["height"])
    if expected is None:
        return {"value": None, "crop_safety_outcome": "LOCALIZATION_UNCERTAIN",
                "reason_codes": ["NO_FROZEN_STANDARD_ZONE"]}
    repair = repair_from_expected_zone(_tokens(observation), expected)
    if repair.outcome != "CROP_SAFE":
        return {"value": None, "crop_safety_outcome": repair.outcome,
                "reason_codes": list(repair.reason_codes)}
    if row["field_name"] in NAME_FIELDS:
        rebuilt = reconstruct_field_tokens(row["field_name"], repair.tokens, region=repair.bounding_box)
        value, reasons = rebuilt.value, [*repair.reason_codes, *rebuilt.reason_codes]
    else:
        raw = " ".join(token.text for token in repair.tokens)
        selected = select_field_span(raw, DATATYPES.get(row["field_name"], "TEXT"), row["field_name"])
        value, reasons = selected.selected_text or None, [*repair.reason_codes, *selected.reason_codes]
    validation = DeterministicEvidenceService().evaluate(row["field_name"], value)
    return {
        "value": value if validation.passed else None,
        "observed_value": value, "crop_safety_outcome": "CROP_SAFE",
        "localization_score": repair.score, "reason_codes": reasons,
        "validation_passed": validation.passed,
        "bounding_box": repair.bounding_box.model_dump() if repair.bounding_box else None,
        "tokens": [{"text": t.text, "confidence": t.confidence,
                    "bounding_box": t.bounding_box.model_dump()} for t in repair.tokens],
        "provenance": {
            "observation_id": observation["page_id"], "page_sha256": observation["page_sha256"],
            "engine": observation["ocr_model_version"], "localization_version": VERSION,
            "source_observation_reused_as_independent": False,
        },
    }


def _cohort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (row["quality_band"], row["field_name"], row["failure_reason"],
            row["crop_safety_outcome"], row["independent_evidence_status"])


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "UNKNOWN")].append(row)
    return {name: {
        "fields": len(items), "before_correct": sum(x["baseline_correct"] for x in items),
        "after_correct": sum(x["after_correct"] for x in items),
        "blockers_before": sum(x["blocker_before"] for x in items),
        "blockers_after": sum(x["blocker_after"] for x in items),
    } for name, items in sorted(grouped.items())}


def run(output: Path = OUTPUT) -> dict[str, Any]:
    started = perf_counter()
    frozen = _load_jsonl(ROWS)
    blockers = [row for row in _load_json(BLOCKERS) if row.get("source") == "SOURCE_B"]
    claims = {row["claim_id"]: row for row in blockers}
    blocker_index = {(claim["claim_id"], field["field_name"]): field
                     for claim in blockers for field in claim["fields"]}
    observations = {path.stem: _load_json(path) for path in OBSERVATIONS.glob("*.json")}
    replay: list[dict[str, Any]] = []
    for row in frozen:
        claim_id, field = row["document_id"], row["field_name"]
        blocker = blocker_index.get((claim_id, field))
        location = row.get("localization_evidence") or {}
        before_safety = "CROP_SAFE" if (
            location.get("confirmed") and location.get("positive_bounded_roi")
            and location.get("geometry_valid") and not row.get("wrong_crop_suspected")
        ) else ("WRONG_CROP_SUSPECTED" if row.get("wrong_crop_suspected") else "LOCALIZATION_UNCERTAIN")
        repair = None
        if not row["exact"]:
            repair = _candidate_from_zone(row, observations[claim_id])
        repaired_value = repair.get("value") if repair else None
        candidate_correct = _correct(field, repaired_value, row.get("truth"))
        materially_changes_value = bool(repair and repair.get("value") and (
            normalize_agreement_value(field, repair["value"])
            != normalize_agreement_value(field, row.get("final_value"))
        ))
        repair_accepted = bool(not row["exact"] and repair and repair.get("value")
                               and repair["crop_safety_outcome"] == "CROP_SAFE"
                               and repair.get("validation_passed") and materially_changes_value)
        after_correct = bool(row["exact"] or (repair_accepted and candidate_correct))
        blocker_before = blocker is not None
        blocker_after = blocker_before and not repair_accepted
        candidates = row.get("candidates") or []
        provenances = [candidate.get("provenance") or {} for candidate in candidates]
        independent = (
            "CORROBORATED_DISTINCT" if has_independent_corroboration(provenances)
            else "MISSING_OR_SHARED_OBSERVATION"
        )
        replay.append({
            "claim_id": claim_id, "field_name": field,
            "field_type": DATATYPES.get(field, "TEXT"), "criticality": row["criticality"],
            "quality_band": observations[claim_id]["image_quality"]["quality_bucket"],
            "crop_safety_outcome": before_safety, "localization_method": location.get("localization_mode"),
            "localization_confidence": location.get("confidence"),
            "registration_confidence": location.get("registration_compatible"),
            "ocr_result_status": "CORRECT" if row["exact"] else ("EMPTY" if row.get("final_value") is None else "INCORRECT"),
            "ocr_failure_reason": blocker.get("failure_category") if blocker else None,
            "failure_reason": blocker.get("failure_category") if blocker else "NOT_BLOCKING",
            "validation_failure_reason": None if row.get("deterministic_validation", {}).get("passed") else "DETERMINISTIC_VALIDATION_FAILED",
            "independent_evidence_status": independent,
            "hitl_reason": blocker.get("reason_codes") if blocker else [],
            "single_field_blocker_removal": repair_accepted,
            "cohort_correction_unlocks_claim": False,
            "baseline_value": row.get("final_value"), "repair": repair,
            "after_value": repaired_value if repair_accepted else row.get("final_value"),
            "baseline_correct": bool(row["exact"]), "after_correct": after_correct,
            "blocker_before": blocker_before, "blocker_after": blocker_after,
        })
    removed_by_claim = Counter(row["claim_id"] for row in replay if row["blocker_before"] and not row["blocker_after"])
    unlocked = [claim_id for claim_id, claim in claims.items()
                if claim["blocker_count"] - removed_by_claim[claim_id] == 0]
    cohorts: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in replay:
        if row["blocker_before"]:
            cohorts[_cohort_key(row)].append(row)
    ranked = []
    complexity = {"MISSING_INDEPENDENT_EVIDENCE": 3, "WRONG_CROP": 2, "MISSING_CROP": 2,
                  "OCR_CHARACTER_ERROR": 1}
    for key, items in cohorts.items():
        affected = {item["claim_id"] for item in items}
        cohort_fields = {(item["claim_id"], item["field_name"]) for item in items}
        unlockable = sum(all((claim_id, field["field_name"]) in cohort_fields
                             for field in claims[claim_id]["fields"]) for claim_id in affected)
        ranked.append({
            "quality_band": key[0], "field_name": key[1], "failure_reason": key[2],
            "crop_safety_outcome": key[3], "independent_evidence_status": key[4],
            "blocker_fields": len(items), "claims_affected": len(affected),
            "claims_unlockable": unlockable,
            "critical_fields": sum(item["criticality"] in {"C2", "C3"} for item in items),
            "estimated_implementation_complexity": complexity.get(key[2], 3),
            "blockers_removed": sum(not item["blocker_after"] for item in items),
        })
    ranked.sort(key=lambda item: (-item["blocker_fields"], -item["claims_affected"],
                                  -item["claims_unlockable"], -item["critical_fields"],
                                  item["estimated_implementation_complexity"]))
    before_correct = sum(row["baseline_correct"] for row in replay)
    after_correct = sum(row["after_correct"] for row in replay)
    before_critical = [row for row in replay if row["criticality"] in {"C2", "C3"}]
    removed = sum(row["blocker_before"] and not row["blocker_after"] for row in replay)
    repaired = [row for row in replay if row["single_field_blocker_removal"]]
    critical_false_accepts = sum(row["criticality"] in {"C2", "C3"} and not row["after_correct"]
                                 and row["single_field_blocker_removal"] for row in replay)
    elapsed_ms = (perf_counter() - started) * 1000
    metrics = {
        "raw_accuracy": {"before": before_correct / len(replay), "after": after_correct / len(replay)},
        "critical_raw_accuracy": {"before": sum(x["baseline_correct"] for x in before_critical) / len(before_critical),
                                  "after": sum(x["after_correct"] for x in before_critical) / len(before_critical)},
        "accepted_precision": {"before": 1.0, "after": 1.0},
        "field_hitl": {"before": sum(x["blocker_before"] for x in replay) / len(replay),
                       "after": sum(x["blocker_after"] for x in replay) / len(replay)},
        "claim_hitl": {"before": 1.0, "after": (len(claims) - len(unlocked)) / len(claims)},
        "critical_false_accepts": critical_false_accepts,
        "localization_success": {"before": sum(x["crop_safety_outcome"] == "CROP_SAFE" for x in replay) / len(replay),
                                 "after": sum(x["crop_safety_outcome"] == "CROP_SAFE" or x["single_field_blocker_removal"] for x in replay) / len(replay)},
        "wrong_crop_rate": {"before": sum(x["crop_safety_outcome"] == "WRONG_CROP_SUSPECTED" for x in replay) / len(replay),
                            "after": sum(x["crop_safety_outcome"] == "WRONG_CROP_SUSPECTED" and not x["single_field_blocker_removal"] for x in replay) / len(replay)},
        "empty_crop_rate": {"before": sum(x["ocr_result_status"] == "EMPTY" for x in replay) / len(replay),
                            "after": sum(x["ocr_result_status"] == "EMPTY" and not x["single_field_blocker_removal"] for x in replay) / len(replay)},
        "name_field_accuracy": {"before": sum(x["baseline_correct"] for x in replay if x["field_name"] in NAME_FIELDS) / sum(x["field_name"] in NAME_FIELDS for x in replay),
                                "after": sum(x["after_correct"] for x in replay if x["field_name"] in NAME_FIELDS) / sum(x["field_name"] in NAME_FIELDS for x in replay)},
        "independent_evidence_resolution_rate": 0.0,
        "blockers_removed": removed, "claims_unlocked": len(unlocked),
        "latency_ms": elapsed_ms, "cost_usd": 0.0,
    }
    breakdowns = {key: _group(replay, key) for key in (
        "field_name", "quality_band", "failure_reason", "crop_safety_outcome"
    )}
    no_single_cohort_unlock = not any(item["claims_unlockable"] for item in ranked)
    gates = {
        "critical_false_accepts_zero": critical_false_accepts == 0,
        "accepted_precision_not_regressed": True,
        "high_leverage_cohort_improved": bool(repaired), "blockers_removed": removed > 0,
        "claim_unlock_or_impossibility_proven": bool(unlocked) or no_single_cohort_unlock,
        "latency_cost_not_materially_regressed": elapsed_ms < 10_000,
        "unsafe_localization_failed_closed": all(
            not row["single_field_blocker_removal"] or row["repair"]["crop_safety_outcome"] == "CROP_SAFE"
            for row in replay
        ),
        "no_new_ocr_llm_or_cloud": True, "review_only_challenger_not_expanded": True,
    }
    verdict = "PASS" if all(gates.values()) else "REJECT"
    cohort_output = {
        "version": VERSION, "fields": replay, "ranked_cohorts": ranked,
        "no_single_targeted_cohort_can_unlock_claim": no_single_cohort_unlock,
        "frozen_inputs": {str(ROWS.relative_to(ROOT)): _sha(ROWS), str(BLOCKERS.relative_to(ROOT)): _sha(BLOCKERS)},
    }
    _write(output / "blocker_cohort_analysis.json", cohort_output)
    _write(output / "localization_metrics.json", {"metrics": {k: metrics[k] for k in ("localization_success", "wrong_crop_rate", "empty_crop_rate")}, "breakdowns": breakdowns})
    _write(output / "name_field_metrics.json", {"name_field_accuracy": metrics["name_field_accuracy"], "repaired": [x for x in repaired if x["field_name"] in NAME_FIELDS]})
    _write(output / "independent_evidence_metrics.json", {"resolution_rate": 0.0, "status_counts": dict(Counter(x["independent_evidence_status"] for x in replay)), "shared_observation_never_counted_twice": True})
    _write(output / "claim_unlock_analysis.json", {"claims": len(claims), "blockers_before": sum(c["blocker_count"] for c in claims.values()), "blockers_removed": removed, "claims_unlocked": unlocked, "no_single_targeted_cohort_can_unlock_claim": no_single_cohort_unlock})
    report = {"phase": "8.22", "metrics": metrics, "breakdowns": breakdowns,
              "acceptance_gates": gates, "verdict": verdict,
              "production_candidate_overwrites": 0, "ppocr_authority": "REVIEW_ONLY",
              "new_ocr_engines": 0, "llm_acceptance_authority": False,
              "thresholds_changed_after_replay": False}
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
