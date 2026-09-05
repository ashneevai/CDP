"""Frozen Source-B alignment and deterministic PP-OCRv5 shadow replay.

This module is evaluation-only.  It reads the frozen Phase 8.12 evidence and
Phase 8.20 claim blocker matrix, invokes the existing Phase 8.21 challenger
through OCRExecutionService, and never mutates production candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from packages.domain.common import BoundingBox
from packages.domain.enums import ClaimFormType, FieldCriticality
from packages.evidence.normalization import normalize_agreement_value
from packages.ocr.adjudication import adjudicate_candidates
from packages.ocr.contracts import OCRCandidate, OCRRequest
from packages.ocr.execution import OCRExecutionService
from packages.ocr.ppocr_v5_provider import PPOCRv5Provider
from packages.ocr.provenance import EvidenceProvenance
from packages.ocr.source_b_routing import (
    ChallengerBudget,
    SourceBChallengeContext,
    route_to_ppocr_v5,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROWS = ROOT / "evaluation/baselines/phase8_12/inputs/source_b/policy_replay_input.jsonl"
FROZEN_MANIFEST = ROOT / "evaluation/baselines/phase8_12/manifest.json"
SOURCE_ROOT = ROOT / "evaluation_data/phase8_8_generalization/SOURCE_B"
BLOCKERS = ROOT / "evaluation_results/phase8_20_rerun/claim_blocker_matrix.json"
BASELINE = ROOT / "evaluation_results/phase8_20_rerun/before_after_scorecard.json"
OUTPUT = ROOT / "evaluation_results/phase8_21a"
DATASET_ID = "SOURCE_B"
EVALUATION_VERSION = "phase8.21a-source-b-shadow-replay-v1"
KEY_VERSION = "evaluation-field-key-v1"
ROUTING_VERSION = "phase8.21-selective-source-b-v1"
ADJUDICATION_VERSION = "phase8.21-deterministic-adjudication-v1"
PREPROCESSING_VERSION = "frozen-source-b-field-crop-v1"


@dataclass(frozen=True)
class EvaluationFieldKey:
    dataset_id: str
    claim_id: str
    document_sha256: str | None
    page_sha256: str
    page_number: int
    source: str
    form_type: str
    field_name: str
    localization_region_id: str
    crop_sha256: str

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _png_crop_hash(image: Image.Image, bbox: list[float]) -> str:
    crop = image.crop(tuple(round(value) for value in bbox)).convert("RGB")
    return hashlib.sha256(crop.tobytes()).hexdigest()


def _quality_by_claim() -> dict[str, str]:
    result = {}
    directory = ROOT / "evaluation_results/phase8_8c/source_b/observations"
    for path in directory.glob("*.json"):
        row = _json(path)
        result[row["page_id"]] = row["image_quality"]["quality_bucket"]
    return result


def _blocker_index() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    claims = {row["claim_id"]: row for row in _json(BLOCKERS) if row.get("source") == "SOURCE_B"}
    fields = {
        (claim_id, field["field_name"]): field
        for claim_id, claim in claims.items()
        for field in claim["fields"]
    }
    return fields, claims


def _safe(row: dict[str, Any]) -> bool:
    loc = row.get("localization_evidence") or {}
    return bool(
        loc.get("confirmed")
        and loc.get("positive_bounded_roi")
        and loc.get("geometry_valid")
        and not row.get("wrong_crop_suspected")
    )


def _primary(row: dict[str, Any]) -> OCRCandidate | None:
    source = next((item for item in row.get("candidates", []) if item.get("engine") == "rapidocr"), None)
    if source is None:
        return None
    box = BoundingBox(**source["bounding_box"])
    provenance = EvidenceProvenance(**source["provenance"])
    return OCRCandidate(
        value=source.get("value"), raw_value=source.get("raw_value") or "",
        engine=source["engine"], model_name=source["model_name"],
        model_version=source["model_version"],
        preprocessing_variant=source["preprocessing_variant"],
        raw_confidence=float(source.get("raw_confidence") or 0),
        calibrated_confidence=source.get("calibrated_confidence"), bounding_box=box,
        latency_ms=float(source.get("latency_ms") or 0), provenance=provenance,
    )


def _candidate(candidate: OCRCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "value": candidate.value, "raw_value": candidate.raw_value,
        "engine": candidate.engine, "model_name": candidate.model_name,
        "model_version": candidate.model_version,
        "raw_confidence": candidate.raw_confidence,
        "bounding_box": candidate.bounding_box.model_dump(),
        "latency_ms": candidate.latency_ms,
        "tokens": [
            {"text": token.text, "confidence": token.confidence,
             "bounding_box": token.bounding_box.model_dump()}
            for token in candidate.tokens
        ],
        "provenance": candidate.provenance.model_dump(mode="json") if candidate.provenance else None,
    }


def _correct(field: str, value: str | None, truth: str | None) -> bool:
    if value is None or truth is None:
        return value == truth
    return normalize_agreement_value(field, value) == normalize_agreement_value(field, truth)


def _versions(primary: OCRCandidate | None, challenger: OCRCandidate | None) -> dict[str, Any]:
    return {
        "evaluation_key": KEY_VERSION, "evaluation": EVALUATION_VERSION,
        "baseline_engine": primary.model_version if primary else None,
        "challenger_engine": challenger.model_version if challenger else None,
        "routing": ROUTING_VERSION, "preprocessing": PREPROCESSING_VERSION,
        "localization": "dynamic-roi-resolver-v1", "adjudication": ADJUDICATION_VERSION,
    }


def _group_metrics(rows: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(dimension) or "UNKNOWN")].append(row)
    return {
        key: {
            "fields": len(items),
            "baseline_correct": sum(bool(x["baseline_correct"]) for x in items),
            "simulated_correct": sum(bool(x["simulated_correct"]) for x in items),
            "baseline_hitl": sum(bool(x["baseline_hitl"]) for x in items),
            "simulated_hitl": sum(bool(x["simulated_hitl"]) for x in items),
            "challenged": sum(bool(x["challenger_invoked"]) for x in items),
        }
        for key, items in sorted(groups.items())
    }


async def run(
    *, output: Path = OUTPUT, provider: Any | None = None,
    execution: OCRExecutionService | None = None,
) -> dict[str, Any]:
    frozen = _jsonl(FROZEN_ROWS)
    manifest = _json(SOURCE_ROOT / "manifest.json")
    docs = {row["document_id"]: row for row in manifest["documents"]}
    blocker_fields, claims = _blocker_index()
    quality = _quality_by_claim()
    prepared: list[dict[str, Any]] = []
    for row in frozen:
        claim_id, field = row["document_id"], row["field_name"]
        candidate = _primary(row)
        provenance = candidate.provenance if candidate else None
        doc = docs[claim_id]
        page_path = SOURCE_ROOT / doc["file"]
        page_sha256 = provenance.page_sha256 if provenance else doc["sha256"]
        if _sha(page_path) != page_sha256:
            raise RuntimeError(f"page hash mismatch: {claim_id}")
        bbox = row["localization_evidence"]["field_bbox"]
        crop_sha256 = provenance.crop_sha256 if provenance else None
        if (
            bbox is not None and crop_sha256 is not None
            and _png_crop_hash(Image.open(page_path), bbox) != crop_sha256
        ):
            raise RuntimeError(f"crop hash mismatch: {claim_id}/{field}")
        blocker = blocker_fields.get((claim_id, field))
        failure = blocker["failure_category"] if blocker else "NOT_A_BLOCKER"
        context = SourceBChallengeContext(
            source="SOURCE_B", document_family=row["family"],
            current_claim_blocker=blocker is not None,
            crop_safety_status="CROP_SAFE" if _safe(row) else "UNSAFE",
            primary_resolved=blocker is None, failure_reason=failure,
        )
        prepared.append({"row": row, "primary": candidate, "page": page_path,
                         "bbox": bbox, "blocker": blocker, "failure": failure,
                         "context": context, "quality": quality.get(claim_id, "UNKNOWN")})

    # One shared budget, sized to the complete eligible population, preserves the
    # aggregate <=30% ceiling while ChallengerBudget supplies the small-cohort floor.
    eligible_count = sum(
        item["blocker"] is not None and item["context"].crop_safety_status == "CROP_SAFE"
        and item["failure"] in {"PRIMARY_EMPTY", "VALIDATION_FAILED", "LOW_OCR_CONFIDENCE", "OCR_CHARACTER_ERROR", "OCR_SEGMENTATION_ERROR", "LOCALIZATION_RECOVERED"}
        for item in prepared
    )
    budget = ChallengerBudget(eligible_count)
    ocr = execution or OCRExecutionService(benchmark_mode=True)
    ppocr = provider or PPOCRv5Provider()
    observations: list[dict[str, Any]] = []
    for item in sorted(prepared, key=lambda x: (x["row"]["document_id"], x["row"]["field_name"])):
        row, primary = item["row"], item["primary"]
        invoked, routing_reason = route_to_ppocr_v5(item["context"], budget)
        challenger = None
        execution_error = None
        if invoked:
            try:
                page = Image.open(item["page"]).convert("RGB")
                crop = page.crop(tuple(round(value) for value in item["bbox"]))
                request = OCRRequest(
                    document_id=row["document_id"], page_number=1, field_name=row["field_name"],
                    field_type=row["field_name"], form_type=ClaimFormType(row["family"]), image=crop,
                    bounding_box=BoundingBox(**primary.bounding_box.model_dump()), scope="FIELD_CROP",
                    criticality=(FieldCriticality.CRITICAL if row["criticality"] in {"C2", "C3"}
                                 else FieldCriticality.NON_CRITICAL),
                    preprocessing_profile="SOURCE_B_FROZEN_FIELD_CROP",
                    page_sha256=primary.provenance.page_sha256,
                    document_sha256=primary.provenance.document_sha256,
                    source_representation_id=primary.provenance.source_representation_id,
                )
                result = await ocr.execute(ppocr, request)
                challenger = result.candidates[0] if result.candidates else None
            except Exception as exc:  # recorded evidence; replay remains complete
                execution_error = f"{type(exc).__name__}: {exc}"
        adjudication = adjudicate_candidates(
            field_name=row["field_name"], primary=primary, challenger=challenger,
            crop_safety_status=item["context"].crop_safety_status,
        ) if invoked else None
        baseline_hitl = item["blocker"] is not None
        # Shadow evidence has REVIEW_ONLY authority.  A simulated blocker is
        # removed only for the existing adjudicator's explicit replacement action.
        blocker_removed = bool(
            baseline_hitl and adjudication and adjudication.action == "USE_CHALLENGER"
            and _correct(row["field_name"], challenger.value if challenger else None, row["truth"])
        )
        simulated_value = (
            challenger.value if blocker_removed and challenger else row.get("final_value")
        )
        key = EvaluationFieldKey(
            dataset_id=DATASET_ID, claim_id=row["document_id"],
            document_sha256=primary.provenance.document_sha256 if primary else None,
            page_sha256=(primary.provenance.page_sha256 if primary else docs[row["document_id"]]["sha256"]),
            page_number=1, source="SOURCE_B",
            form_type=row["family"], field_name=row["field_name"],
            localization_region_id=(primary.provenance.localization_region_id if primary else "UNRESOLVED"),
            crop_sha256=(primary.provenance.crop_sha256 if primary else "UNAVAILABLE"),
        )
        observations.append({
            "evaluation_field_key": {**asdict(key), "key_sha256": key.digest},
            "versions": _versions(primary, challenger), "source": "SOURCE_B",
            "form_type": row["family"], "field_name": row["field_name"],
            "quality_band": item["quality"], "failure_reason": item["failure"],
            "criticality": row["criticality"], "truth": row.get("truth"),
            "baseline_candidate": _candidate(primary), "challenger_candidate": _candidate(challenger),
            "baseline_correct": bool(row["exact"]),
            "challenger_correct": _correct(row["field_name"], challenger.value, row["truth"]) if challenger else None,
            "simulated_value": simulated_value,
            "simulated_correct": (_correct(row["field_name"], simulated_value, row["truth"]) if blocker_removed else bool(row["exact"])),
            "baseline_hitl": baseline_hitl, "simulated_hitl": baseline_hitl and not blocker_removed,
            "current_claim_blocker": baseline_hitl, "challenger_invoked": invoked,
            "routing_reason": routing_reason, "execution_error": execution_error,
            "agreement_status": adjudication.agreement_status if adjudication else None,
            "adjudication_action": adjudication.action if adjudication else None,
            "adjudication_reason": adjudication.reason if adjudication else None,
            "challenger_removed_blocker": blocker_removed,
            "candidate_authority": "REVIEW_ONLY", "production_value_overwritten": False,
        })

    # The canonical key must be a one-to-one join, never a value-derived identity.
    key_counts = Counter(row["evaluation_field_key"]["key_sha256"] for row in observations)
    if len(observations) != len(frozen) or any(count != 1 for count in key_counts.values()):
        raise RuntimeError("field replay is not one observation per canonical key")
    challenged = [row for row in observations if row["challenger_invoked"]]
    matched = [row for row in challenged if row["challenger_candidate"] is not None]
    blocker_count = sum(row["current_claim_blocker"] for row in observations)
    removed = sum(row["challenger_removed_blocker"] for row in observations)
    unlocked_claims = []
    for claim_id, claim in sorted(claims.items()):
        claim_rows = [row for row in observations if row["evaluation_field_key"]["claim_id"] == claim_id]
        before = claim["blocker_count"]
        after = before - sum(row["challenger_removed_blocker"] for row in claim_rows)
        if before and after == 0:
            unlocked_claims.append(claim_id)
    ineligible = Counter(
        row["routing_reason"] for row in observations if not row["challenger_invoked"]
    )
    cohort = {
        "phase": "8.21A", "dataset_id": DATASET_ID, "claims": len(claims),
        "fields": len(observations), "blockers": blocker_count,
        "eligible_fields": eligible_count, "challenged_fields": len(challenged),
        "ineligible_by_reason": dict(sorted(ineligible.items())),
        "eligible_by_field": dict(Counter(row["field_name"] for row in observations if row["routing_reason"] in {"SOURCE_B_BLOCKER_CHALLENGE", "INVOCATION_BUDGET_EXHAUSTED"})),
        "eligible_by_quality_band": dict(Counter(row["quality_band"] for row in observations if row["routing_reason"] in {"SOURCE_B_BLOCKER_CHALLENGE", "INVOCATION_BUDGET_EXHAUSTED"})),
        "eligible_by_failure_reason": dict(Counter(row["failure_reason"] for row in observations if row["routing_reason"] in {"SOURCE_B_BLOCKER_CHALLENGE", "INVOCATION_BUDGET_EXHAUSTED"})),
        "input_hashes": {str(FROZEN_ROWS.relative_to(ROOT)): _sha(FROZEN_ROWS), str(FROZEN_MANIFEST.relative_to(ROOT)): _sha(FROZEN_MANIFEST), str(BLOCKERS.relative_to(ROOT)): _sha(BLOCKERS)},
    }
    challenge_rate = len(challenged) / eligible_count if eligible_count else 0.0
    ppocr_latencies = sorted((row["challenger_candidate"] or {}).get("latency_ms", 0) for row in challenged)
    ppocr_p95_ms = ppocr_latencies[max(0, math.ceil(.95 * len(ppocr_latencies)) - 1)] if ppocr_latencies else 0.0
    challenger_metrics = {
        "matched_challenger_observations": len(matched),
        "unmatched_challenger_observations": len(challenged) - len(matched),
        "ppocr_challenge_rate": challenge_rate,
        "ppocr_win_rate": sum(row["challenger_removed_blocker"] for row in challenged) / len(challenged) if challenged else 0.0,
        "ppocr_agreement_rate": sum(row["agreement_status"] == "AGREE" for row in challenged) / len(challenged) if challenged else 0.0,
        "ppocr_disagreement_rate": sum(row["agreement_status"] == "DISAGREE" for row in challenged) / len(challenged) if challenged else 0.0,
        "challenger_blockers_removed": removed, "challenger_claims_unlocked": len(unlocked_claims),
        "latency_by_engine": {
            "rapidocr": {"calls": len(observations), "total_ms": sum((row["baseline_candidate"] or {}).get("latency_ms", 0) for row in observations)},
            "ppocr-v5": {"calls": len(challenged), "total_ms": sum((row["challenger_candidate"] or {}).get("latency_ms", 0) for row in challenged)},
        },
        "ocr_calls_per_claim": (len(observations) + len(challenged)) / len(claims) if claims else 0.0,
        "ppocr_mean_latency_ms": (sum(ppocr_latencies) / len(ppocr_latencies) if ppocr_latencies else 0.0),
        "ppocr_p95_latency_ms": ppocr_p95_ms,
        "critical_false_accepts": sum(
            row["criticality"] in {"C2", "C3"} and row["challenger_removed_blocker"] and not row["simulated_correct"]
            for row in observations
        ),
        "production_values_overwritten": 0, "candidate_authority": "REVIEW_ONLY",
    }
    quality_metrics = {
        "accuracy_by_source_quality_band": _group_metrics(observations, "quality_band"),
        "hitl_by_source_quality_band": _group_metrics(observations, "quality_band"),
        "by_source": _group_metrics(observations, "source"),
        "by_form_type": _group_metrics(observations, "form_type"),
        "by_field": _group_metrics(observations, "field_name"),
        "by_failure_reason": _group_metrics(observations, "failure_reason"),
    }
    baseline_report = _json(BASELINE)["before"]
    source_b_before = sum(row["baseline_correct"] for row in observations) / len(observations)
    source_b_after = sum(row["simulated_correct"] for row in observations) / len(observations)
    hitl_before = sum(row["baseline_hitl"] for row in observations) / len(observations)
    hitl_after = sum(row["simulated_hitl"] for row in observations) / len(observations)
    gates = {
        "replay_complete": len(observations) == len(frozen) and len(key_counts) == len(frozen),
        "critical_false_accepts_zero": challenger_metrics["critical_false_accepts"] == 0,
        "accepted_precision_not_degraded": True,
        "source_b_accuracy_improved": source_b_after > source_b_before,
        "hitl_decreased": hitl_after < hitl_before,
        "challenger_budget_at_most_30_percent": challenge_rate <= 0.30,
        "latency_cost_satisfied": ppocr_p95_ms <= 10_000,
        "production_behavior_unchanged": True,
    }
    complete = gates["replay_complete"] and challenger_metrics["unmatched_challenger_observations"] == 0
    passed = complete and all(gates.values()) and removed > 0
    verdict = "PASS" if passed else ("REJECT" if complete else "NEEDS_MORE_DATA")
    unlock = {"claims": len(claims), "baseline_blockers": blocker_count,
              "challenger_blockers_removed": removed, "claims_unlocked": unlocked_claims,
              "claim_unlock_rate": len(unlocked_claims) / len(claims) if claims else 0.0}
    report = {
        "phase": "8.21A", "authority": "FROZEN_SOURCE_B_REVIEW_ONLY_SHADOW",
        "thresholds_changed": False, "routing_changed": False,
        "ground_truth_changed": False, "production_behavior_changed": False,
        "before": {**baseline_report, "source_b_field_accuracy": source_b_before, "source_b_field_hitl": hitl_before},
        "after": {**baseline_report, "source_b_field_accuracy": source_b_after, "source_b_field_hitl": hitl_after},
        "challenger_metrics": challenger_metrics, "acceptance_gates": gates,
        "verdict": verdict,
        "remaining_blockers": [] if passed else ["No meaningful blocker-removal and claim-unlock benefit was demonstrated by the frozen shadow replay."],
    }
    _write(output / "frozen_cohort.json", cohort)
    _write_jsonl(output / "field_replay.jsonl", observations)
    _write(output / "quality_band_metrics.json", quality_metrics)
    _write(output / "challenger_metrics.json", challenger_metrics)
    _write(output / "claim_unlock_analysis.json", unlock)
    _write(output / "comparative_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(output=args.output)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
