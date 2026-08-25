"""Unchanged, truth-blind Phase-6 production-representative holdout runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image

from evaluation.audit_production_holdout_v2 import DEFAULT_DATASET, DEFAULT_OUTPUT, audit
from packages.claim_decision import ClaimDecisionContext, ClaimDecisionService
from packages.criticality import CriticalityPolicy, DEFAULT_CRITICALITY_PATH
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.evidence_decision import DecisionContext, EvidenceDecisionService, FieldDisposition
from packages.extraction_geometry import FormIdentityDecision, FormIdentityStatus
from packages.field_verification import verify_field
from packages.layout_intelligence import BundleDLayoutEngine
from packages.ocr.contracts import OCRCandidate
from packages.page_observation import PageObservationService
from packages.templates.registry import DEFAULT_TEMPLATE_DIR, TemplateRegistry
from workers.cascade.tesseract_adapter import TesseractTextExtractor
from workers.cascade.isolated_ocr import IsolatedTextExtractor, OCRTimeoutError
from workers.document_preparation.preprocessing import (
    apply_orientation, denoise, deskew, detect_orientation, detect_skew_angle,
)
from workers.page_detection.router import PageRoutingService
from workers.page_detection.text_extraction import (
    RapidOCRFullPageTextExtractor,
    RapidOCRTextExtractor,
)
from workers.standard_form_extraction.consumer import _resolve_geometry
from workers.standard_form_extraction.extractor import StandardFormExtractionService
from workers.standard_form_extraction.processing import StandardFormProcessingService


ACTUAL_TO_TRUTH = {
    "patient_dob": "dob", "insured_id_number": "member_id",
    "federal_tax_id": "federal_tax_no", "patient_account_no": "account_no",
    "patient_control_number": "account_no", "patient_sex": "sex",
    "diagnosis_codes": "diagnosis", "insured_unique_id": "member_id",
    "provider_name_address": "provider_name", "billing_provider_info": "provider_name",
    "rel_code": "relationship",
}
TRUTH_ROUTE = {
    "CMS1500_0212": "CMS1500", "UB04_CMS1450_COMPAT": "UB04",
    "CUSTOM_PROFESSIONAL_CLAIM": "UNKNOWN_STRUCTURED",
    "CLAIM_ATTACHMENT": "UNKNOWN_UNSTRUCTURED", "NON_CLAIM": "NON_CLAIM",
}
CLAIM_FAMILIES = {"CMS1500_0212", "UB04_CMS1450_COMPAT", "CUSTOM_PROFESSIONAL_CLAIM"}


def _norm(value) -> str:
    return re.sub(r"[^A-Z0-9.]", "", str(value or "").upper())


def _p(values: list[float], percentile: float):
    return sorted(values)[max(0, math.ceil(percentile * len(values))-1)] if values else None


def _prepare(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    image = apply_orientation(image, detect_orientation(image))
    image = deskew(image, detect_skew_angle(image))
    return denoise(image)


def _prepare_profiled(path: Path) -> tuple[Image.Image, dict[str, float]]:
    stages: dict[str, float] = {}
    started = time.perf_counter()
    image = Image.open(path).convert("L")
    image.load()
    stages["image_decode"] = time.perf_counter() - started
    started = time.perf_counter()
    orientation = detect_orientation(image)
    stages["orientation_detection"] = time.perf_counter() - started
    started = time.perf_counter()
    image = apply_orientation(image, orientation)
    stages["orientation_apply"] = time.perf_counter() - started
    started = time.perf_counter()
    skew = detect_skew_angle(image)
    stages["skew_detection"] = time.perf_counter() - started
    started = time.perf_counter()
    image = deskew(image, skew)
    stages["deskew"] = time.perf_counter() - started
    started = time.perf_counter()
    image = denoise(image)
    stages["denoise"] = time.perf_counter() - started
    return image, stages


@lru_cache(maxsize=1)
def _runtime_components():
    """Load immutable configuration and model-backed services once per process."""
    registry = TemplateRegistry.load_from_directory(DEFAULT_TEMPLATE_DIR)
    cms, ub = registry.get("cms1500", "02-12"), registry.get("ub04", "2014")
    page_ocr = TesseractTextExtractor(psm=11)
    router = PageRoutingService(
        cms, ub, page_ocr, registry.load_reference_image(cms),
        registry.load_reference_image(ub),
    )
    thread_cap = int(os.environ.get("CDP_INTERNAL_THREAD_CAP", "0") or 0)
    rapid_kwargs = (
        {"intra_op_num_threads": thread_cap, "inter_op_num_threads": 1}
        if thread_cap > 0 else {}
    )
    regional = RapidOCRTextExtractor(**rapid_kwargs)
    observation = PageObservationService(
        RapidOCRFullPageTextExtractor(**rapid_kwargs),
        preprocessing_version="phase9b-production-prepared-v1",
    )
    standard_processing = StandardFormProcessingService(
        observation,
        StandardFormExtractionService(regional),
    )
    return (
        registry,
        router,
        standard_processing,
        IsolatedTextExtractor(
            "workers.page_detection.text_extraction",
            "PaddleOCRTextExtractor",
            provider_kwargs={"cpu_threads": thread_cap or 2},
            timeout_seconds=float(os.environ.get("CDP_OCR_TIMEOUT_SECONDS", "30")),
            engine_name="paddleocr",
            model_name="PP-OCRv4",
            model_version="paddleocr-2.x",
        ),
        BundleDLayoutEngine(),
        EvidenceDecisionService(route_mode="runtime"),
        CriticalityPolicy.load(DEFAULT_CRITICALITY_PATH),
        ClaimDecisionService.load(),
    )


def infer(dataset: Path, output: Path, limit: int | None = None, offset: int = 0,
          sample_size: int | None = None, sample_seed: int = 62026) -> list[dict]:
    metadata = [json.loads(line) for line in (dataset / "metadata/document_metadata.jsonl").read_text("utf-8").splitlines()]
    # Family, quality, rotation and all other evaluation metadata are removed
    # before any runtime component sees the document.
    inputs = [{"document_id": row["document_id"], "path": row["path"]} for row in metadata]
    if sample_size is not None:
        if sample_size > len(inputs):
            raise ValueError("sample size exceeds dataset size")
        inputs = random.Random(sample_seed).sample(inputs, sample_size)
        output.mkdir(parents=True, exist_ok=True)
        (output / "sample_manifest.json").write_text(json.dumps({
            "method": "simple_random_without_replacement",
            "seed": sample_seed, "sample_size": sample_size,
            "document_ids": [item["document_id"] for item in inputs],
            "truth_used_for_selection": False,
        }, indent=2), "utf-8")
    inputs = inputs[offset:offset + limit if limit else None]
    registry, router, standard_processing, paddle, layout, decisions, criticality, claims = (
        _runtime_components()
    )
    predictions = []
    for position, item in enumerate(inputs, 1):
        stage, counters = {}, Counter()
        wall0, cpu0 = time.perf_counter(), time.process_time()
        started = time.perf_counter()
        image, preparation_stages = _prepare_profiled(dataset / item["path"])
        stage.update(preparation_stages)
        stage["preparation"] = time.perf_counter()-started
        started = time.perf_counter(); routed = router.route_single_page(image); stage["classification"] = time.perf_counter()-started
        fields, field_decisions, route, schema = {}, [], None, None
        alignment_method, alignment_accepted, registration_confidence = None, False, None
        table_payload = None
        if routed.template is not None:
            template = routed.template
            route = "CMS1500" if template.template_id == "cms1500" else "UB04"
            resized = image.resize((template.reference_dimensions.width_px,
                                    template.reference_dimensions.height_px))
            started = time.perf_counter()
            identity = FormIdentityDecision(
                family=(DocumentClass.CMS1500 if template.template_id == "cms1500"
                        else DocumentClass.UB04),
                status=FormIdentityStatus.VERIFIED,
                score=routed.page_scores[1].confidence,
                template_version=template.version,
                supporting_evidence=("CANONICAL_RUNTIME_ROUTER",),
            )
            ready, geometry = _resolve_geometry(
                resized, template, registry.load_reference_image(template), identity)
            registration = geometry.registration
            alignment_method = (
                registration.algorithm if registration is not None
                else geometry.mode.value.lower()
            )
            stage["registration"] = time.perf_counter()-started
            alignment_accepted = geometry.authorizes_fixed_roi
            registration_confidence = registration.alignment_confidence if registration else None
            started = time.perf_counter()
            processing = (
                standard_processing.process(
                    ready,
                    template,
                    1,
                    identity,
                    page_id=item["document_id"],
                    registered_geometry=geometry,
                )
                if ready is not None and alignment_accepted else None
            )
            extracted = processing.fields if processing is not None else []
            stage["ocr"] = time.perf_counter()-started
            if processing is not None:
                counters["rapidocr_calls"] += (
                    processing.diagnostics.full_page_ocr_calls
                    + processing.diagnostics.regional_ocr_calls
                )
                counters["full_page_ocr_calls"] += processing.diagnostics.full_page_ocr_calls
                counters["regional_ocr_calls"] += processing.diagnostics.regional_ocr_calls
            source_candidates = []
            for field in extracted:
                canonical = ACTUAL_TO_TRUTH.get(field.field_name, field.field_name)
                value = field.normalized_value or field.raw_value
                if canonical not in fields or field.confidence > fields[canonical]["confidence"]:
                    fields[canonical] = {"value": value, "raw": field.raw_value,
                                         "confidence": field.confidence,
                                         "bbox": field.bounding_box.model_dump(mode="json"),
                                         "source_field": field.field_name}
                source_candidates.append((canonical, value, field.confidence, field.bounding_box, None))
            schema = route
        else:
            counters["paddleocr_calls"] += 1
            started = time.perf_counter()
            try:
                tokens = paddle.extract(image)
            except OCRTimeoutError:
                stage["ocr"] = time.perf_counter()-started
                counters["ocr_timeouts"] += 1
                route, schema = "UNKNOWN_UNSTRUCTURED", "UNKNOWN_UNSTRUCTURED"
                source_candidates = []
            else:
                stage["ocr"] = time.perf_counter()-started
                started = time.perf_counter(); result = layout.extract(
                    tokens, page_number=1, width=image.width, height=image.height,
                    engine=paddle.engine_name); stage["layout"] = time.perf_counter()-started
                route, schema = result.route.value, result.schema_evidence.schema_family
                table_payload = result.table.model_dump(mode="json")
                source_candidates = []
                for canonical, candidates in result.candidates.items():
                    best = candidates[0]
                    fields[canonical] = {"value": best.value, "raw": best.value,
                                         "confidence": best.confidence,
                                         "bbox": best.bbox.model_dump(mode="json"),
                                         "label": best.original_label,
                                         "alias": best.matched_alias,
                                         "relationship": best.relationship_evidence.relationship}
                    source_candidates.append((canonical, best.value, best.confidence, best.bbox,
                                              best.relationship_evidence.relationship))
        started = time.perf_counter()
        for canonical, value, confidence, bbox, structural in source_candidates:
            verification = verify_field(canonical, value)
            candidate = OCRCandidate(
                value=value, raw_value=value or "", engine=(paddle.engine_name if routed.template is None else "rapidocr"),
                model_name=(paddle.model_name if routed.template is None else "RapidOCR-ONNX"),
                model_version=(paddle.model_version if routed.template is None else "rapidocr-onnxruntime"),
                preprocessing_variant="production_prepared", raw_confidence=confidence,
                calibrated_confidence=None, bounding_box=bbox, latency_ms=0,
                validation_results=(verification.reason_code,),
                evidence_reference=f"layout:{structural}" if structural else "template:regional",
                registration_confidence=registration_confidence,
            )
            decision = decisions.decide(DecisionContext(
                field_name=canonical, document_family=schema,
                criticality=criticality.for_field(canonical), candidates=[candidate],
                deterministic_evidence={verification.reason_code} if verification.valid else set(),
                hard_validation_passed=verification.valid,
                registration_confidence=registration_confidence,
                structural_evidence_source=structural,
            ))
            field_decisions.append(decision)
            fields[canonical]["decision"] = decision.model_dump(mode="json")
        stage["evidence"] = time.perf_counter()-started
        started = time.perf_counter()
        claim = claims.decide(ClaimDecisionContext(
            claim_id=item["document_id"], document_family=schema,
            field_decisions=field_decisions,
            registration_integrity_valid=(alignment_accepted if routed.template is not None else True),
            enforce_configured_required_fields=True,
        )) if route not in {"NON_CLAIM", "UNKNOWN_UNSTRUCTURED"} else None
        stage["claim_decision"] = time.perf_counter()-started
        predictions.append({
            "document_id": item["document_id"], "route": route, "schema": schema,
            "route_reasons": routed.reason_codes, "alignment_method": alignment_method,
            "alignment_accepted": alignment_accepted,
            "registration_confidence": registration_confidence,
            "fields": fields, "claim_decision": claim.model_dump(mode="json") if claim else None,
            "table": table_payload, "stage_seconds": stage,
            "wall_seconds": time.perf_counter()-wall0, "cpu_seconds": time.process_time()-cpu0,
            "counters": dict(counters), "cloud_cost_usd": 0,
        })
        print(f"[{position}/{len(inputs)}] {item['document_id']} {route}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("predictions.json" if limit is None and offset == 0
                     else f"predictions_{offset}_{len(inputs)}.json")
    path.write_text(json.dumps(predictions, indent=2), "utf-8")
    return predictions


def score(dataset: Path, output: Path, predictions: list[dict], limit: int | None = None) -> dict:
    truth = {item["document_id"]: item for item in
             (json.loads(line) for line in (dataset / "ground_truth/ground_truth.jsonl").read_text("utf-8").splitlines())}
    metadata = {item["document_id"]: item for item in
                (json.loads(line) for line in (dataset / "metadata/document_metadata.jsonl").read_text("utf-8").splitlines())}
    counts, routes, errors = Counter(), defaultdict(Counter), []
    by_family, by_quality, disposition_counts, claim_counts = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    latencies, cpu_times, stage_times = [], [], defaultdict(list)
    for prediction in predictions:
        expected, meta = truth[prediction["document_id"]], metadata[prediction["document_id"]]
        family, quality = expected["family"], meta["quality_bucket"]
        truth_route, predicted_route = TRUTH_ROUTE[family], prediction["route"]
        routes[family][predicted_route] += 1
        route_correct = predicted_route == truth_route or (
            family == "CLAIM_ATTACHMENT" and predicted_route in {"UNKNOWN_UNSTRUCTURED", "UNKNOWN_STRUCTURED"})
        counts["routing_total"] += 1; counts["routing_correct"] += int(route_correct)
        by_family[family]["documents"] += 1; by_family[family]["routing_correct"] += int(route_correct)
        by_quality[quality]["documents"] += 1; by_quality[quality]["routing_correct"] += int(route_correct)
        expected_fields = {key: value for key, value in expected["fields"].items() if key != "service_lines"}
        for name, expected_value in expected_fields.items():
            actual = prediction["fields"].get(name)
            correct = bool(actual) and _norm(actual["value"]) == _norm(expected_value)
            counts["fields_total"] += 1; counts["fields_correct"] += int(correct)
            by_family[family]["fields_total"] += 1; by_family[family]["fields_correct"] += int(correct)
            by_quality[quality]["fields_total"] += 1; by_quality[quality]["fields_correct"] += int(correct)
            critical = CriticalityPolicy.load(DEFAULT_CRITICALITY_PATH).for_field(name).value in {"C2", "C3"}
            counts["critical_total"] += int(critical); counts["critical_correct"] += int(critical and correct)
            if actual and actual.get("decision"):
                disposition = actual["decision"]["disposition"]
                disposition_counts[disposition] += 1
                accepted = disposition in {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED"}
                counts["accepted"] += int(accepted); counts["review"] += int(not accepted)
                counts["false_accepts"] += int(accepted and not correct)
                counts["critical_false_accepts"] += int(critical and accepted and not correct)
            else:
                counts["review"] += 1
            if not correct:
                if not route_correct:
                    cause = "ROUTING_ERROR"
                elif not actual and predicted_route.startswith("UNKNOWN"):
                    cause = "LABEL_NOT_DETECTED"
                elif not actual:
                    cause = "VALUE_NOT_DETECTED"
                else:
                    cause = "OCR_TOKEN_ERROR"
                errors.append({"document_id": prediction["document_id"], "family": family,
                               "quality": quality, "route": predicted_route, "field": name,
                               "truth": expected_value, "prediction": actual["value"] if actual else None,
                               "disposition": actual.get("decision", {}).get("disposition") if actual else None,
                               "root_cause": cause, "critical": critical,
                               "image": meta["path"]})
        if prediction["claim_decision"]:
            claim_counts[prediction["claim_decision"]["disposition"]] += 1
        latencies.append(prediction["wall_seconds"]); cpu_times.append(prediction["cpu_seconds"])
        for name, value in prediction["stage_seconds"].items(): stage_times[name].append(value)
    ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
    summary = {
        "qualification": {"untouched_first_run": True, "tuning_prohibited": True,
                          "cloud_ai_enabled": False, "docling_enabled": False,
                          "production_authority": "SHADOW_READINESS_ONLY"},
        "routing": {"accuracy": ratio("routing_correct", "routing_total"),
                    "confusion_matrix": {family: dict(values) for family, values in routes.items()}},
        "extraction": {"field_exact_match": ratio("fields_correct", "fields_total"),
                       "critical_field_exact_match": ratio("critical_correct", "critical_total"),
                       "correct": counts["fields_correct"], "total": counts["fields_total"]},
        "decision": {"dispositions": dict(disposition_counts),
                     "safe_coverage": ratio("accepted", "fields_total"),
                     "field_hitl": ratio("review", "fields_total"),
                     "false_accepts": counts["false_accepts"],
                     "critical_false_accepts": counts["critical_false_accepts"]},
        "claim": {"dispositions": dict(claim_counts),
                  "claim_stp": sum(value for key, value in claim_counts.items() if key.startswith("STP_")) / sum(claim_counts.values()) if claim_counts else None,
                  "claim_hitl": sum(value for key, value in claim_counts.items() if "REVIEW" in key) / sum(claim_counts.values()) if claim_counts else None},
        "latency": {"mean": statistics.fmean(latencies), "p50": _p(latencies,.5),
                    "p95": _p(latencies,.95), "p99": _p(latencies,.99),
                    "stage_mean": {key: statistics.fmean(values) for key, values in stage_times.items()}},
        "cost": {"cloud_cost_usd": 0, "mean_cpu_seconds": statistics.fmean(cpu_times),
                 "paddleocr_calls": sum(item["counters"].get("paddleocr_calls",0) for item in predictions),
                 "rapidocr_calls": sum(item["counters"].get("rapidocr_calls",0) for item in predictions),
                 "docling_calls": 0, "ai_calls": 0, "retry_count": 0},
        "by_family": {key: {"routing_accuracy": value["routing_correct"]/value["documents"],
                            "raw_accuracy": value["fields_correct"]/value["fields_total"] if value["fields_total"] else None}
                      for key, value in by_family.items()},
        "by_quality": {key: {"routing_accuracy": value["routing_correct"]/value["documents"],
                             "raw_accuracy": value["fields_correct"]/value["fields_total"] if value["fields_total"] else None}
                       for key, value in by_quality.items()},
        "error_taxonomy": dict(Counter(item["root_cause"] for item in errors)),
        "errors": errors,
    }
    report_path = output / ("baseline_report.json" if limit is None else f"baseline_report_pilot_{limit}.json")
    report_path.write_text(json.dumps(summary, indent=2), "utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=62026)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); audit(args.dataset, args.output)
    predictions = infer(args.dataset, args.output, args.limit, args.offset,
                        args.sample_size, args.sample_seed)
    if args.no_score:
        print(json.dumps({"documents": len(predictions), "offset": args.offset}))
        return 0
    report = score(args.dataset, args.output, predictions, args.limit)
    print(json.dumps({key: report[key] for key in ("qualification","routing","extraction","decision","claim","latency","cost")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
