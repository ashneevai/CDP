"""Run local Tesseract against deterministic synthetic field crops."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from time import monotonic

from PIL import Image

from packages.field_verification import (
    normalize_field_value,
    repair_npi_missing_leading_digit,
    verify_field,
)
from workers.cascade.tesseract_adapter import for_field_type
from workers.document_preparation.preprocessing import deskew, detect_skew_angle
from workers.page_detection.template_alignment import align_to_reference
from workers.page_detection.text_extraction import PaddleOCRTextExtractor, RapidOCRTextExtractor
from workers.retry.alternate_preprocessing import aggressive_contrast, upscale

REFERENCE_CONDITION_PRIORITY = {
    "clean_scan": 0,
    "fax": 1,
    "low_contrast": 2,
    "handwriting": 3,
    "poor_dpi": 4,
    "skew": 5,
    "rotation": 6,
    "cropped_edges": 7,
}


def _select_reference_ids(manifest: dict) -> dict[str, str]:
    """Choose the least geometrically damaged available page per form family."""
    selected: dict[str, tuple[int, str]] = {}
    for document_id, metadata in manifest.items():
        family = metadata["form_type"]
        candidate = (REFERENCE_CONDITION_PRIORITY.get(metadata["condition"], 99), document_id)
        if family not in selected or candidate < selected[family]:
            selected[family] = candidate
    return {family: candidate[1] for family, candidate in selected.items()}


def _should_register(skew_degrees: float) -> bool:
    """Avoid homography after a mild affine/shear correction already localized well.

    Large rotation and near-axis-aligned pages still benefit from canonical
    registration.  The bounded middle band is kept on the cheaper deskewed
    path because a second projective warp can displace otherwise valid crops.
    """
    magnitude = abs(skew_degrees)
    return magnitude < 0.20 or magnitude > 1.00


def _retry_variant(field_type: str, image: Image.Image) -> Image.Image:
    if field_type == "code":
        return upscale(image, 3)
    if field_type in {"npi", "date", "currency", "tax_id"}:
        return aggressive_contrast(upscale(image, 2))
    return aggressive_contrast(image)


def _candidate_score(
    value: str | None, confidence: float, field_name: str
) -> tuple[int, float, int]:
    evidence = verify_field(field_name, value)
    strength = {"NONE": 0, "PRESENCE": 1, "SYNTAX": 2, "SEMANTIC": 2, "CHECKSUM": 3}[
        evidence.strength
    ]
    return (strength if evidence.valid else 0, confidence, len(normalize_field_value(value)))


def _should_retry_field(field_type: str, confidence: float) -> bool:
    # NPI retains an independent read for checksum-backed consensus. Code
    # crops receive one bounded retry only when the first read is weak; this
    # recovers thin/italic leading glyphs without expanding full-page OCR.
    return (
        field_type == "npi"
        or (field_type == "code" and confidence < 0.85)
        or (field_type == "date" and confidence < 0.70)
    )


def _extract_crop(extractor, crop: Image.Image):
    if hasattr(extractor, "extract_region") and not hasattr(extractor, "psm"):
        return extractor.extract_region(crop, 0, 0, crop.width, crop.height)
    return extractor.extract(crop)


def _norm(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation_data/synthetic_public_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/synthetic_public_v1")
    )
    parser.add_argument("--page-registration", action="store_true")
    parser.add_argument("--field-routing", action="store_true")
    parser.add_argument(
        "--member-id-engine",
        choices=("tesseract", "rapidocr", "paddleocr"),
        default="tesseract",
        help="isolated insured_id_number engine experiment",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = json.loads((args.dataset / "ground_truth.json").read_text("utf-8"))["documents"]
    manifest = json.loads((args.dataset / "document_manifest.json").read_text("utf-8"))
    reference_ids = _select_reference_ids(manifest)
    reference_pages = {
        family: Image.open(args.dataset / manifest[document_id]["file_name"]).convert("RGB")
        for family, document_id in reference_ids.items()
    }
    predictions = []
    counters: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    latencies = []
    registration: defaultdict[str, int] = defaultdict(int)
    verification: defaultdict[str, int] = defaultdict(int)
    optional_extractors = {
        "rapidocr": RapidOCRTextExtractor(),
        "paddleocr": PaddleOCRTextExtractor(),
    }
    type_map = {
        "patient_dob": "date",
        "provider_npi": "npi",
        "total_charge": "currency",
        "total_charges": "currency",
        "type_of_bill": "code",
        "principal_diagnosis": "code",
        "federal_tax_no": "tax_id",
        "insured_id_number": "code",
        "patient_name": "text",
    }
    for document in truth:
        document_id = document["document_id"]
        document_meta = manifest[document_id]
        with Image.open(args.dataset / document_meta["file_name"]) as source:
            page = source.convert("RGB")
        skew_degrees = detect_skew_angle(page)
        localized_page = deskew(page, skew_degrees)
        registration_method = "DISABLED"
        if args.page_registration and _should_register(skew_degrees):
            aligned = align_to_reference(localized_page, reference_pages[document["form_type"]])
            registration_method = aligned.method
            registration["attempted"] += 1
            if aligned.success and aligned.warped is not None:
                localized_page = aligned.warped
                registration["accepted"] += 1
            else:
                registration["rejected"] += 1
        elif args.page_registration:
            registration_method = "SKIPPED_MILD_AFFINE_SKEW"
            registration["skipped_mild_affine_skew"] += 1
        predicted_fields = []
        for field in document["fields"]:
            name = field["field_name"]
            crop_box = tuple(document_meta["crop_boxes"][name])
            started = monotonic()
            crop = localized_page.crop(crop_box)
            field_type = type_map.get(name, "text")
            selected_engine = args.member_id_engine if name == "insured_id_number" else "tesseract"
            extractor = (
                optional_extractors[selected_engine]
                if selected_engine != "tesseract"
                else for_field_type(field_type)
            )
            attempts = [_extract_crop(extractor, crop)]
            original_confidence = (
                sum(word.confidence for word in attempts[0]) / len(attempts[0])
                if attempts[0]
                else 0.0
            )
            if args.field_routing and _should_retry_field(field_type, original_confidence):
                attempts.append(_extract_crop(extractor, _retry_variant(field_type, crop)))
            latencies.append((monotonic() - started) * 1000)
            candidates = []
            for words in attempts:
                candidate_value = " ".join(word.text for word in words).strip() or None
                if field_type == "npi":
                    candidate_value = (
                        repair_npi_missing_leading_digit(candidate_value) or candidate_value
                    )
                confidence = sum(word.confidence for word in words) / len(words) if words else 0.0
                candidates.append((candidate_value, confidence))
            value, confidence = max(
                candidates, key=lambda item: _candidate_score(item[0], item[1], name)
            )
            agreement = sum(
                normalize_field_value(candidate[0]) == normalize_field_value(value)
                for candidate in candidates
            )
            evidence = verify_field(name, value, independent_agreement=agreement)
            correct = _norm(value) == _norm(field["expected_raw"])
            accepted = evidence.auto_verifiable
            verification["accepted"] += int(accepted)
            verification["false_accepts"] += int(accepted and not correct)
            verification["retried"] += int(len(attempts) > 1)
            for key in (
                "overall",
                f"family:{document['form_type']}",
                f"condition:{manifest[document_id]['condition']}",
                f"field:{name}",
            ):
                counters[key][1] += 1
                counters[key][0] += int(correct)
            predicted_fields.append(
                {
                    "field_name": name,
                    "raw_value": value,
                    "expected": field["expected_raw"],
                    "correct": correct,
                    "accepted": accepted,
                    "engine": selected_engine,
                    "confidence": confidence,
                    "attempts": len(attempts),
                    "verification_strength": evidence.strength,
                    "verification_reason": evidence.reason_code,
                }
            )
        predictions.append(
            {
                "document_id": document_id,
                "fields": predicted_fields,
                "skew_degrees": skew_degrees,
                "registration_method": registration_method,
            }
        )
    metrics: dict[str, object] = {
        key: {
            "correct": value[0],
            "total": value[1],
            "accuracy": value[0] / value[1] if value[1] else 0,
        }
        for key, value in sorted(counters.items())
    }
    sorted_latency = sorted(latencies)
    metrics["runtime"] = {
        "calls": len(latencies),
        "p95_latency_ms": sorted_latency[int(0.95 * (len(sorted_latency) - 1))],
        "mean_latency_ms": sum(latencies) / len(latencies),
    }
    metrics["qualification"] = {
        "synthetic_only": True,
        "production_holdout": False,
        "production_qualified": False,
        "false_accepts": verification["false_accepts"],
        "note": "Synthetic accuracy cannot qualify production behavior",
    }
    metrics["registration"] = {
        "enabled": args.page_registration,
        "reference_document_ids": reference_ids,
        **registration,
    }
    accepted = verification["accepted"]
    false_accepts = verification["false_accepts"]
    field_calls = len(latencies)
    metrics["field_routing"] = {
        "enabled": args.field_routing,
        **verification,
        "safe_field_coverage": accepted / field_calls if field_calls else 0.0,
        "field_hitl_proxy": 1 - accepted / field_calls if field_calls else 0.0,
        "accepted_precision": ((accepted - false_accepts) / accepted if accepted else None),
        "acceptance_policy": (
            "Only independently agreeing checksum-valid values may be auto-accepted"
        ),
    }
    (args.output / "predictions.json").write_text(
        json.dumps({"documents": predictions}, indent=2), "utf-8"
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "overall": metrics["overall"],
                "runtime": metrics["runtime"],
                "field_routing": metrics["field_routing"],
                "qualification": metrics["qualification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
