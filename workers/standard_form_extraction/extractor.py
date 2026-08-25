"""Standard-form candidate materialization.

The Phase 8 default consumes dynamic ROIs and tokens from one canonical
full-page observation without another OCR call. Registered template-region
OCR remains an optional compatibility-proven fast path for known lineages.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256

from packages.domain.claim import ServiceLine
from packages.domain.common import BoundingBox
from packages.domain.enums import ExtractionMethod, ValidationStatus
from packages.domain.extraction import ExtractedField, FieldEvidence
from packages.extraction_geometry import ExtractionGeometryDecision, ExtractionGeometryMode
from packages.extraction_recovery import select_field_span
from packages.field_localization import FieldDefinition
from packages.local_evidence_cascade import decide_local_candidate
from packages.ocr.independence import independence_group
from packages.ocr.provenance import EvidenceProvenance
from packages.page_observation import PageObservation, line_clustered_reading_order
from packages.roi_resolution import ROIResolutionMode, ROIResolutionResult
from packages.templates.models import FieldRegion, Template
from workers.page_detection.text_extraction import TextExtractor
from workers.standard_form_extraction.field_processors import normalize
from workers.table_extraction import UB04ServiceLineExtractor

REGION_PADDING_PX = 4
REGION_COALESCE_TOLERANCE_PX = 8


def _apply_postprocessor(raw_text: str, postprocessor: str | None) -> str:
    """Apply deterministic template semantics before type normalization."""
    if postprocessor not in {"person_name_first", "person_name_last"}:
        return raw_text
    parts = [part.strip() for part in re.split(r"\s*,\s*|\s+", raw_text.strip()) if part.strip()]
    if len(parts) < 2:
        return raw_text
    return parts[0] if postprocessor == "person_name_last" else parts[1]


def _region_bounds(image, region: FieldRegion | tuple) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (
        (region.x0, region.y0, region.x1, region.y1) if isinstance(region, FieldRegion) else region
    )
    padding = region.padding_px if isinstance(region, FieldRegion) else REGION_PADDING_PX
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(image.width, x1 + padding),
        min(image.height, y1 + padding),
    )


def _region_text(extractor: TextExtractor, image, region: FieldRegion | tuple) -> tuple[str, float]:
    """Returns (joined text, mean per-line OCR confidence -- 0.0 if the
    region has no lines), matching the averaging approach already used by
    `workers.retry.retry_service._combine_lines`."""
    x0, y0, x1, y1 = _region_bounds(image, region)
    lines = extractor.extract_region(image, x0, y0, x1, y1)
    ordered = line_clustered_reading_order(lines)
    text = " ".join(line.text for line in ordered)
    confidence = sum(line.confidence for line in ordered) / len(ordered) if ordered else 0.0
    return text, confidence


def _compact_alnum(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _crop_sha256(image, bbox: tuple[int, int, int, int]) -> str | None:
    if image is None:
        return None
    return sha256(image.crop(bbox).convert("RGB").tobytes()).hexdigest()


def _reconcile_secondary_name(primary: str, secondary: str) -> str:
    """Drop one isolated OCR artifact only when primary evidence proves it extraneous."""
    if _compact_alnum(primary) == _compact_alnum(secondary):
        return secondary
    parts = secondary.split()
    for index, part in enumerate(parts):
        if len(part) == 1:
            candidate = " ".join(parts[:index] + parts[index + 1 :])
            if _compact_alnum(primary) == _compact_alnum(candidate):
                return candidate
    return secondary


def _clean_secondary_name(value: str, *, split_md: bool = True) -> str:
    """Remove a duplicated OCR initial only when its following word agrees."""
    value = re.sub(r"[.:'\u00b7\uff1a\ufffd]+", " ", value)
    value = value.upper()
    if split_md:
        value = re.sub(r"(?<=[A-Z])MD$", " MD", value)
    parts = value.split()
    return " ".join(
        part
        for index, part in enumerate(parts)
        if not (
            len(part) == 1
            and index + 1 < len(parts)
            and parts[index + 1].upper().startswith(part.upper())
        )
    )


def _make_field(
    template: Template,
    field_name: str,
    field_type: str,
    raw_text: str,
    confidence: float,
    region_x0: int,
    region_y0: int,
    region_x1: int,
    region_y1: int,
    page_number: int,
    image_width: int,
    image_height: int,
    extraction_method: ExtractionMethod = ExtractionMethod.REGIONAL_PADDLEOCR,
    postprocessor: str | None = None,
) -> ExtractedField:
    raw_text = _apply_postprocessor(raw_text, postprocessor)
    normalized_value, ok = normalize(field_type, raw_text)
    if not (normalized_value or "").strip():
        confidence = 0.0
        ok = False
    return ExtractedField(
        field_name=field_name,
        raw_value=raw_text,
        normalized_value=normalized_value,
        confidence=confidence,
        page_number=page_number,
        bounding_box=BoundingBox(
            x0=region_x0,
            y0=region_y0,
            x1=region_x1,
            y1=region_y1,
            image_width=image_width,
            image_height=image_height,
        ),
        extraction_method=extraction_method,
        template_version=f"{template.template_id}@{template.version}",
        validation_status=ValidationStatus.PENDING if ok else ValidationStatus.INVALID,
        validation_reasons=[]
        if ok
        else [
            "required_or_unvalidated_blank"
            if not (normalized_value or "").strip()
            else "normalization_failed"
        ],
    )


class StandardFormExtractionService:
    def __init__(self, text_extractor: TextExtractor) -> None:
        self._text_extractor = text_extractor
        # The engine is long-lived with the worker.  A UB table is OCRed once;
        # row/column reconstruction consumes those tokens without cell OCR.
        self._ub04_service_lines = UB04ServiceLineExtractor(text_extractor)
        self.last_field_ocr_cost: dict[str, int | float] = {}
        self.last_candidate_trace: dict[str, dict] = {}

    def extract_fields_from_observation(
        self,
        observation: PageObservation,
        template: Template,
        page_number: int,
        roi_results: dict[str, ROIResolutionResult],
        field_definitions: dict[str, FieldDefinition] | None = None,
        image=None,
    ) -> list[ExtractedField]:
        """Extract candidates from the one canonical full-page OCR observation.

        Dynamic anchor and structural regions are first-class. This method
        performs no OCR call; higher-resolution regional OCR belongs to the
        selective secondary-evidence policy.
        """
        region_by_name = {region.field_name: region for region in template.field_regions}
        fields: list[ExtractedField] = []
        traces: dict[str, dict] = {}
        for name, resolved in roi_results.items():
            if resolved.bbox is None or resolved.mode == ROIResolutionMode.UNRESOLVED:
                continue
            region = region_by_name.get(name)
            definition = (field_definitions or {}).get(name)
            if region is None and definition is None:
                continue
            x0, y0, x1, y1 = resolved.bbox
            tokens = [
                token
                for token in observation.ocr_tokens
                if x0 <= (token.bbox[0] + token.bbox[2]) / 2 <= x1
                and y0 <= (token.bbox[1] + token.bbox[3]) / 2 <= y1
            ]
            ordered = line_clustered_reading_order(tokens)
            text = " ".join(token.text for token in ordered)
            primary_raw = text
            confidence = (
                sum(token.confidence for token in ordered) / len(ordered) if ordered else 0.0
            )
            type_map = {
                "DATE": "date",
                "CURRENCY": "currency",
                "NPI": "npi",
                "CHECKBOX": "checkbox",
                "ALPHANUMERIC_ID": "code",
                "CPT_HCPCS": "code",
                "ICD_CODE": "code",
                "TYPE_OF_BILL": "code",
                "TAX_IDENTIFIER": "tax_id",
            }
            field_type = region.field_type if region else type_map.get(definition.datatype, "text")
            if definition is not None and definition.datatype in {
                "PERSON_NAME",
                "PERSON_OR_ORGANIZATION",
            }:
                text = _clean_secondary_name(text, split_md=False)
            if definition is not None and definition.datatype == "ALPHANUMERIC_ID":
                identifiers = re.findall(r"(?:MBR|MEM|PLN)-[A-Z0-9]+", text.upper())
                if len(identifiers) == 1:
                    text = identifiers[0]
            if definition is not None and definition.datatype not in {
                "PERSON_NAME",
                "PERSON_OR_ORGANIZATION",
                "CHECKBOX",
            }:
                valid_tokens = [
                    token
                    for token in ordered
                    if decide_local_candidate(token.text, definition.datatype).accepted
                ]
                if len(valid_tokens) == 1:
                    text = valid_tokens[0].text
                    confidence = valid_tokens[0].confidence
            primary_span = (
                select_field_span(text, definition.datatype, name)
                if definition is not None
                else None
            )
            if primary_span is not None:
                text = primary_span.selected_text
            # Template name postprocessors encode fixed-form cell semantics and
            # must not reinterpret already ordered dynamic observation text.
            postprocessor = region.postprocessor if region and definition is None else None
            secondary_invoked = False
            regional_text = None
            regional_confidence = None
            regional = None
            regional_span = None
            regional_trigger_reason = None
            primary = decide_local_candidate(text, definition.datatype) if definition else None
            if definition is not None and image is not None:
                # A second pass through the same RapidOCR family cannot add
                # independent evidence for an NPI checksum failure. Golden
                # evaluation showed zero resolutions across 90 such calls;
                # retain the candidate for safe deterministic rejection.
                secondary_eligible = definition.datatype != "NPI"
                if not primary.accepted and secondary_eligible:
                    regional_trigger_reason = (
                        "NO_PRIMARY_TOKEN" if not text.strip()
                        else "DETERMINISTIC_VALIDATION_FAILURE"
                    )
                    if hasattr(self._text_extractor, "set_context"):
                        self._text_extractor.set_context(
                            field=name, reason=regional_trigger_reason
                        )
                    regional_text, regional_confidence = _region_text(
                        self._text_extractor, image, resolved.bbox
                    )
                    regional_span = select_field_span(
                        regional_text, definition.datatype, name
                    )
                    regional_text = regional_span.selected_text
                    regional = decide_local_candidate(regional_text, definition.datatype)
                    secondary_invoked = True
                    if definition.datatype in {"PERSON_NAME", "PERSON_OR_ORGANIZATION"}:
                        regional_text = _clean_secondary_name(regional_text)
                        regional_text = _reconcile_secondary_name(text, regional_text)
                        regional = decide_local_candidate(regional_text, definition.datatype)
                        if text and _compact_alnum(text) != _compact_alnum(regional_text):
                            regional = decide_local_candidate("", definition.datatype)
                    if regional.accepted:
                        text, confidence = regional_text, regional_confidence
            field = _make_field(
                template,
                name,
                field_type,
                text,
                confidence,
                x0,
                y0,
                x1,
                y1,
                page_number,
                observation.width,
                observation.height,
                ExtractionMethod.REGIONAL_RAPIDOCR,
                postprocessor,
            )
            field.validation_reasons.extend(resolved.reason_codes)
            localization_id = (
                f"{observation.page_id}:{name}:{resolved.resolver_version}:"
                f"{','.join(str(item) for item in resolved.bbox)}"
            )
            crop_hash = _crop_sha256(image, resolved.bbox)
            if primary_raw:
                primary_id = f"{localization_id}:page-observation"
                field.candidates.append(
                    FieldEvidence(
                        source=ExtractionMethod.REGIONAL_RAPIDOCR,
                        raw_text=primary_raw,
                        confidence=confidence
                        if not secondary_invoked
                        else (
                            sum(token.confidence for token in ordered) / len(ordered)
                            if ordered
                            else 0
                        ),
                        bounding_box=field.bounding_box,
                        model_name="RapidOCR-ONNX-full-page-observation",
                        model_version=observation.ocr_model_version,
                        provenance=EvidenceProvenance(
                            page_sha256=observation.page_sha256,
                            source_representation_id=(
                                f"{observation.page_sha256}:{observation.observation_version}"
                            ),
                            observation_id=observation.page_id,
                            crop_sha256=crop_hash,
                            localization_id=localization_id,
                            localization_region_id=resolved.localization_evidence_id,
                            localization_method=resolved.mode.value,
                            localization_version=resolved.resolver_version,
                            preprocessing_profile="PAGE_OBSERVATION",
                            preprocessing_sha256=sha256(
                                f"PAGE_OBSERVATION|{observation.preprocessing_version}|{crop_hash}".encode()
                            ).hexdigest(),
                            preprocessing_version=observation.preprocessing_version,
                            engine_family=independence_group("rapidocr"),
                            engine_name="rapidocr",
                            engine_version=observation.ocr_model_version,
                            model_family="RAPIDOCR_ONNX",
                            model_name="RapidOCR-ONNX-full-page-observation",
                            model_version=observation.ocr_model_version,
                            source_candidate_id=primary_id,
                            invocation_id=f"{primary_id}:invocation",
                            shared_dependency_ids=(f"crop:{crop_hash}",) if crop_hash else (),
                            bbox=field.bounding_box,
                            produced_at=datetime.now(UTC),
                        ),
                    )
                )
            if secondary_invoked and regional_text:
                field.candidates.append(
                    FieldEvidence(
                        source=ExtractionMethod.ALTERNATE_PREPROCESS_OCR,
                        raw_text=regional_text,
                        confidence=regional_confidence or 0,
                        bounding_box=field.bounding_box,
                        model_name="RapidOCR-ONNX-regional",
                        model_version=getattr(self._text_extractor, "model_version", "unknown"),
                        provenance=EvidenceProvenance(
                            page_sha256=observation.page_sha256,
                            source_representation_id=f"{observation.page_sha256}:regional",
                            observation_id=observation.page_id,
                            crop_sha256=crop_hash,
                            localization_id=localization_id,
                            localization_region_id=resolved.localization_evidence_id,
                            localization_method=resolved.mode.value,
                            localization_version=resolved.resolver_version,
                            preprocessing_profile="REGIONAL_DEFAULT",
                            preprocessing_sha256=sha256(
                                f"REGIONAL_DEFAULT|{getattr(self._text_extractor, 'preprocessing_version', 'unknown')}|{crop_hash}".encode()
                            ).hexdigest(),
                            preprocessing_version=getattr(
                                self._text_extractor, "preprocessing_version", "unknown"
                            ),
                            engine_family=independence_group(
                                getattr(self._text_extractor, "engine_name", "rapidocr")
                            ),
                            engine_name=getattr(self._text_extractor, "engine_name", "rapidocr"),
                            engine_version=getattr(
                                self._text_extractor, "model_version", "unknown"
                            ),
                            model_family="RAPIDOCR_ONNX",
                            model_name="RapidOCR-ONNX-regional",
                            model_version=getattr(
                                self._text_extractor, "model_version", "unknown"
                            ),
                            source_candidate_id=f"{localization_id}:regional",
                            invocation_id=f"{localization_id}:regional:invocation",
                            shared_dependency_ids=(f"crop:{crop_hash}",) if crop_hash else (),
                            bbox=field.bounding_box,
                            produced_at=datetime.now(UTC),
                        ),
                    )
                )
            if definition is not None and field_definitions is not None:
                compact_value = _compact_alnum(field.raw_value)
                known_labels = {
                    _compact_alnum(alias)
                    for other in field_definitions.values()
                    for alias in other.aliases
                }
                if compact_value in known_labels:
                    field.validation_status = ValidationStatus.INVALID
                    field.validation_reasons.append("OBSERVED_LABEL_REJECTED_AS_VALUE")
            if (
                definition is not None
                and not decide_local_candidate(field.raw_value, definition.datatype).accepted
            ):
                field.validation_status = ValidationStatus.INVALID
                field.validation_reasons.append("DETERMINISTIC_FIELD_VALIDATION_FAILED")
            if secondary_invoked:
                field.validation_reasons.append("HIGH_RESOLUTION_REGIONAL_OCR")
            traces[name] = {
                "primary_value": primary_raw,
                "primary_selected_span": (
                    primary_span.selected_text if primary_span is not None else primary_raw
                ),
                "primary_span_rule": primary_span.rule_id if primary_span is not None else None,
                "primary_span_confidence": (
                    primary_span.confidence if primary_span is not None else None
                ),
                "primary_normalized": (
                    primary.normalized_value if definition is not None else None
                ),
                "primary_accepted": primary.accepted if definition is not None else None,
                "regional_value": regional_text,
                "regional_span_rule": regional_span.rule_id if regional_span is not None else None,
                "regional_confidence": regional_confidence,
                "regional_normalized": regional.normalized_value if regional is not None else None,
                "regional_accepted": regional.accepted if regional is not None else None,
                "selected_raw_value": field.raw_value,
                "selected_normalized_value": field.normalized_value,
                "selected_confidence": field.confidence,
                "secondary_invoked": secondary_invoked,
                "regional_ocr_trigger_reason": regional_trigger_reason,
                "changed_output": bool(
                    secondary_invoked
                    and regional_text is not None
                    and field.raw_value != " ".join(token.text for token in ordered)
                ),
                "validation_status": field.validation_status.value,
                "reason_codes": list(field.validation_reasons),
            }
            fields.append(field)
        self.last_candidate_trace = traces
        self.last_field_ocr_cost = {
            "full_page_ocr_calls": observation.full_page_ocr_calls,
            "logical_regional_requests": len(roi_results),
            "executed_regional_requests": sum(
                "HIGH_RESOLUTION_REGIONAL_OCR" in field.validation_reasons for field in fields
            ),
            "reused_observation_tokens": sum(len(observation.ocr_tokens) for _ in (0,)),
            "request_reduction_rate": 1.0 if roi_results else 0.0,
        }
        return fields

    def extract_fields_from_resolved_rois(
        self,
        image,
        template: Template,
        page_number: int,
        geometry: ExtractionGeometryDecision,
        roi_results: dict[str, ROIResolutionResult],
    ) -> list[ExtractedField]:
        """Canonical standard-field entry point.

        Raw template coordinates are never accepted here.  Every field must
        have been resolved by ``ROIResolver`` under the same geometry decision.
        """
        if geometry.mode not in {
            ExtractionGeometryMode.REGISTERED_FIXED,
            ExtractionGeometryMode.ANCHOR_RELATIVE,
        }:
            raise ValueError("FIELD_OCR_REQUIRES_RESOLVED_EXTRACTION_GEOMETRY")
        allowed_mode = (
            ROIResolutionMode.FIXED_REGISTERED
            if geometry.mode == ExtractionGeometryMode.REGISTERED_FIXED
            else ROIResolutionMode.ANCHOR_RELATIVE
        )
        missing = [
            region.field_name
            for region in template.field_regions
            if region.field_name not in roi_results
            or roi_results[region.field_name].mode != allowed_mode
            or roi_results[region.field_name].bbox is None
        ]
        if missing and geometry.mode == ExtractionGeometryMode.REGISTERED_FIXED:
            raise ValueError(f"UNRESOLVED_REQUIRED_FIELD_ROIS:{','.join(missing)}")
        if geometry.mode == ExtractionGeometryMode.ANCHOR_RELATIVE:
            fields: list[ExtractedField] = []
            logical_requests = executed_requests = 0
            method = (
                ExtractionMethod.REGIONAL_RAPIDOCR
                if getattr(self._text_extractor, "engine_name", "") == "rapidocr"
                else ExtractionMethod.REGIONAL_PADDLEOCR
            )
            region_by_name = {region.field_name: region for region in template.field_regions}
            for name, resolved in roi_results.items():
                if resolved.mode != ROIResolutionMode.ANCHOR_RELATIVE or resolved.bbox is None:
                    continue
                region = region_by_name.get(name)
                if region is None:
                    continue
                logical_requests += 1
                executed_requests += 1
                raw_text, confidence = _region_text(self._text_extractor, image, resolved.bbox)
                x0, y0, x1, y1 = resolved.bbox
                field = _make_field(
                    template,
                    name,
                    region.field_type,
                    raw_text,
                    confidence,
                    x0,
                    y0,
                    x1,
                    y1,
                    page_number,
                    image.width,
                    image.height,
                    method,
                    region.postprocessor,
                )
                field.validation_reasons.append("ANCHOR_RELATIVE_ROI")
                fields.append(field)
            if not fields:
                raise ValueError("ANCHOR_RELATIVE_GEOMETRY_HAS_NO_RESOLVED_FIELDS")
            self.last_field_ocr_cost = {
                "logical_regional_requests": logical_requests,
                "executed_regional_requests": executed_requests,
                "coalesced_requests": 0,
                "request_reduction_rate": 0.0,
            }
            return fields
        boxes = {
            name: (result.bbox,) for name, result in roi_results.items() if result.bbox is not None
        }
        return self.extract_fields(image, template, page_number, boxes)

    def extract_fields(
        self,
        image,
        template: Template,
        page_number: int,
        crop_boxes_by_field: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None,
    ) -> list[ExtractedField]:
        width, height = (
            template.reference_dimensions.width_px,
            template.reference_dimensions.height_px,
        )
        fields = []
        logical_requests = 0
        executed_requests = 0
        # Coalesce only geometrically equivalent crops. This avoids repeated
        # detector inference for semantic projections (for example first and
        # last name) while preserving regional, field-local OCR.
        crop_readings: list[tuple[tuple[int, int, int, int], tuple[str, float]]] = []
        method = (
            ExtractionMethod.REGIONAL_RAPIDOCR
            if getattr(self._text_extractor, "engine_name", "") == "rapidocr"
            else ExtractionMethod.REGIONAL_PADDLEOCR
        )
        for region in template.field_regions:
            if hasattr(self._text_extractor, "set_context"):
                self._text_extractor.set_context(
                    field=region.field_name, reason="PRIMARY_FIELD_OCR"
                )
            variants = (crop_boxes_by_field or {}).get(region.field_name)
            disagreement = False
            if variants and len(variants) > 1:
                logical_requests += len(variants)
                executed_requests += len(variants)
                readings = [_region_text(self._text_extractor, image, box) for box in variants]
                populated = [(text.strip(), score) for text, score in readings if text.strip()]
                values = {text.casefold() for text, _ in populated}
                disagreement = len(values) > 1
                raw_text, confidence = (
                    max(populated, key=lambda item: item[1]) if populated else ("", 0.0)
                )
            else:
                logical_requests += 1
                bounds = _region_bounds(image, region)
                cached = next(
                    (
                        reading
                        for prior, reading in crop_readings
                        if all(
                            abs(left - right) <= REGION_COALESCE_TOLERANCE_PX
                            for left, right in zip(prior, bounds)
                        )
                    ),
                    None,
                )
                if cached is None:
                    cached = _region_text(self._text_extractor, image, region)
                    crop_readings.append((bounds, cached))
                    executed_requests += 1
                raw_text, confidence = cached
            fields.append(
                _make_field(
                    template,
                    region.field_name,
                    region.field_type,
                    raw_text,
                    confidence,
                    region.x0,
                    region.y0,
                    region.x1,
                    region.y1,
                    page_number,
                    width,
                    height,
                    method,
                    region.postprocessor,
                )
            )
            field = fields[-1]
            field.model_name = getattr(self._text_extractor, "model_name", None)
            field.model_version = getattr(self._text_extractor, "model_version", None)
            if disagreement:
                field.validation_status = ValidationStatus.NEEDS_REVIEW
                field.validation_reasons.append("multi_crop_disagreement")
        self.last_field_ocr_cost = {
            "logical_regional_requests": logical_requests,
            "executed_regional_requests": executed_requests,
            "coalesced_requests": logical_requests - executed_requests,
            "request_reduction_rate": (
                (logical_requests - executed_requests) / logical_requests
                if logical_requests
                else 0.0
            ),
        }
        return fields

    def extract_ub04_service_lines(
        self,
        image,
        template: Template,
        page_number: int,
        *,
        registration_confidence: float,
        claim_total=None,
    ):
        """Run the structural FL42-FL48 engine on one registered table crop."""
        table = template.service_line_region
        if table is None:
            return [], None
        result = self._ub04_service_lines.extract(
            image,
            template,
            registration_confidence=registration_confidence,
            claim_total=claim_total,
        )
        return self.materialize_ub04_service_lines(result, template, page_number), result

    def materialize_ub04_service_lines(self, result, template: Template, page_number: int):
        """Convert canonical reconstruction evidence into persisted domain lines."""
        table = template.service_line_region
        if table is None:
            return []
        width, height = (
            template.reference_dimensions.width_px,
            template.reference_dimensions.height_px,
        )
        type_by_name = {
            "revenue_code": "code",
            "description": "text",
            "hcpcs_rate_hipps_code": "code",
            "service_date": "date",
            "service_units": "number",
            "total_charges": "currency",
            "non_covered_charges": "currency",
        }
        lines = []
        for reconstructed in result.lines:
            values = {
                "revenue_code": reconstructed.revenue_code,
                "description": reconstructed.description,
                "hcpcs_rate_hipps_code": reconstructed.hcpcs,
                "service_date": reconstructed.service_date,
                "service_units": reconstructed.units,
                "total_charges": reconstructed.charge,
                "non_covered_charges": reconstructed.non_covered_charge,
            }
            row_y0 = int(table.table_y0 + (reconstructed.line_number - 1) * table.row_height_px)
            fields = []
            for column in table.columns:
                raw = (
                    "" if values.get(column.field_name) is None else str(values[column.field_name])
                )
                field = _make_field(
                    template,
                    column.field_name,
                    type_by_name.get(column.field_name, column.field_type),
                    raw,
                    reconstructed.mean_confidence,
                    column.x0,
                    row_y0,
                    column.x1,
                    row_y0 + table.row_height_px,
                    page_number,
                    width,
                    height,
                )
                if reconstructed.validation_errors or not reconstructed.automatically_eligible:
                    field.validation_status = ValidationStatus.NEEDS_REVIEW
                    field.validation_reasons.extend(
                        result.reason_codes + reconstructed.validation_errors
                    )
                fields.append(field)
            line = ServiceLine(
                line_number=reconstructed.line_number,
                service_date_from=reconstructed.service_date,
                units=reconstructed.units,
                charge_amount=reconstructed.charge,
                revenue_code=reconstructed.revenue_code,
                hcpcs_code=reconstructed.hcpcs,
                non_covered_charge_amount=reconstructed.non_covered_charge,
                fields=fields,
            )
            lines.append(line)
        return lines

    def extract_service_lines(
        self, image, template: Template, page_number: int
    ) -> list[ServiceLine]:
        table = template.service_line_region
        if table is None:
            return []
        width, height = (
            template.reference_dimensions.width_px,
            template.reference_dimensions.height_px,
        )

        lines: list[ServiceLine] = []
        method = (
            ExtractionMethod.REGIONAL_RAPIDOCR
            if getattr(self._text_extractor, "engine_name", "") == "rapidocr"
            else ExtractionMethod.REGIONAL_PADDLEOCR
        )
        for row_index in range(table.max_rows):
            row_y0 = table.table_y0 + row_index * table.row_height_px
            row_y1 = min(row_y0 + table.row_height_px, table.table_y1)

            row_fields: list[ExtractedField] = []
            for column in table.columns:
                if hasattr(self._text_extractor, "set_context"):
                    self._text_extractor.set_context(
                        field=column.field_name, reason="SERVICE_LINE_CELL_OCR"
                    )
                raw_text, confidence = _region_text(
                    self._text_extractor, image, (column.x0, row_y0, column.x1, row_y1)
                )
                row_fields.append(
                    _make_field(
                        template,
                        column.field_name,
                        column.field_type,
                        raw_text,
                        confidence,
                        column.x0,
                        row_y0,
                        column.x1,
                        row_y1,
                        page_number,
                        width,
                        height,
                        method,
                    )
                )

            if _row_is_blank(row_fields):
                break  # service lines are contiguous from the top; stop at the first empty row

            line = ServiceLine(line_number=row_index + 1, fields=row_fields)
            _populate_service_line_shortcuts(line)
            lines.append(line)

        return lines


def _row_is_blank(fields: list[ExtractedField]) -> bool:
    return all(not f.raw_value.strip() for f in fields)


def _populate_service_line_shortcuts(line: ServiceLine) -> None:
    """Copy commonly-needed values onto ServiceLine's typed shortcut
    attributes (Phase 3 validation reads these directly rather than
    re-searching `line.fields` by name every time)."""
    by_name = {f.field_name: f for f in line.fields}
    if (f := by_name.get("procedure_code")) or (f := by_name.get("cpt_hcpcs")):
        line.procedure_code = f.normalized_value
    if f := by_name.get("place_of_service"):
        line.place_of_service = f.normalized_value
    if f := by_name.get("revenue_code"):
        line.revenue_code = f.normalized_value
    if (f := by_name.get("modifier")) and f.normalized_value:
        line.modifiers = [f.normalized_value]
    if (f := by_name.get("diagnosis_pointer")) and f.normalized_value:
        line.diagnosis_pointers = list(f.normalized_value.replace(" ", ""))
    for charge_field in ("charges", "total_charges"):
        if (f := by_name.get(charge_field)) and f.normalized_value:
            from decimal import Decimal, InvalidOperation

            try:
                line.charge_amount = Decimal(f.normalized_value)
            except InvalidOperation:
                pass
    if (f := by_name.get("units")) and f.normalized_value:
        from decimal import Decimal, InvalidOperation

        try:
            line.units = Decimal(f.normalized_value)
        except InvalidOperation:
            pass
    for date_field in ("date_from", "service_date"):
        if (f := by_name.get(date_field)) and f.normalized_value:
            from datetime import date

            line.service_date_from = date.fromisoformat(f.normalized_value)
    if (f := by_name.get("date_to")) and f.normalized_value:
        from datetime import date

        line.service_date_to = date.fromisoformat(f.normalized_value)
