"""Canonical Phase-8 standard-form candidate generation.

Production and evaluation supply identity through different authorities, but
both call this service after identity is established. Ground truth and route
forcing are intentionally absent from this module.
"""

from __future__ import annotations

import statistics
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PIL import Image
from pydantic import Field

from packages.document_taxonomy.taxonomy import DocumentClass
from packages.domain.claim import ServiceLine
from packages.domain.common import DomainModel
from packages.domain.extraction import ExtractedField
from packages.extraction_geometry import (
    ExtractionGeometryDecision,
    ExtractionGeometryMode,
    FormIdentityDecision,
    FormIdentityStatus,
)
from packages.field_localization import DynamicROIResolver, FieldDefinitionRegistry, FieldLocator
from packages.field_localization.contracts import FieldDefinition, FieldLocationEvidence
from packages.forms.cms1500 import CMS1500FieldGraph
from packages.forms.ub04 import UB04StructuralMap, UB04StructuralMapDetector
from packages.page_observation import PageObservation, PageObservationService
from packages.roi_resolution import ROIResolutionResult
from packages.templates.models import Template
from workers.standard_form_extraction.extractor import StandardFormExtractionService
from workers.table_extraction.observation_service_lines import UB04ObservationServiceLineExtractor
from workers.table_extraction.ub04_service_lines import UB04ReconstructionResult

ROOT = Path(__file__).resolve().parents[2]


class ExtractionDiagnostics(DomainModel):
    service_version: str
    stage_ms: dict[str, float] = Field(default_factory=dict)
    full_page_ocr_calls: int = 0
    regional_ocr_calls: int = 0
    observation_tokens: int = 0
    roi_requests: int = 0
    resolved_rois: int = 0
    tokens_in_rois: int = 0
    tokens_overlapping_multiple_fields: int = 0
    tokens_outside_all_fields: int = 0
    label_tokens_rejected: int = 0
    candidate_tokens_accepted: int = 0
    multi_token_concatenations: int = 0
    regional_ocr_reasons: dict[str, int] = Field(default_factory=dict)
    regional_fields_improved: int = 0
    regional_fields_unchanged: int = 0
    regional_fields_worsened: int = 0
    reason_codes: list[str] = Field(default_factory=list)


class StandardFormProcessingResult(DomainModel):
    observation: PageObservation
    geometry: ExtractionGeometryDecision
    fields: list[ExtractedField]
    service_lines: list[ServiceLine]
    roi_results: dict[str, ROIResolutionResult]
    field_locations: dict[str, FieldLocationEvidence]
    field_definitions: dict[str, FieldDefinition]
    diagnostics: ExtractionDiagnostics
    ub_structure: UB04StructuralMap | None = None
    ub_reconstruction: UB04ReconstructionResult | None = None


class StandardFormProcessingService:
    """One candidate-generation implementation for runtime and evaluation."""

    version = "standard-form-processing-v1"

    def __init__(
        self,
        observation_service: PageObservationService,
        extraction_service: StandardFormExtractionService,
        *,
        cms_graph: CMS1500FieldGraph | None = None,
        ub_registry: FieldDefinitionRegistry | None = None,
        ub_service_lines: UB04ObservationServiceLineExtractor | None = None,
    ) -> None:
        self.observation_service = observation_service
        self.extraction_service = extraction_service
        self.cms_graph = cms_graph or CMS1500FieldGraph()
        self.ub_registry = ub_registry or FieldDefinitionRegistry.load(
            ROOT / "config/field_definitions/ub04_v1.yaml"
        )
        self.ub_service_lines = ub_service_lines or UB04ObservationServiceLineExtractor()

    def process(
        self,
        image: Image.Image,
        template: Template,
        page_number: int,
        form_identity: FormIdentityDecision,
        *,
        page_id: str,
        page_sha256: str | None = None,
        observation: PageObservation | None = None,
        registered_geometry: ExtractionGeometryDecision | None = None,
    ) -> StandardFormProcessingResult:
        if form_identity.status != FormIdentityStatus.VERIFIED:
            raise ValueError("PHASE8_PROCESSING_REQUIRES_VERIFIED_FORM_IDENTITY")
        expected = (
            DocumentClass.CMS1500 if template.template_id == "cms1500" else DocumentClass.UB04
        )
        if form_identity.family != expected:
            raise ValueError("PHASE8_PROCESSING_IDENTITY_TEMPLATE_MISMATCH")

        stages: dict[str, float] = {}
        started = time.perf_counter()
        if observation is None:
            observation = self.observation_service.observe(
                page_id, image, page_sha256=page_sha256
            )
        stages["page_observation"] = (time.perf_counter()-started)*1000
        regional_extractor = getattr(self.extraction_service, "_text_extractor", None)
        if hasattr(regional_extractor, "set_context"):
            regional_extractor.set_context(page_hash=observation.page_sha256)

        started = time.perf_counter()
        ub_structure = None
        if expected == DocumentClass.CMS1500:
            definitions = {
                item.field_name: item
                for item in self.cms_graph.registry.for_family("CMS1500")
            }
            locations = self.cms_graph.locate(observation)
            structures = {}
            mode = ExtractionGeometryMode.ANCHOR_RELATIVE
            reason = "CMS_FIELD_GRAPH"
            structural_confidence = statistics.fmean(
                item.confidence for item in locations.values() if item.bbox is not None
            ) if any(item.bbox is not None for item in locations.values()) else 0.0
        else:
            definitions = {
                item.field_name: item for item in self.ub_registry.for_family("UB04")
            }
            locator = FieldLocator()
            locations = {
                item.field_name: locator.locate(observation, item)
                for item in definitions.values()
            }
            ub_structure = UB04StructuralMapDetector().detect(observation)
            structures = {
                name: ub_structure.field_region(name) for name in definitions
            }
            mode = ExtractionGeometryMode.STRUCTURAL_LAYOUT
            reason = "UB_STRUCTURAL_MAP"
            structural_confidence = ub_structure.confidence
        stages["layout_inference"] = (time.perf_counter()-started)*1000

        dynamic_geometry = ExtractionGeometryDecision(
            mode=mode, form_identity=form_identity,
            template_id=template.template_id, template_version=template.version,
            structural_confidence=structural_confidence,
            reason_codes=("DYNAMIC_LAYOUT_DEFAULT", reason),
        )
        geometry = registered_geometry or dynamic_geometry
        if registered_geometry is not None:
            if not registered_geometry.authorizes_fixed_roi:
                raise ValueError("TEMPLATE_FALLBACK_REQUIRES_AUTHORIZED_REGISTERED_GEOMETRY")
            if registered_geometry.form_identity.family != form_identity.family:
                raise ValueError("TEMPLATE_FALLBACK_FORM_IDENTITY_MISMATCH")
        started = time.perf_counter()
        resolver = DynamicROIResolver()
        template_regions = {item.field_name: item for item in template.field_regions}
        # Preserve the complete registered template field contract. Dynamic
        # definitions enrich known fields, but must not silently remove legacy
        # canonical fields from the candidate set. Template-only fields remain
        # fail-closed unless registered geometry passes the existing safety gate.
        field_names = tuple(dict.fromkeys((*definitions, *template_regions)))
        rois = {
            name: resolver.resolve(
                name, anchor=locations.get(name), structural=structures.get(name),
                geometry=geometry,
                registered_template_bbox=(
                    (
                        template_regions[name].x0,
                        template_regions[name].y0,
                        template_regions[name].x1,
                        template_regions[name].y1,
                    )
                    if name in template_regions else None
                ),
                page_size=(observation.width, observation.height),
            )
            for name in field_names
        }
        stages["roi_resolution"] = (time.perf_counter()-started)*1000

        started = time.perf_counter()
        fields = self.extraction_service.extract_fields_from_observation(
            observation, template, page_number, rois, definitions, image
        )
        stages["field_candidate_generation"] = (time.perf_counter()-started)*1000

        service_lines: list[ServiceLine] = []
        ub_result = None
        if ub_structure is not None:
            total = next((field for field in fields if field.field_name == "total_charge"), None)
            try:
                claim_total = Decimal(total.normalized_value) if total and total.normalized_value else None
            except (InvalidOperation, ValueError):
                claim_total = None
            started = time.perf_counter()
            ub_result = self.ub_service_lines.extract(
                observation, ub_structure, claim_total=claim_total,
                image=image, text_extractor=regional_extractor,
            )
            service_lines = self.extraction_service.materialize_ub04_service_lines(
                ub_result, template, page_number
            )
            stages["ub_service_line_reconstruction"] = (time.perf_counter()-started)*1000

        cost = self.extraction_service.last_field_ocr_cost
        resolved_boxes = [result.bbox for result in rois.values() if result.bbox is not None]
        token_memberships = [
            sum(
                x0 <= (token.bbox[0] + token.bbox[2]) / 2 <= x1
                and y0 <= (token.bbox[1] + token.bbox[3]) / 2 <= y1
                for x0, y0, x1, y1 in resolved_boxes
            )
            for token in observation.ocr_tokens
        ]
        traces = self.extraction_service.last_candidate_trace
        regional_reasons: dict[str, int] = {}
        for trace in traces.values():
            reason_code = trace.get("regional_ocr_trigger_reason")
            if reason_code:
                regional_reasons[reason_code] = regional_reasons.get(reason_code, 0) + 1
        diagnostics = ExtractionDiagnostics(
            service_version=self.version,
            stage_ms=stages,
            full_page_ocr_calls=observation.full_page_ocr_calls,
            regional_ocr_calls=(int(cost.get("executed_regional_requests", 0)) +
                                (ub_result.regional_ocr_calls if ub_result else 0)),
            observation_tokens=len(observation.ocr_tokens),
            roi_requests=len(rois),
            resolved_rois=sum(result.bbox is not None for result in rois.values()),
            tokens_in_rois=sum(value > 0 for value in token_memberships),
            tokens_overlapping_multiple_fields=sum(value > 1 for value in token_memberships),
            tokens_outside_all_fields=sum(value == 0 for value in token_memberships),
            label_tokens_rejected=sum(
                "OBSERVED_LABEL_REJECTED_AS_VALUE" in field.validation_reasons for field in fields
            ),
            candidate_tokens_accepted=sum(
                bool(trace.get("primary_accepted")) for trace in traces.values()
            ),
            multi_token_concatenations=sum(
                len(str(trace.get("primary_value") or "").split()) > 1
                for trace in traces.values()
            ),
            regional_ocr_reasons=regional_reasons,
            regional_fields_improved=sum(
                bool(trace.get("regional_accepted")) and bool(trace.get("changed_output"))
                for trace in traces.values()
            ),
            regional_fields_unchanged=sum(
                bool(trace.get("secondary_invoked")) and not bool(trace.get("changed_output"))
                for trace in traces.values()
            ),
            regional_fields_worsened=0,
            reason_codes=["CANONICAL_PHASE8_PROCESSING"],
        )
        return StandardFormProcessingResult(
            observation=observation, geometry=geometry, fields=fields,
            service_lines=service_lines, roi_results=rois, field_locations=locations,
            field_definitions=definitions, diagnostics=diagnostics,
            ub_structure=ub_structure, ub_reconstruction=ub_result,
        )
