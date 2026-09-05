"""Validation Worker Consumer: consumes `extraction.completed`, converts
extracted fields into a canonical `Claim` domain model, evaluates field
rules and confidence thresholds via `ValidationEngine`, and outboxes either
`claim.validated` or `field.retry.requested` events.  Extraction and
validation never create HITL tasks; retry owns the post-decision authority.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from apps.ingestion_api.db.mappers import orm_to_extracted_field
from apps.ingestion_api.db.models import ExtractedFieldORM, PageClassificationORM, PageORM
from apps.ingestion_api.db.repository import (
    DocumentRepository,
    SqlAlchemyOutboxRepository,
)
from packages.claim_decision import ClaimDecisionContext, ClaimDecisionService
from packages.claim_evidence import ClaimEvidenceBuilder
from packages.criticality import DEFAULT_CRITICALITY_PATH, CriticalityLevel, CriticalityPolicy
from packages.deterministic_evidence import DeterministicEvidenceService
from packages.domain.claim import Claim, ServiceLine
from packages.domain.enums import ClaimFormType, DocumentStatus, ValidationStatus
from packages.events.bus import EventBus
from packages.events.envelope import EventEnvelope
from packages.events.outbox import OutboxRecord
from packages.events.topics import Topic
from packages.evidence import StructuralLocalizationEvidence, StructuralLocalizationType
from packages.evidence_decision import DecisionContext, EvidenceDecisionService, FieldDisposition
from packages.evidence_decision.adapters import ocr_candidates_from_field
from packages.evidence_router import ReferenceSourceState
from packages.reference_enrichment.evidence_adapter import ReferenceEvidenceService
from packages.runtime_profile import DecisionServiceFactory
from packages.templates.registry import DEFAULT_TEMPLATE_DIR, TemplateRegistry
from packages.validation_rules.engine import ValidationEngine
from packages.validation_rules.thresholds import ThresholdRegistry

logger = logging.getLogger(__name__)
DEFAULT_REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "reference_enrichment.yaml"
)

CONSUMER_GROUP = "validation-worker"


def registration_confidence_from_evidence(evidence: dict | None) -> float:
    """Return measured alignment confidence; unavailable/rejected evidence fails closed."""
    if not evidence or evidence.get("accepted") is not True:
        return 0.0
    value = evidence.get("alignment_confidence")
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def reference_source_state(
    service: ReferenceEvidenceService, field_name: str | None = None
) -> ReferenceSourceState:
    def supports(provider) -> bool:
        if field_name is None:
            return True
        availability = getattr(provider, "is_available", None)
        if callable(availability):
            return bool(availability(field_name))
        checker = getattr(provider, "supports", None)
        if callable(checker):
            return bool(checker(field_name))
        configured = getattr(provider, "config", {}).get("supported_fields", [])
        return not configured or field_name in configured

    providers = [provider for provider in service.providers if supports(provider)]
    if any(provider.authorized and not provider.test_only for provider in providers):
        return ReferenceSourceState.AUTHORIZED
    if any(provider.test_only for provider in providers):
        return ReferenceSourceState.TEST_FIXTURE
    return ReferenceSourceState.DISABLED


def extraction_geometry_evidence(
    payload: dict, field_page_number: int
) -> tuple[float | None, str | None]:
    """Return measured dynamic geometry evidence carried by extraction.

    Template registration and dynamic structural localization are distinct,
    but both satisfy E3 only when their measured confidence clears policy.
    """
    if payload.get("page_number") != field_page_number:
        return None, None
    geometry = payload.get("extraction_geometry") or {}
    mode = geometry.get("mode")
    if mode not in {"ANCHOR_RELATIVE", "STRUCTURAL_LAYOUT"}:
        return None, None
    confidence = geometry.get("structural_confidence")
    if confidence is None:
        return None, None
    return max(0.0, min(1.0, float(confidence))), f"DYNAMIC_GEOMETRY:{mode}"


def qualified_structural_localization(
    payload: dict,
    field_page_number: int,
    field_name: str,
) -> StructuralLocalizationEvidence | None:
    """Build E3 only from persisted, measurable geometry checks."""
    if payload.get("page_number") != field_page_number:
        return None
    geometry = payload.get("extraction_geometry") or {}
    form_identity = geometry.get("form_identity") or {}
    roi = (payload.get("roi_resolution") or {}).get(field_name) or {}
    roi_mode = roi.get("mode")
    mode = {
        "FIXED_REGISTERED": "REGISTERED_FIXED",
        "STRUCTURAL_REGION": "STRUCTURAL_LAYOUT",
    }.get(roi_mode, roi_mode) or geometry.get("mode")
    confidence = float(roi.get("field_structural_confidence") or 0)
    reasons = set(roi.get("reason_codes") or [])
    positive_bbox = bool(
        roi.get("bbox")
        and len(roi["bbox"]) == 4
        and roi["bbox"][2] > roi["bbox"][0]
        and roi["bbox"][3] > roi["bbox"][1]
    )
    wrong_crop = any(reason.startswith("WRONG_CROP_") for reason in reasons)
    common = (
        form_identity.get("status") == "VERIFIED"
        and confidence >= 0.80
        and positive_bbox
        and not wrong_crop
    )
    if mode == "ANCHOR_RELATIVE":
        required = {"DYNAMIC_PRIORITY_1_ANCHOR", "BOUNDED_ALIAS_MATCH"}
        geometry_proof = bool(
            reasons
            & {
                "OBSERVED_VALUE_TOKEN_GEOMETRY",
                "OBSERVED_VALUE_SPAN_GEOMETRY",
                "FIELD_SPECIFIC_SPATIAL_CONTRACT",
            }
        )
        confirmed = common and required <= reasons and geometry_proof
        subtype = StructuralLocalizationType.ANCHOR_RELATIVE_LOCALIZATION_CONFIRMED
    elif mode == "STRUCTURAL_LAYOUT":
        confirmed = common and "DYNAMIC_PRIORITY_2_STRUCTURE" in reasons
        subtype = StructuralLocalizationType.STRUCTURAL_LAYOUT_CONFIRMED
    elif mode == "REGISTERED_FIXED":
        compatibility = geometry.get("compatibility") or {}
        registration = geometry.get("registration") or {}
        confirmed = (
            common
            and compatibility.get("status") != "INCOMPATIBLE"
            and registration.get("accepted") is True
            and registration.get("corner_validity") is True
            and geometry.get("transformed_geometry_valid") is True
            and "DYNAMIC_PRIORITY_3_TEMPLATE_FAST_PATH" in reasons
        )
        subtype = StructuralLocalizationType.TEMPLATE_REGISTRATION_CONFIRMED
    else:
        return None
    audit_reasons = tuple(
        sorted(
            {
                *reasons,
                "DOCUMENT_FAMILY_VERIFIED"
                if form_identity.get("status") == "VERIFIED"
                else "DOCUMENT_FAMILY_NOT_VERIFIED",
                "POSITIVE_BOUNDED_ROI" if positive_bbox else "ROI_NOT_POSITIVE",
                "STRUCTURAL_CONFIDENCE_PASSED"
                if confidence >= 0.80
                else "STRUCTURAL_CONFIDENCE_FAILED",
                "WRONG_CROP_FIREWALL_FAILED"
                if wrong_crop
                else "WRONG_CROP_FIREWALL_PASSED",
            }
        )
    )
    return StructuralLocalizationEvidence(
        evidence_type=subtype,
        confidence=confidence,
        confirmed=confirmed,
        reason_codes=audit_reasons,
        source=f"DYNAMIC_GEOMETRY:{mode}",
        field_name=field_name,
        field_bbox=tuple(roi["bbox"]) if positive_bbox else None,
        localization_mode=mode,
        anchor_id=next(iter(roi.get("anchor_ids") or []), None),
        anchor_confidence=confidence if mode == "ANCHOR_RELATIVE" else None,
        neighbor_evidence=tuple(
            reason for reason in reasons if "NEIGHBOR" in reason or "BOUND" in reason
        ),
        positive_bounded_roi=positive_bbox,
        geometry_valid=positive_bbox,
        registration_compatible=(
            (geometry.get("compatibility") or {}).get("status") != "INCOMPATIBLE"
            if mode == "REGISTERED_FIXED" else None
        ),
    )


class ValidationWorker:
    def __init__(
        self,
        event_bus: EventBus,
        session_factory: sessionmaker,
        pipeline_version: str,
        templates: TemplateRegistry,
        validation_engine: ValidationEngine | None = None,
        decision_service: EvidenceDecisionService | None = None,
        deterministic_service: DeterministicEvidenceService | None = None,
        reference_service: ReferenceEvidenceService | None = None,
        claim_decision_service: ClaimDecisionService | None = None,
        claim_evidence_builder: ClaimEvidenceBuilder | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._session_factory = session_factory
        self._pipeline_version = pipeline_version
        self._templates = templates
        self._validation_engine = (
            validation_engine
            if validation_engine is not None
            else ValidationEngine(ThresholdRegistry.load_from_directory())
        )
        canonical = (
            DecisionServiceFactory.from_profile()
            if decision_service is None or claim_decision_service is None
            else None
        )
        self._decision_service = decision_service or canonical.evidence_decision
        self._deterministic_service = deterministic_service or DeterministicEvidenceService()
        self._reference_service = reference_service or canonical.reference_evidence
        self._claim_decision_service = claim_decision_service or canonical.claim_decision
        self._claim_evidence_builder = claim_evidence_builder or ClaimEvidenceBuilder.load()
        self._criticality = (
            canonical.criticality
            if canonical is not None
            else CriticalityPolicy.load(DEFAULT_CRITICALITY_PATH)
        )

    async def handle_one(self, envelope: EventEnvelope) -> None:
        document_id = envelope.document_id
        if document_id is None:
            logger.warning("extraction.completed event missing document_id, skipping")
            return

        with self._session_factory() as session:
            documents = DocumentRepository(session)
            outbox = SqlAlchemyOutboxRepository(session)

            document = documents.get(document_id)
            if document is None:
                logger.warning("document %s not found, skipping", document_id)
                return

            from sqlalchemy import select

            stmt = (
                select(ExtractedFieldORM)
                .where(ExtractedFieldORM.document_id == document_id)
                .order_by(ExtractedFieldORM.page_number)
            )
            rows = session.execute(stmt).scalars().all()
            if not rows:
                logger.warning(
                    "document %s has no extracted fields, routing to review", document_id
                )
                document.status = DocumentStatus.NEEDS_REVIEW
                document.updated_at = datetime.now(UTC)
                documents.update(document)
                session.commit()
                return

            classification_rows = (
                session.execute(
                    select(PageClassificationORM)
                    .where(PageClassificationORM.document_id == document_id)
                    .order_by(PageClassificationORM.classified_at.desc())
                )
                .scalars()
                .all()
            )
            registration_by_page: dict[int, dict] = {}
            page_numbers = {
                page_id: page_number
                for page_id, page_number in session.execute(
                    select(PageORM.page_id, PageORM.page_number).where(
                        PageORM.document_id == document_id
                    )
                ).all()
            }
            for classification in classification_rows:
                page_number = page_numbers.get(classification.page_id)
                if page_number is not None and page_number not in registration_by_page:
                    registration_by_page[page_number] = classification.registration_evidence or {}

            header_fields = []
            service_lines_map: dict[int, list] = {}

            for r in rows:
                field = orm_to_extracted_field(r)
                if r.service_line_number is None:
                    header_fields.append(field)
                else:
                    service_lines_map.setdefault(r.service_line_number, []).append(field)

            service_lines = [
                ServiceLine(line_number=line_num, fields=f_list)
                for line_num, f_list in sorted(service_lines_map.items())
            ]

            from packages.templates.selection import (
                exact_family_template,
                form_type_from_template_lineage,
            )

            form_type = form_type_from_template_lineage(rows[0].template_version)
            if form_type == ClaimFormType.UNSTRUCTURED:
                template = None
            else:
                template = exact_family_template(self._templates, form_type)

            total_charge_val = None
            total_charge_field = next(
                (f for f in header_fields if f.field_name == "total_charge"), None
            )
            if total_charge_field and total_charge_field.raw_value:
                try:
                    total_charge_val = Decimal(
                        total_charge_field.raw_value.replace("$", "").replace(",", "").strip()
                    )
                except (InvalidOperation, ValueError):
                    logger.warning("invalid total charge on document %s", document_id)

            if (
                service_lines
                and total_charge_val is not None
                and not any(l.charge_amount for l in service_lines)
            ):
                service_lines[0].charge_amount = total_charge_val
            elif not service_lines and total_charge_val is not None:
                service_lines = [ServiceLine(line_number=1, charge_amount=total_charge_val)]

            claim = Claim(
                claim_id=document.claim_id or document_id,
                document_id=document_id,
                tenant_id=document.tenant_id,
                correlation_id=envelope.correlation_id,
                form_type=form_type,
                total_charge_amount=total_charge_val,
                schema_version=document.schema_version,
                template_version=template.version if template else None,
                header_fields=header_fields,
                service_lines=service_lines,
            )

            validation_results = self._validation_engine.validate_claim(claim, template)
            claim_values = {
                field.field_name: field.normalized_value or field.raw_value
                for field in claim.all_fields()
            }
            claim_value_occurrences: dict[str, list[str]] = {}
            for field in claim.all_fields():
                value = field.normalized_value or field.raw_value
                if value is not None:
                    claim_value_occurrences.setdefault(field.field_name, []).append(value)
            service_line_charges = [
                field.normalized_value or field.raw_value
                for line in service_lines
                for field in line.fields
                if field.field_name in {"charges", "total_charges", "charge_amount"}
                and (field.normalized_value or field.raw_value)
            ]
            if service_line_charges:
                claim_values["service_line_charges"] = ",".join(service_line_charges)
            evidence_values: dict[str, object] = {
                **claim_values,
                **{
                    name: values if len(values) > 1 else values[0]
                    for name, values in claim_value_occurrences.items()
                },
            }
            claim_evidence = self._claim_evidence_builder.build(
                claim_id=str(claim.claim_id),
                document_family=form_type.value,
                claim_values=evidence_values,
                service_lines=[
                    {
                        field.field_name: field.normalized_value or field.raw_value
                        for field in line.fields
                    }
                    for line in service_lines
                    if line.fields
                ],
            )

            # Map validation results by field_id
            results_by_field_id = {}
            for res in validation_results:
                if res.field_id:
                    results_by_field_id.setdefault(res.field_id, []).append(res)

            needs_retry_count = 0
            field_decisions = []
            pending_retries: list[tuple[EventEnvelope, str]] = []

            # Process each field
            for r in rows:
                field = orm_to_extracted_field(r)
                field_results = results_by_field_id.get(field.field_id, [])

                # Check if hard validation passes
                reasons = []
                for rule_res in field_results:
                    if rule_res.status in (ValidationStatus.INVALID, ValidationStatus.NEEDS_REVIEW):
                        reasons.append(rule_res.rule_name)
                        from packages.domain.enums import FieldCriticality
                        from packages.observability.metrics import validation_failure_total

                        validation_failure_total.labels(
                            rule_name=rule_res.rule_name,
                            criticality="critical"
                            if self._validation_engine._criticality(field.field_name)
                            == FieldCriticality.CRITICAL
                            else "non_critical",
                        ).inc()

                deterministic = self._deterministic_service.evaluate(
                    field.field_name,
                    field.normalized_value or field.raw_value,
                    claim_values=claim_values,
                )
                hard_validation_passed = deterministic.passed
                field_policy = self._decision_service.field_policy.for_field(
                    form_type.value,
                    field.field_name,
                )
                level = field_policy.criticality
                is_critical = level in {CriticalityLevel.C2, CriticalityLevel.C3}
                r.is_critical = is_critical
                crop_reasons = set(field.validation_reasons)
                wrong_crop = bool(
                    crop_reasons
                    & {
                        "WRONG_CROP_SUSPECTED",
                        "wrong_crop_suspected",
                        "alignment_quality_not_verified",
                    }
                )
                registration_evidence = registration_by_page.get(field.page_number)
                dynamic_confidence, dynamic_source = extraction_geometry_evidence(
                    envelope.payload, field.page_number
                )
                registration_confidence = (
                    dynamic_confidence
                    if dynamic_confidence is not None
                    else registration_confidence_from_evidence(registration_evidence)
                )
                wrong_crop = wrong_crop or registration_confidence < 0.60
                structural_localization = qualified_structural_localization(
                    envelope.payload,
                    field.page_number,
                    field.field_name,
                )
                reference, reference_provenance = self._reference_service.evidence(
                    document_id=str(document_id),
                    page_number=field.page_number,
                    document_family=form_type.value,
                    field_name=field.field_name,
                    criticality=level,
                    raw_value=field.raw_value,
                    normalized_value=field.normalized_value,
                    claim_values=claim_values,
                )
                r.reference_evidence = reference_provenance
                decision = self._decision_service.decide(
                    DecisionContext(
                        field_id=str(field.field_id),
                        field_name=field.field_name,
                        document_family=form_type.value,
                        criticality=level,
                        required=field_policy.required,
                        blocks_stp=field_policy.blocks_stp,
                        requires_review_when_unresolved=field_policy.requires_review_when_unresolved,
                        candidates=ocr_candidates_from_field(field),
                        deterministic_evidence=deterministic.evidence,
                        deterministic_evidence_version=self._deterministic_service.policy_version,
                        hard_validation_passed=hard_validation_passed,
                        registration_confidence=registration_confidence,
                        structural_evidence_source=(
                            dynamic_source
                            or (
                                f"MEASURED_REGISTRATION:{registration_evidence.get('algorithm', 'unknown')}"
                                if registration_evidence
                                else None
                            )
                        ),
                        structural_localization=structural_localization,
                        wrong_crop_suspected=wrong_crop,
                        cross_field_evidence=(
                            set(deterministic.cross_field_evidence)
                            | claim_evidence.evidence_types_for(field.field_name)
                        ),
                        reference=reference,
                        reference_source_state=reference_source_state(
                            self._reference_service, field.field_name
                        ),
                    )
                )
                field_decisions.append(decision)
                r.disposition = decision.disposition.value
                accepted = decision.disposition in {
                    FieldDisposition.AUTO_ACCEPTED,
                    FieldDisposition.REFERENCE_CONFIRMED,
                }
                r.validation_status = "VALID" if accepted else "NEEDS_REVIEW"
                r.validation_reasons = list(
                    dict.fromkeys([*r.validation_reasons, *decision.reason_codes])
                )

                if (
                    not accepted
                    and decision.disposition is not FieldDisposition.UNRESOLVED_NON_BLOCKING
                ):
                    needs_retry_count += 1
                    retry_envelope = EventEnvelope(
                        event_type=Topic.FIELD_RETRY_REQUESTED.value,
                        correlation_id=envelope.correlation_id,
                        document_id=document_id,
                        claim_id=claim.claim_id,
                        pipeline_version=self._pipeline_version,
                        payload={
                            "field_id": str(field.field_id),
                            "field_name": field.field_name,
                            "source_dataset": envelope.payload.get("source_dataset")
                            or envelope.payload.get("source"),
                            "source_quality_band": envelope.payload.get("source_quality_band")
                            or envelope.payload.get("quality_bucket"),
                            "crop_safety_status": (
                                "CROP_SAFE"
                                if structural_localization is not None
                                and "WRONG_CROP_SUSPECTED" not in field.validation_reasons
                                else "UNSAFE"
                            ),
                            "next_action": decision.next_action.value,
                            "reason_codes": decision.reason_codes,
                            "policy_version": decision.policy_version,
                            "hard_validation_passed": hard_validation_passed,
                            "deterministic_policy_version": self._deterministic_service.policy_version,
                            "deterministic_failures": deterministic.failure_reasons,
                            "available_evidence": decision.available_evidence,
                            "missing_evidence": decision.missing_evidence,
                            "reference_evidence": reference_provenance,
                            "registration_evidence": registration_evidence,
                            "evidence_bundle": (
                                decision.evidence_bundle.model_dump(mode="json")
                                if decision.evidence_bundle
                                else None
                            ),
                            "decision_context_evidence": {
                                "document_family": form_type.value,
                                "criticality": level.value,
                                "required": field_policy.required,
                                "blocks_stp": field_policy.blocks_stp,
                                "requires_review_when_unresolved": field_policy.requires_review_when_unresolved,
                                "deterministic_evidence": sorted(deterministic.evidence),
                                "deterministic_evidence_version": self._deterministic_service.policy_version,
                                "cross_field_evidence": sorted(
                                    set(deterministic.cross_field_evidence)
                                    | claim_evidence.evidence_types_for(field.field_name)
                                ),
                                "registration_confidence": registration_confidence,
                                "structural_evidence_source": (
                                    dynamic_source
                                    or (
                                        f"MEASURED_REGISTRATION:{registration_evidence.get('algorithm', 'unknown')}"
                                        if registration_evidence
                                        else None
                                    )
                                ),
                                "structural_localization": (
                                    structural_localization.model_dump(mode="json")
                                    if structural_localization
                                    else None
                                ),
                                "reference": reference.model_dump(mode="json")
                                if reference
                                else None,
                                "reference_source_state": reference_source_state(
                                    self._reference_service, field.field_name
                                ).value,
                            },
                        },
                    )
                    pending_retries.append((retry_envelope, field.field_name))

            claim_decision = self._claim_decision_service.decide(
                ClaimDecisionContext(
                    claim_id=str(claim.claim_id),
                    document_family=form_type.value,
                    field_decisions=field_decisions,
                    claim_evidence=claim_evidence.evidence_items,
                    contradictions=claim_evidence.contradictions,
                    policy_id=self._claim_decision_service.policy_id,
                    policy_version=self._claim_decision_service.policy_version,
                    dependent_field_groups=(
                        [["total_charge", "charges", "charge_amount"]]
                        if form_type is ClaimFormType.CMS1500
                        else [["revenue_code", "hcpcs_code", "units", "charges", "charge_amount"]]
                    ),
                )
            )

            # ClaimDecisionService remains the sole authority for blocker state.
            # Attach its result to field-review work only after the canonical
            # claim decision exists, so reviewers can prioritize claim unlocks.
            blockers = set(claim_decision.blocking_unresolved_fields)
            blocker_count = len(blockers)
            for retry_envelope, field_name in pending_retries:
                blocks_stp = field_name in blockers
                retry_envelope.payload.update(
                    {
                        "blocks_stp": blocks_stp,
                        "blocking_field_count": blocker_count,
                        "single_blocker_claim": blocks_stp and blocker_count == 1,
                        "claim_unlock_value": (
                            1.0 / blocker_count if blocks_stp and blocker_count else 0.0
                        ),
                        "claim_impact": (
                            "THIS FIELD BLOCKS CLAIM STP" if blocks_stp else "NONBLOCKING"
                        ),
                    }
                )
                await outbox.add(
                    OutboxRecord(
                        topic=Topic.FIELD_RETRY_REQUESTED.value,
                        envelope=retry_envelope,
                        partition_key=str(document_id),
                    )
                )

            # Outbox the canonical field and claim decisions for all downstream consumers.
            completed_envelope = EventEnvelope(
                event_type=Topic.CLAIM_VALIDATED.value,
                correlation_id=envelope.correlation_id,
                document_id=document_id,
                claim_id=claim.claim_id,
                pipeline_version=self._pipeline_version,
                payload={
                    "document_id": str(document_id),
                    "needs_retry_count": needs_retry_count,
                    "tenant_id": document.tenant_id,
                    "form_type": form_type.value,
                    "validation_results_count": len(validation_results),
                    "field_decisions": [
                        decision.model_dump(mode="json") for decision in field_decisions
                    ],
                    "claim_evidence": claim_evidence.model_dump(mode="json"),
                    "claim_decision": claim_decision.model_dump(mode="json"),
                },
            )
            await outbox.add(
                OutboxRecord(
                    topic=Topic.CLAIM_VALIDATED.value,
                    envelope=completed_envelope,
                    partition_key=str(document_id),
                )
            )

            document.status = (
                DocumentStatus.COMPLETED
                if claim_decision.stp_eligible
                else DocumentStatus.VALIDATING
            )
            document.updated_at = datetime.now(UTC)
            documents.update(document)
            session.commit()

    async def run_forever(self) -> None:
        async for _topic, envelope in self._event_bus.subscribe(
            [Topic.EXTRACTION_COMPLETED.value], group_id=CONSUMER_GROUP
        ):
            try:
                await self.handle_one(envelope)
            except Exception:
                logger.exception("failed to validate extraction output")


async def _run(worker: "ValidationWorker", relay) -> None:
    relay_task = asyncio.create_task(relay.run_forever())
    try:
        await worker.run_forever()
    finally:
        relay.stop()
        await relay_task


def main() -> None:
    from apps.ingestion_api.db.repository import PollingOutboxRepository
    from apps.ingestion_api.db.session import make_session_factory
    from packages.events.bus import AIOKafkaEventBus
    from packages.events.outbox import OutboxRelay
    from packages.observability import configure_logging
    from packages.settings import get_settings

    configure_logging("validation-worker")
    settings = get_settings()
    session_factory = make_session_factory(settings.database_url)
    event_bus = AIOKafkaEventBus(settings.kafka_bootstrap_servers)
    worker = ValidationWorker(
        event_bus=event_bus,
        session_factory=session_factory,
        pipeline_version=settings.pipeline_version,
        templates=TemplateRegistry.load_from_directory(DEFAULT_TEMPLATE_DIR),
    )
    # ValidationWorker writes FIELD_RETRY_REQUESTED/etc. rows to the outbox
    # table (transactional-outbox pattern); a relay must run alongside the
    # consumer loop to actually publish those rows to Kafka, or downstream
    # HITL task creation never fires. See packages/events/outbox.py.
    relay = OutboxRelay(
        repository=PollingOutboxRepository(session_factory),
        event_bus=event_bus,
    )
    asyncio.run(_run(worker, relay))


if __name__ == "__main__":
    main()
