"""Retry/escalation worker: consumes `field.retry.requested`, uses `ModelRouter`
to pick the next stage, and executes it. Outboxes either another `field.retry.requested`
(if still unresolved but attempts remain) or `human.review.requested` (if exhausted).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import sessionmaker

from apps.ingestion_api.db.mappers import orm_to_extracted_field
from apps.ingestion_api.db.models import ExtractedFieldORM
from apps.ingestion_api.db.repository import (
    DocumentRepository,
    SqlAlchemyOutboxRepository,
)
from packages.deterministic_evidence import DeterministicEvidenceService
from packages.domain.enums import ClaimFormType, ExtractionMethod, FieldCriticality
from packages.domain.extraction import FieldEvidence
from packages.events.bus import EventBus
from packages.events.envelope import EventEnvelope
from packages.events.outbox import OutboxRecord
from packages.events.topics import Topic
from packages.evidence.builder import engine_family
from packages.evidence.normalization import normalize_agreement_value
from packages.evidence_decision import (
    DecisionContext,
    EvidenceDecisionService,
    FieldDisposition,
    NextAction,
    ReferenceEvidence,
)
from packages.evidence_decision.adapters import ocr_candidates_from_field
from packages.evidence_router import ReferenceSourceState
from packages.human_review_authority import CanonicalHITLAuthority
from packages.llm_adjudication import AzureShadowAdjudicationService
from packages.model_router.inputs import RouterInput
from packages.model_router.router import ModelRouter
from packages.ocr.adjudication import adjudicate_candidates
from packages.ocr.contracts import OCRRequest
from packages.ocr.execution import OCRExecutionService
from packages.ocr.ppocr_v5_provider import PPOCRv5Provider
from packages.ocr.source_b_routing import (
    ChallengerBudget,
    SourceBChallengeContext,
    route_to_ppocr_v5,
)
from packages.runtime_profile import DecisionServiceFactory
from packages.storage.object_store import ObjectStore, ObjectStoreSettings
from workers.page_detection.text_extraction import PaddleOCRTextExtractor, RapidOCRTextExtractor
from workers.retry.alternate_preprocessing import PreprocessingContext
from workers.retry.retry_service import retry_field
from workers.unstructured_extraction.layoutlmv3_adapter import (
    LayoutLMv3Adapter,
)
from workers.unstructured_extraction.table_transformer_adapter import TableTransformerAdapter
from workers.vlm_fallback.adapter import VLMAdapter

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "retry-worker"


def _field_type(field_name: str) -> str:
    name = field_name.casefold()
    if "date" in name:
        return "date"
    if any(token in name for token in ("charge", "amount", "total", "units")):
        return "currency" if "units" not in name else "number"
    if any(token in name for token in ("code", "npi", "id", "number")):
        return "code"
    return "text"


class RetryWorker:
    def __init__(
        self,
        event_bus: EventBus,
        object_store: ObjectStore,
        session_factory: sessionmaker,
        pipeline_version: str,
        vlm_enabled: bool = False,
        decision_service: EvidenceDecisionService | None = None,
        deterministic_service: DeterministicEvidenceService | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._object_store = object_store
        self._session_factory = session_factory
        self._pipeline_version = pipeline_version
        self._router = ModelRouter(vlm_enabled=vlm_enabled)
        self._vlm_enabled = vlm_enabled
        decision_bundle = DecisionServiceFactory.from_profile()
        self._decision_service = decision_service or decision_bundle.evidence_decision
        self._hitl_authority = CanonicalHITLAuthority()
        self._deterministic_service = deterministic_service or DeterministicEvidenceService()
        self._criticality = decision_bundle.criticality
        self._engine_cache: dict[str, object] = {}
        self._ocr_execution = OCRExecutionService()
        self._ppocr_provider = PPOCRv5Provider()
        self._challenger_budgets: dict[str, ChallengerBudget] = {}
        self._llm_shadow = AzureShadowAdjudicationService.from_env()

    def _engine(self, name: str, factory):
        """Lazily initialize each OCR/layout engine once per worker process."""
        if name not in self._engine_cache:
            self._engine_cache[name] = factory()
        return self._engine_cache[name]

    async def handle_one(self, envelope: EventEnvelope) -> None:
        document_id = envelope.document_id
        field_id_str = envelope.payload.get("field_id")
        if not document_id or not field_id_str:
            logger.warning("field.retry.requested event missing document_id or field_id, skipping")
            return

        with self._session_factory() as session:
            documents = DocumentRepository(session)
            outbox = SqlAlchemyOutboxRepository(session)

            document = documents.get(document_id)
            if document is None:
                logger.warning("document %s not found, skipping", document_id)
                return

            from sqlalchemy import select

            stmt = select(ExtractedFieldORM).where(ExtractedFieldORM.field_id == UUID(field_id_str))
            field_orm = session.execute(stmt).scalar_one_or_none()
            if not field_orm:
                logger.warning("field %s not found, skipping", field_id_str)
                return

            field = orm_to_extracted_field(field_orm)

            # Build attempted methods from candidates
            attempted = set()
            for cand in field.candidates:
                attempted.add(cand.source)
            attempted.add(field.extraction_method)

            router_input = RouterInput(
                field_name=field.field_name,
                field_criticality=FieldCriticality.CRITICAL
                if field.is_critical
                else FieldCriticality.NON_CRITICAL,
                ocr_confidence=field.confidence,
                validation_failed=bool(field.validation_reasons),
                ocr_disagreement=False,
                cache_hit=False,
                is_table_field=(field_orm.service_line_number is not None),
                is_unstructured_document=(document.bundle_type == "D_UNSTRUCTURED"),
                vlm_enabled=self._vlm_enabled,
                attempted_methods=frozenset(attempted),
            )

            decision = self._router.decide(router_input)
            requested_action = envelope.payload.get("next_action")
            preserved = envelope.payload.get("decision_context_evidence") or {}
            document_family = preserved.get("document_family") or (
                "UNSTRUCTURED"
                if document.bundle_type == "D_UNSTRUCTURED"
                else "UB04"
                if "ub" in (field.template_version or "").casefold()
                else "CMS1500"
            )
            approved_route = self._decision_service.production_route_for(
                document_family, field.field_name
            )
            present_families = {
                engine_family(candidate.source.value) for candidate in field.candidates
            }
            present_families.add(engine_family(field.extraction_method.value))
            independent_engine = (
                next(
                    (
                        engine
                        for engine in (
                            approved_route.primary_engine,
                            approved_route.confirmation_engine,
                        )
                        if engine_family(engine) not in present_families
                    ),
                    None,
                )
                if approved_route
                else None
            )
            if requested_action == NextAction.HUMAN_REVIEW.value:
                next_stage = ExtractionMethod.HUMAN_REVIEW
            elif (
                requested_action == NextAction.SECONDARY_OCR.value
                and independent_engine == "rapidocr"
            ):
                next_stage = ExtractionMethod.REGIONAL_RAPIDOCR
            elif (
                requested_action == NextAction.SECONDARY_OCR.value
                and independent_engine == "paddleocr"
            ):
                next_stage = ExtractionMethod.REGIONAL_PADDLEOCR
            elif requested_action == NextAction.SECONDARY_OCR.value:
                next_stage = ExtractionMethod.HUMAN_REVIEW
            else:
                next_stage = decision.selected_route
            budget_key = str(envelope.claim_id or document_id)
            budget = self._challenger_budgets.setdefault(
                budget_key,
                ChallengerBudget(int(envelope.payload.get("blocking_field_count") or 0)),
            )
            challenge_context = SourceBChallengeContext(
                source=str(envelope.payload.get("source_dataset") or "").upper(),
                document_family=document_family,
                current_claim_blocker=bool(envelope.payload.get("blocks_stp")),
                crop_safety_status=str(envelope.payload.get("crop_safety_status") or "UNSAFE"),
                primary_resolved=False,
                failure_reason=str((envelope.payload.get("reason_codes") or [""])[0]),
            )
            if os.getenv("PPOCR_V5_CHALLENGER_ENABLED", "true").casefold() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                routed, _ = route_to_ppocr_v5(challenge_context, budget)
                if routed:
                    next_stage = ExtractionMethod.PPOCR_V5_CHALLENGER
            new_text = ""
            challenger_force_hitl = False
            challenger_candidate = None
            challenger_adjudication = None

            if next_stage in (
                ExtractionMethod.ALTERNATE_PREPROCESS_OCR,
                ExtractionMethod.REGIONAL_RAPIDOCR,
                ExtractionMethod.REGIONAL_PADDLEOCR,
                ExtractionMethod.LAYOUTLMV3,
                ExtractionMethod.TABLE_TRANSFORMER,
                ExtractionMethod.VLM_FALLBACK,
                ExtractionMethod.PPOCR_V5_CHALLENGER,
            ):
                import io

                from PIL import Image

                from workers.document_preparation.codecs import decode_pdf_pages

                raw_bytes = await asyncio.to_thread(
                    self._object_store.get_bytes, document.original_object
                )
                if "pdf" in document.detected_format.value.lower():
                    pages = decode_pdf_pages(raw_bytes)
                    page_image = pages[field.page_number - 1].image
                else:
                    page_image = Image.open(io.BytesIO(raw_bytes))

                bbox = field.bounding_box
                if bbox:
                    region = (int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1))
                else:
                    region = (0, 0, page_image.width, page_image.height)

                new_confidence = 0.0
                new_text = ""
                new_source = next_stage

                try:
                    if next_stage == ExtractionMethod.PPOCR_V5_CHALLENGER:
                        crop = page_image.crop(region)
                        request = OCRRequest(
                            document_id=str(document_id),
                            page_number=field.page_number,
                            field_name=field.field_name,
                            field_type=_field_type(field.field_name),
                            form_type=ClaimFormType(document_family),
                            image=crop,
                            bounding_box=field.bounding_box,
                            criticality=(
                                FieldCriticality.CRITICAL
                                if field.is_critical
                                else FieldCriticality.NON_CRITICAL
                            ),
                            scope="FIELD_CROP",
                            preprocessing_profile="SOURCE_CROP",
                        )
                        result = await self._ocr_execution.execute(self._ppocr_provider, request)
                        challenger_candidate = result.candidates[0] if result.candidates else None
                        existing = ocr_candidates_from_field(field)
                        primary_candidate = next(
                            (item for item in existing if "rapidocr" in item.engine.casefold()),
                            existing[0] if existing else None,
                        )
                        challenger_adjudication = adjudicate_candidates(
                            field_name=field.field_name,
                            primary=primary_candidate,
                            challenger=challenger_candidate,
                            crop_safety_status=challenge_context.crop_safety_status,
                            deterministic=self._deterministic_service,
                        )
                        if challenger_candidate is not None:
                            new_text = challenger_candidate.raw_value
                            new_confidence = challenger_candidate.raw_confidence
                        if challenger_adjudication.action == "USE_CHALLENGER":
                            pass
                        elif challenger_adjudication.action == "KEEP_PRIMARY":
                            new_text = field.normalized_value or field.raw_value
                        else:
                            challenger_force_hitl = True
                    elif next_stage in {
                        ExtractionMethod.REGIONAL_RAPIDOCR,
                        ExtractionMethod.REGIONAL_PADDLEOCR,
                    }:
                        crop = page_image.crop(region)
                        extractor = self._engine(
                            next_stage.value,
                            RapidOCRTextExtractor
                            if next_stage == ExtractionMethod.REGIONAL_RAPIDOCR
                            else PaddleOCRTextExtractor,
                        )
                        words = await asyncio.to_thread(
                            extractor.extract_region,
                            crop,
                            0,
                            0,
                            crop.width,
                            crop.height,
                        )
                        new_text = " ".join(word.text for word in words).strip()
                        new_confidence = (
                            sum(word.confidence for word in words) / len(words) if words else 0
                        )
                    elif next_stage == ExtractionMethod.ALTERNATE_PREPROCESS_OCR:
                        extractor = self._engine("alternate_paddleocr", PaddleOCRTextExtractor)
                        quality = envelope.payload.get("image_quality") or {}
                        registration = envelope.payload.get("registration_evidence") or {}
                        reasons = field.validation_reasons or envelope.payload.get(
                            "reason_codes", []
                        )
                        context = PreprocessingContext(
                            field_type=_field_type(field.field_name),
                            quality_score=quality.get("overall_score")
                            if isinstance(quality, dict)
                            else None,
                            failure_reason=reasons[0] if reasons else None,
                            registration_confidence=(
                                registration.get("alignment_confidence")
                                if isinstance(registration, dict)
                                else None
                            ),
                        )
                        res = await asyncio.to_thread(
                            retry_field,
                            page_image,
                            region,
                            extractor,
                            field.confidence,
                            context,
                        )
                        if res.improved:
                            new_text = res.text
                            new_confidence = res.confidence
                    elif next_stage == ExtractionMethod.LAYOUTLMV3:
                        adapter = self._engine("layoutlmv3", LayoutLMv3Adapter)
                        res = await asyncio.to_thread(
                            adapter.extract, page_image, [field.field_name]
                        )
                        if res:
                            new_text = res[0].value
                            new_confidence = res[0].confidence
                    elif next_stage == ExtractionMethod.TABLE_TRANSFORMER:
                        adapter = self._engine("table_transformer", TableTransformerAdapter)
                    elif next_stage == ExtractionMethod.VLM_FALLBACK:
                        crop = page_image.crop(region)
                        adapter = self._engine("vlm_fallback", VLMAdapter)
                        from workers.vlm_fallback.schema import VLMFieldSchema

                        schema = VLMFieldSchema(
                            field_name=field.field_name, type="string", description=""
                        )
                        res = await asyncio.to_thread(adapter.extract_fields, crop, [schema], "")
                        if field.field_name in res:
                            new_text = str(res[field.field_name])
                            new_confidence = 1.0
                except Exception:
                    logger.exception("Failed to run adapter %s", next_stage)

                # Retry providers only append evidence. They never overwrite the
                # canonical value before the common decision service authorizes it.
                retry_evidence = FieldEvidence(
                    source=new_source,
                    raw_text=new_text,
                    confidence=new_confidence,
                    bounding_box=field.bounding_box,
                    model_name=(challenger_candidate.model_name if challenger_candidate else None),
                    model_version=(
                        challenger_candidate.model_version if challenger_candidate else None
                    ),
                    provenance=challenger_candidate.provenance if challenger_candidate else None,
                    tokens=tuple(
                        {
                            "text": token.text,
                            "confidence": token.confidence,
                            "bounding_box": token.bounding_box.model_dump(mode="json"),
                        }
                        for token in challenger_candidate.tokens
                    )
                    if challenger_candidate
                    else (),
                    adjudication_metadata=(
                        {
                            "quality_bucket": envelope.payload.get("source_quality_band"),
                            "failure_reason": challenge_context.failure_reason,
                            "primary_candidate": primary_candidate.raw_value
                            if primary_candidate
                            else None,
                            "challenger_candidate": challenger_candidate.raw_value
                            if challenger_candidate
                            else None,
                            "agreement_status": challenger_adjudication.agreement_status,
                            "adjudication_reason": challenger_adjudication.reason,
                            "challenger_removed_claim_blocker": False,
                        }
                        if challenger_adjudication
                        else None
                    ),
                )
                field.candidates.append(retry_evidence)
                field_orm.candidates = [
                    *(field_orm.candidates or []),
                    retry_evidence.model_dump(mode="json"),
                ]

                if next_stage == ExtractionMethod.ALTERNATE_PREPROCESS_OCR:
                    from packages.observability.metrics import retry_total

                    retry_total.labels(improved=str(bool(new_text))).inc()
                elif next_stage == ExtractionMethod.VLM_FALLBACK:
                    from packages.observability.metrics import vlm_invocation_total

                    vlm_invocation_total.labels(insufficient_evidence="false").inc()

            field_policy = self._decision_service.field_policy.for_field(
                document_family,
                field.field_name,
            )
            level = field_policy.criticality
            deterministic = self._deterministic_service.evaluate(
                field.field_name,
                field.normalized_value or field.raw_value,
            )
            # Candidate-specific evidence is only reusable when the retry agrees
            # with the canonical value. A conflicting retry must be selected and
            # validated in a subsequent decision pass.
            retry_agrees = not new_text or normalize_agreement_value(
                field.field_name,
                new_text,
            ) == normalize_agreement_value(
                field.field_name,
                field.normalized_value or field.raw_value,
            )
            hard_validation_passed = deterministic.passed and retry_agrees
            reference_payload = preserved.get("reference")
            decision = self._decision_service.decide(
                DecisionContext(
                    field_id=str(field.field_id),
                    field_name=field.field_name,
                    document_family=document_family,
                    criticality=level,
                    required=preserved.get("required", field_policy.required),
                    blocks_stp=preserved.get("blocks_stp", field_policy.blocks_stp),
                    requires_review_when_unresolved=preserved.get(
                        "requires_review_when_unresolved",
                        field_policy.requires_review_when_unresolved,
                    ),
                    candidates=ocr_candidates_from_field(field),
                    deterministic_evidence=(
                        deterministic.evidence if hard_validation_passed else set()
                    ),
                    deterministic_evidence_version=preserved.get(
                        "deterministic_evidence_version",
                        self._deterministic_service.policy_version,
                    ),
                    hard_validation_passed=hard_validation_passed,
                    registration_confidence=preserved.get("registration_confidence"),
                    structural_evidence_source=preserved.get("structural_evidence_source"),
                    structural_localization=preserved.get("structural_localization"),
                    wrong_crop_suspected="WRONG_CROP_SUSPECTED" in set(field.validation_reasons),
                    cross_field_evidence=set(preserved.get("cross_field_evidence", [])),
                    reference=(
                        ReferenceEvidence.model_validate(reference_payload)
                        if reference_payload
                        else None
                    ),
                    reference_source_state=ReferenceSourceState(
                        preserved.get("reference_source_state", "DISABLED")
                    ),
                )
            )
            if self._llm_shadow is not None and decision.disposition in {
                FieldDisposition.ESCALATE,
                FieldDisposition.HUMAN_REVIEW_REQUIRED,
                FieldDisposition.INSUFFICIENT_EVIDENCE,
            }:
                try:
                    self._llm_shadow.observe(
                        field_name=field.field_name,
                        field_type=_field_type(field.field_name),
                        candidates=[candidate.raw_text for candidate in field.candidates],
                        claim_blocking=bool(
                            envelope.payload.get("blocks_stp", decision.blocks_stp)
                        ),
                        crop_safe=str(envelope.payload.get("crop_safety_status") or "UNSAFE")
                        == "CROP_SAFE",
                        localization_confidence=float(
                            preserved.get("registration_confidence") or 0.0
                        ),
                        critical=field.is_critical,
                        authoritative_conflict=bool(
                            reference_payload and reference_payload.get("contradiction")
                        ),
                        page_key=f"{document_id}:{field.page_number}",
                        claim_key=str(envelope.claim_id or document_id),
                        claim_distance=max(
                            1, int(envelope.payload.get("blocking_field_count") or 1)
                        ),
                        evidence={
                            "semantic_section": envelope.payload.get("semantic_section"),
                            "conflict": bool(
                                reference_payload and reference_payload.get("contradiction")
                            ),
                        },
                    )
                except Exception:
                    logger.exception("Azure LLM shadow adjudication failed closed")
            accepted = not challenger_force_hitl and decision.disposition in {
                FieldDisposition.AUTO_ACCEPTED,
                FieldDisposition.REFERENCE_CONFIRMED,
            }
            if accepted and decision.selected_value is not None:
                field_orm.raw_value = decision.selected_value
                field_orm.normalized_value = decision.selected_value
                field_orm.confidence = decision.calibrated_probability
                field_orm.disposition = decision.disposition.value
                field_orm.validation_status = "VALID"
                field_orm.validation_reasons = decision.reason_codes
                env_out = EventEnvelope(
                    event_type=Topic.EXTRACTION_COMPLETED.value,
                    correlation_id=envelope.correlation_id,
                    document_id=document_id,
                    claim_id=document.claim_id,
                    pipeline_version=self._pipeline_version,
                    payload={
                        "field_id": str(field.field_id),
                        "decision_policy": decision.policy_version,
                    },
                )
                await outbox.add(
                    OutboxRecord(
                        topic=Topic.EXTRACTION_COMPLETED.value,
                        envelope=env_out,
                        partition_key=str(document_id),
                    )
                )
            elif (
                next_stage == ExtractionMethod.HUMAN_REVIEW
                or decision.next_action is NextAction.HUMAN_REVIEW
            ):
                from packages.observability.metrics import human_review_total

                reason_str = decision.reason_codes[0] if decision.reason_codes else "unknown"
                human_review_total.labels(reason=reason_str).inc()

                env_out = self._hitl_authority.create_field_review_event(
                    correlation_id=envelope.correlation_id,
                    document_id=document_id,
                    claim_id=document.claim_id,
                    pipeline_version=self._pipeline_version,
                    field=field,
                    decision=decision,
                    required_policy=self._decision_service.evidence_policy.version,
                    blocks_stp=bool(envelope.payload.get("blocks_stp", decision.blocks_stp)),
                    blocking_field_count=int(envelope.payload.get("blocking_field_count", 0)),
                    claim_unlock_value=float(envelope.payload.get("claim_unlock_value", 0)),
                    single_blocker_claim=bool(envelope.payload.get("single_blocker_claim", False)),
                )
                await outbox.add(
                    OutboxRecord(
                        topic=Topic.HUMAN_REVIEW_REQUESTED.value,
                        envelope=env_out,
                        partition_key=str(document_id),
                    )
                )
            else:
                env_out = EventEnvelope(
                    event_type=Topic.FIELD_RETRY_REQUESTED.value,
                    correlation_id=envelope.correlation_id,
                    document_id=document_id,
                    claim_id=document.claim_id,
                    pipeline_version=self._pipeline_version,
                    payload={
                        "field_id": str(field.field_id),
                        "field_name": field.field_name,
                        "hard_validation_passed": hard_validation_passed,
                        "reason_codes": decision.reason_codes,
                        "policy_version": decision.policy_version,
                        "decision_context_evidence": preserved,
                        "blocks_stp": bool(envelope.payload.get("blocks_stp", decision.blocks_stp)),
                        "blocking_field_count": int(
                            envelope.payload.get("blocking_field_count", 0)
                        ),
                        "claim_unlock_value": float(envelope.payload.get("claim_unlock_value", 0)),
                        "single_blocker_claim": bool(
                            envelope.payload.get("single_blocker_claim", False)
                        ),
                        "claim_impact": envelope.payload.get("claim_impact"),
                    },
                )
                await outbox.add(
                    OutboxRecord(
                        topic=Topic.FIELD_RETRY_REQUESTED.value,
                        envelope=env_out,
                        partition_key=str(document_id),
                    )
                )

            document.updated_at = datetime.now(UTC)
            documents.update(document)
            session.commit()

    async def run_forever(self) -> None:
        async for _topic, envelope in self._event_bus.subscribe(
            [Topic.FIELD_RETRY_REQUESTED.value], group_id=CONSUMER_GROUP
        ):
            try:
                await self.handle_one(envelope)
            except Exception:
                logger.exception("failed to retry field")


async def _run(worker: RetryWorker, relay) -> None:
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

    configure_logging("retry-worker")
    settings = get_settings()

    import os

    vlm_enabled = os.environ.get("VLM_ENABLED", "false").lower() == "true"

    object_store = ObjectStore(
        ObjectStoreSettings(
            endpoint_url=settings.object_store_endpoint,
            access_key=settings.object_store_access_key,
            secret_key=settings.object_store_secret_key,
            use_ssl=settings.object_store_use_ssl,
        )
    )

    session_factory = make_session_factory(settings.database_url)
    event_bus = AIOKafkaEventBus(settings.kafka_bootstrap_servers)
    worker = RetryWorker(
        event_bus=event_bus,
        object_store=object_store,
        session_factory=session_factory,
        pipeline_version=settings.pipeline_version,
        vlm_enabled=vlm_enabled,
    )
    # RetryWorker writes HUMAN_REVIEW_REQUESTED/etc. rows to the outbox
    # table; a relay must run alongside the consumer loop to actually
    # publish those rows to Kafka. See packages/events/outbox.py.
    relay = OutboxRelay(
        repository=PollingOutboxRepository(session_factory),
        event_bus=event_bus,
    )
    asyncio.run(_run(worker, relay))


if __name__ == "__main__":
    main()
