"""Governed OCR execution with mandatory candidate provenance."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from packages.ocr.contracts import OCRCandidate, OCRProvider, OCRRequest, OCRResult
from packages.ocr.independence import independence_group
from packages.ocr.provenance import EvidenceProvenance
from packages.ocr_cache import InMemoryOCRCache, OCRCacheEntry, ocr_cache_key


class OCRExecutionService:
    """The single boundary for primary and secondary local OCR providers."""

    version = "ocr-execution-service-v1"

    def __init__(self, cache: InMemoryOCRCache | None = None) -> None:
        self._cache = cache or InMemoryOCRCache()

    async def execute(self, provider: OCRProvider, request: OCRRequest) -> OCRResult:
        provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        provider_version = str(getattr(provider, "provider_version", "unknown"))
        key = ocr_cache_key(
            crop_bytes=request.image.convert("RGB").tobytes(),
            engine=str(provider_name), model_version=provider_version,
            preprocessing_version=request.preprocessing_profile or "AUTO",
            configuration={
                "document_id": request.document_id,
                "page_number": request.page_number,
                "field_name": request.field_name, "field_type": request.field_type,
                "scope": request.scope,
            }, page_hash=request.page_sha256,
            region_bbox=tuple(round(value) for value in request.bounding_box.normalized()),
        )
        cached = self._cache.get(key)
        if cached is not None and isinstance(cached.value, OCRResult):
            return replace(cached.value, cache_hit=True, execution_cache_key=key)
        result = await provider.extract(request)
        invocation_id = str(uuid4())
        crop_pixels = request.image.convert("RGB").tobytes()
        crop_hash = sha256(crop_pixels).hexdigest()
        location = request.localization_evidence
        localization_id = (
            location.candidate_region_hash if location else
            f"{request.document_id}:{request.page_number}:{request.field_name}:"
            f"{','.join(str(value) for value in request.bounding_box.normalized())}"
        )
        candidates = tuple(
            replace(candidate, provenance=self._complete(
                candidate, request, invocation_id, crop_hash, localization_id
            ))
            for candidate in result.candidates
        )
        completed = OCRResult(
            candidates=candidates,
            provider=result.provider,
            provider_version=result.provider_version,
            latency_ms=result.latency_ms,
            execution_cache_key=key,
            cache_hit=False,
        )
        stored = self._cache.put_if_absent(
            key, OCRCacheEntry(value=completed, evidence_reference=f"ocr-cache:{key}")
        )
        return stored.value if isinstance(stored.value, OCRResult) else completed

    def _complete(
        self,
        candidate: OCRCandidate,
        request: OCRRequest,
        invocation_id: str,
        crop_hash: str,
        localization_id: str,
    ) -> EvidenceProvenance:
        existing = candidate.provenance or EvidenceProvenance()
        location = request.localization_evidence
        preprocessing = candidate.preprocessing_variant or request.preprocessing_profile or "DEFAULT"
        preprocessing_hash = sha256(
            f"{preprocessing}|{candidate.preprocessing_version}|{crop_hash}".encode()
        ).hexdigest()
        return EvidenceProvenance(
            **{
                **existing.model_dump(),
                "document_sha256": existing.document_sha256 or request.document_sha256,
                "page_sha256": existing.page_sha256 or request.page_sha256,
                "source_representation_id": (
                    existing.source_representation_id or request.source_representation_id
                ),
                "observation_id": existing.observation_id
                or f"{request.document_id}:{request.page_number}",
                "crop_sha256": existing.crop_sha256 or crop_hash,
                "localization_id": existing.localization_id or localization_id,
                "localization_region_id": existing.localization_region_id or localization_id,
                "localization_method": existing.localization_method
                or (location.region_source if location else request.scope),
                "localization_version": existing.localization_version
                or (location.locator_version if location else "request-bbox-v1"),
                "preprocessing_profile": existing.preprocessing_profile or preprocessing,
                "preprocessing_sha256": existing.preprocessing_sha256 or preprocessing_hash,
                "preprocessing_version": existing.preprocessing_version
                or candidate.preprocessing_version,
                "engine_family": existing.engine_family or independence_group(candidate.engine),
                "engine_name": existing.engine_name or candidate.engine,
                "engine_version": existing.engine_version or candidate.model_version,
                "model_name": existing.model_name or candidate.model_name,
                "model_version": existing.model_version or candidate.model_version,
                "invocation_id": existing.invocation_id or invocation_id,
                "source_candidate_id": existing.source_candidate_id
                or f"{invocation_id}:{candidate.engine}",
                "bbox": existing.bbox or candidate.bounding_box,
                "produced_at": existing.produced_at or datetime.now(UTC),
            }
        )
