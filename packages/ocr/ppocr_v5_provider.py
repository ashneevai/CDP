"""Lazy PP-OCRv5 field-crop provider.

The challenger deliberately implements the same OCRProvider contract and is
executed only through OCRExecutionService. Models are pooled per language for
the lifetime of the worker process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from importlib import metadata
from threading import Lock
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from packages.domain.common import BoundingBox
from packages.ocr.contracts import OCRCandidate, OCRRequest, OCRResult, OCRToken


class UnsafeChallengerScopeError(ValueError):
    """Raised before inference when a challenger request is not a field crop."""


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


class PPOCRv5Provider:
    provider_name = "ppocr-v5"
    _pool: ClassVar[dict[str, Callable[[np.ndarray], Any]]] = {}
    _pool_lock: ClassVar[Lock] = Lock()

    def __init__(self, *, language: str = "en", backend: Callable[[np.ndarray], Any] | None = None):
        self.language = language
        self._backend = backend
        self.provider_version = _version("paddleocr")

    def _load_backend(self) -> Callable[[np.ndarray], Any]:
        if self._backend is not None:
            return self._backend
        with self._pool_lock:
            backend = self._pool.get(self.language)
            if backend is None:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "PP-OCRv5 is not installed; install the 'ppocr-v5' runtime extra"
                    ) from exc
                pipeline = PaddleOCR(
                    lang=self.language,
                    ocr_version="PP-OCRv5",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
                backend = pipeline.predict
                self._pool[self.language] = backend
            return backend

    @staticmethod
    def _enforce_scope(request: OCRRequest) -> None:
        if request.scope != "FIELD_CROP":
            raise UnsafeChallengerScopeError(
                "PP-OCRv5 challenger execution is restricted to safe field crops"
            )

    @staticmethod
    def _payload(result: Any) -> dict[str, Any]:
        if hasattr(result, "json"):
            result = result.json
        if isinstance(result, dict) and isinstance(result.get("res"), dict):
            return result["res"]
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _box(points: Any, request: OCRRequest, width: int, height: int) -> BoundingBox:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        crop_width = request.bounding_box.x1 - request.bounding_box.x0
        crop_height = request.bounding_box.y1 - request.bounding_box.y0
        return BoundingBox(
            x0=request.bounding_box.x0 + min(xs) * crop_width / width,
            y0=request.bounding_box.y0 + min(ys) * crop_height / height,
            x1=request.bounding_box.x0 + max(xs) * crop_width / width,
            y1=request.bounding_box.y0 + max(ys) * crop_height / height,
            image_width=request.bounding_box.image_width,
            image_height=request.bounding_box.image_height,
        )

    def _extract_sync(self, request: OCRRequest) -> OCRResult:
        self._enforce_scope(request)
        started = perf_counter()
        image = np.asarray(request.image.convert("RGB"))
        raw_results = self._load_backend()(image)
        raw_results = list(raw_results) if raw_results is not None else []
        payload = self._payload(raw_results[0]) if raw_results else {}
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
        height, width = image.shape[:2]
        tokens = tuple(
            OCRToken(
                text=str(text),
                confidence=float(scores[index]) if index < len(scores) else 0.0,
                bounding_box=self._box(polygons[index], request, width, height),
            )
            for index, text in enumerate(texts)
            if index < len(polygons) and polygons[index] is not None
        )
        joined = " ".join(str(text) for text in texts).strip()
        confidence = (
            sum(float(score) for score in scores) / len(scores) if scores else 0.0
        )
        latency = (perf_counter() - started) * 1000
        candidate = OCRCandidate(
            value=joined or None,
            raw_value=joined,
            engine=self.provider_name,
            model_name="PP-OCRv5",
            model_version=self.provider_version,
            preprocessing_variant=request.preprocessing_profile or "SOURCE_CROP",
            raw_confidence=confidence,
            calibrated_confidence=None,
            bounding_box=request.bounding_box,
            latency_ms=latency,
            validation_results=("SCOPE_FIELD_CROP", "CHALLENGER_ONLY"),
            preprocessing_version="source-crop-v1",
            tokens=tokens,
        )
        return OCRResult(
            candidates=(candidate,) if joined else (),
            provider=self.provider_name,
            provider_version=self.provider_version,
            latency_ms=latency,
        )

    async def extract(self, request: OCRRequest) -> OCRResult:
        return await asyncio.to_thread(self._extract_sync, request)
