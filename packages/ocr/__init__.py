"""Shared, engine-neutral OCR contracts."""

from packages.ocr.contracts import (
    OCRCandidate,
    OCREngine,
    OCRProvider,
    OCRRequest,
    OCRResult,
    OCRToken,
)
from packages.ocr.execution import OCRExecutionService
from packages.ocr.provenance import EvidenceProvenance
from packages.ocr.rapidocr_provider import RapidOCRProvider

__all__ = [
    "EvidenceProvenance",
    "OCRCandidate",
    "OCREngine",
    "OCRExecutionService",
    "OCRProvider",
    "OCRRequest",
    "OCRResult",
    "OCRToken",
    "RapidOCRProvider",
]
