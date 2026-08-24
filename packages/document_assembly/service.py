from __future__ import annotations

from hashlib import sha256
from statistics import fmean

from .contracts import AssembledDocument, AssemblyResult, PageSignal


class DocumentAssemblyService:
    """Group classified pages into logical documents without making claim decisions.

    The service is intentionally conservative: uncertain/blank/duplicate pages are
    surfaced explicitly rather than force-grouped into a claim document.
    """

    def __init__(self, *, minimum_confidence: float = 0.65) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be within [0, 1]")
        self.minimum_confidence = minimum_confidence

    def assemble(self, pages: list[PageSignal]) -> AssemblyResult:
        ordered = sorted(pages, key=lambda page: page.page_index)
        ignored: list[str] = []
        uncertain: list[str] = []
        documents: list[AssembledDocument] = []
        current: list[PageSignal] = []
        boundary_reason = "PACKAGE_START"

        def flush(reason: str) -> None:
            nonlocal current, boundary_reason
            if not current:
                boundary_reason = reason
                return
            document_class = self._dominant_class(current)
            page_ids = tuple(page.page_id for page in current)
            digest = sha256("|".join(page_ids).encode("utf-8")).hexdigest()[:20]
            documents.append(
                AssembledDocument(
                    document_id=f"doc_{digest}",
                    document_class=document_class,
                    page_ids=page_ids,
                    start_page_index=current[0].page_index,
                    end_page_index=current[-1].page_index,
                    confidence=fmean(page.confidence for page in current),
                    boundary_reason=boundary_reason,
                )
            )
            current = []
            boundary_reason = reason

        for page in ordered:
            if page.is_duplicate or page.is_blank:
                flush("IGNORED_PAGE_BOUNDARY")
                ignored.append(page.page_id)
                continue
            if page.confidence < self.minimum_confidence:
                flush("UNCERTAIN_PAGE_BOUNDARY")
                uncertain.append(page.page_id)
                continue
            if page.is_cover_sheet:
                flush("COVER_SHEET_BOUNDARY")
                documents.append(
                    AssembledDocument(
                        document_id=f"doc_{sha256(page.page_id.encode('utf-8')).hexdigest()[:20]}",
                        document_class=page.document_class,
                        page_ids=(page.page_id,),
                        start_page_index=page.page_index,
                        end_page_index=page.page_index,
                        confidence=page.confidence,
                        boundary_reason="COVER_SHEET",
                    )
                )
                boundary_reason = "AFTER_COVER_SHEET"
                continue
            if page.explicit_boundary_before:
                flush("EXPLICIT_BOUNDARY")
            elif current and not page.continuation_hint and page.document_class != current[-1].document_class:
                flush("CLASS_CHANGE")
            current.append(page)

        flush("PACKAGE_END")
        return AssemblyResult(
            documents=tuple(documents),
            ignored_page_ids=tuple(ignored),
            uncertain_page_ids=tuple(uncertain),
        )

    @staticmethod
    def _dominant_class(pages: list[PageSignal]) -> str:
        scores: dict[str, float] = {}
        for page in pages:
            scores[page.document_class] = scores.get(page.document_class, 0.0) + page.confidence
        return max(scores, key=scores.get)
