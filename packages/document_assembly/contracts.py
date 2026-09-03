from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PageSignal:
    page_id: str
    page_index: int
    document_class: str
    confidence: float
    is_blank: bool = False
    is_duplicate: bool = False
    is_cover_sheet: bool = False
    continuation_hint: bool = False
    explicit_boundary_before: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssembledDocument:
    document_id: str
    document_class: str
    page_ids: tuple[str, ...]
    start_page_index: int
    end_page_index: int
    confidence: float
    boundary_reason: str


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    documents: tuple[AssembledDocument, ...]
    ignored_page_ids: tuple[str, ...]
    uncertain_page_ids: tuple[str, ...]
