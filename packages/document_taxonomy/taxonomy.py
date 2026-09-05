"""Canonical business-semantic document taxonomy. No routing implementation lives here."""

from __future__ import annotations

from enum import StrEnum

from packages.domain.common import DomainModel


class DocumentClass(StrEnum):
    DOCUMENT = "DOCUMENT"
    CLAIM = "CLAIM"
    STANDARD_CLAIM = "STANDARD_CLAIM"
    CMS1500 = "CMS1500"
    UB04 = "UB04"
    NON_STANDARD_CLAIM = "NON_STANDARD_CLAIM"
    OTHER_CLAIM_FORM = "OTHER_CLAIM_FORM"
    CUSTOM_PROFESSIONAL = "CUSTOM_PROFESSIONAL"
    CUSTOM_INSTITUTIONAL = "CUSTOM_INSTITUTIONAL"
    OTHER_STRUCTURED_CLAIM = "OTHER_STRUCTURED_CLAIM"
    CLAIM_SUPPORT = "CLAIM_SUPPORT"
    EOB = "EOB"
    ITEMIZED_BILL = "ITEMIZED_BILL"
    MEDICAL_INVOICE = "MEDICAL_INVOICE"
    LAB_REPORT = "LAB_REPORT"
    CLINICAL_NOTE = "CLINICAL_NOTE"
    CORRESPONDENCE = "CORRESPONDENCE"
    OTHER_ATTACHMENT = "OTHER_ATTACHMENT"
    NON_CLAIM = "NON_CLAIM"
    COVER_PAGE = "COVER_PAGE"
    DOCUMENT_SEPARATOR = "DOCUMENT_SEPARATOR"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    BLANK_OR_NEAR_BLANK = "BLANK_OR_NEAR_BLANK"
    OTHER_NON_CLAIM = "OTHER_NON_CLAIM"
    UNKNOWN = "UNKNOWN"


PARENT: dict[DocumentClass, DocumentClass | None] = {
    DocumentClass.DOCUMENT: None,
    DocumentClass.CLAIM: DocumentClass.DOCUMENT,
    DocumentClass.STANDARD_CLAIM: DocumentClass.CLAIM,
    DocumentClass.CMS1500: DocumentClass.STANDARD_CLAIM,
    DocumentClass.UB04: DocumentClass.STANDARD_CLAIM,
    DocumentClass.NON_STANDARD_CLAIM: DocumentClass.CLAIM,
    DocumentClass.OTHER_CLAIM_FORM: DocumentClass.NON_STANDARD_CLAIM,
    DocumentClass.CUSTOM_PROFESSIONAL: DocumentClass.NON_STANDARD_CLAIM,
    DocumentClass.CUSTOM_INSTITUTIONAL: DocumentClass.NON_STANDARD_CLAIM,
    DocumentClass.OTHER_STRUCTURED_CLAIM: DocumentClass.NON_STANDARD_CLAIM,
    DocumentClass.CLAIM_SUPPORT: DocumentClass.DOCUMENT,
    DocumentClass.EOB: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.ITEMIZED_BILL: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.MEDICAL_INVOICE: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.LAB_REPORT: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.CLINICAL_NOTE: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.CORRESPONDENCE: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.OTHER_ATTACHMENT: DocumentClass.CLAIM_SUPPORT,
    DocumentClass.NON_CLAIM: DocumentClass.DOCUMENT,
    DocumentClass.COVER_PAGE: DocumentClass.NON_CLAIM,
    DocumentClass.DOCUMENT_SEPARATOR: DocumentClass.NON_CLAIM,
    DocumentClass.ADMINISTRATIVE: DocumentClass.NON_CLAIM,
    DocumentClass.BLANK_OR_NEAR_BLANK: DocumentClass.NON_CLAIM,
    DocumentClass.OTHER_NON_CLAIM: DocumentClass.NON_CLAIM,
    DocumentClass.UNKNOWN: DocumentClass.DOCUMENT,
}


class TaxonomyNode(DomainModel):
    code: DocumentClass
    parent: DocumentClass | None
    is_leaf: bool


class DocumentTaxonomyV1:
    version = "document-taxonomy-v1.0.0"

    @classmethod
    def parent_of(cls, code: DocumentClass) -> DocumentClass | None:
        return PARENT[code]

    @classmethod
    def ancestors(cls, code: DocumentClass) -> tuple[DocumentClass, ...]:
        result = []
        while (code := PARENT[code]) is not None:
            result.append(code)
        return tuple(result)

    @classmethod
    def children_of(cls, code: DocumentClass) -> tuple[DocumentClass, ...]:
        return tuple(child for child, parent in PARENT.items() if parent == code)

    @classmethod
    def nodes(cls) -> tuple[TaxonomyNode, ...]:
        return tuple(
            TaxonomyNode(code=code, parent=parent, is_leaf=not cls.children_of(code))
            for code, parent in PARENT.items()
        )

    @classmethod
    def validate_label(cls, label: DocumentClass, parent_labels: tuple[DocumentClass, ...]) -> None:
        required = cls.ancestors(label)
        if tuple(parent_labels) != tuple(reversed(required)):
            raise ValueError(f"parent path for {label} must be {tuple(reversed(required))}")
