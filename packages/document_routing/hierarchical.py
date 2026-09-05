"""Deterministic stage baseline. It nominates; it never authorizes extraction."""

from packages.document_taxonomy.contracts import DocumentClassification
from packages.document_taxonomy.taxonomy import DocumentClass

from .evidence import HierarchicalRoutingEvidence
from .router import MultiSignalRoute


class DeterministicHierarchicalBaseline:
    classifier_version = "deterministic-hierarchical-baseline-v1"

    def classify(
        self, document_id: str, page_id: str, evidence: HierarchicalRoutingEvidence
    ) -> DocumentClassification:
        route = evidence.legacy_route
        if route == MultiSignalRoute.NON_CLAIM:
            top, family, subtype = (
                DocumentClass.NON_CLAIM,
                DocumentClass.NON_CLAIM,
                DocumentClass.OTHER_NON_CLAIM,
            )
        elif route == MultiSignalRoute.OTHER_CLAIM_FORM:
            top = DocumentClass.CLAIM
            family = DocumentClass.NON_STANDARD_CLAIM
            subtype = DocumentClass.OTHER_CLAIM_FORM
        elif route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
            top, family = DocumentClass.CLAIM, DocumentClass.STANDARD_CLAIM
            subtype = DocumentClass(route.value)
        else:
            # Legacy UNKNOWN_* does not prove CLAIM vs SUPPORT. Preserve structure and abstain taxonomy.
            top = family = subtype = DocumentClass.UNKNOWN
        return DocumentClassification(
            document_id=document_id,
            page_id=page_id,
            top_level_class=top,
            document_family=family,
            document_subtype=subtype,
            structured=evidence.structured,
            claim_related=evidence.claim_related,
            standard_candidate=route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04},
            confidence=evidence.confidence,
            supporting_evidence=evidence.supporting_codes,
            contradicting_evidence=evidence.contradicting_codes,
            ambiguity_reason=(
                "LEGACY_ROUTE_DOES_NOT_RESOLVE_TAXONOMY" if top == DocumentClass.UNKNOWN else None
            ),
            classifier_version=self.classifier_version,
        )
