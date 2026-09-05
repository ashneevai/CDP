"""Sole canonical producer of final ProcessingRoute decisions."""

from __future__ import annotations

from packages.document_taxonomy.contracts import DocumentClassification
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.processing_routes.resolver import ProcessingRouteResolver
from packages.standard_form_verification.evidence import (
    StandardFormEvidence,
    evidence_from_router_features,
)
from packages.standard_form_verification.service import StandardFormVerificationService

from .contracts import DocumentRoutingDecision
from .evidence import from_routing_evidence
from .hierarchical import DeterministicHierarchicalBaseline
from .router import RoutingEvidence


class DocumentRoutingDecisionService:
    version = "document-routing-decision-v2"

    def __init__(self, classifier=None, verification_service=None, route_resolver=None):
        self.classifier = classifier or DeterministicHierarchicalBaseline()
        self.verification_service = verification_service or StandardFormVerificationService()
        self.route_resolver = route_resolver or ProcessingRouteResolver()

    def decide(
        self,
        document_id: str,
        page_id: str,
        routing_evidence: RoutingEvidence,
        standard_evidence: StandardFormEvidence | None = None,
        evaluation_only: bool = False,
    ) -> DocumentRoutingDecision:
        classification = self.classifier.classify(
            document_id, page_id, from_routing_evidence(routing_evidence)
        )
        if classification.standard_candidate and standard_evidence is None:
            standard_evidence = evidence_from_router_features(
                classification.document_subtype, None, routing_evidence
            )
        return self.decide_classification(
            classification, standard_evidence, evaluation_only=evaluation_only
        )

    def decide_classification(
        self,
        classification: DocumentClassification,
        standard_evidence: StandardFormEvidence | None = None,
        evaluation_only: bool = False,
    ) -> DocumentRoutingDecision:
        verification = None
        if classification.standard_candidate:
            candidate = classification.document_subtype
            if standard_evidence is None or standard_evidence.candidate_family != candidate:
                standard_evidence = StandardFormEvidence(
                    candidate_family=candidate,
                    contradiction_codes=("STANDARD_VERIFICATION_EVIDENCE_MISSING",),
                )
            verification = self.verification_service.verify(standard_evidence)
        route = self.route_resolver.resolve(classification, verification)
        return DocumentRoutingDecision(
            classification=classification,
            standard_verification=verification,
            processing_route=route.route,
            route_reason_codes=route.reason_codes,
            decision_service_version=self.version,
            evaluation_only=evaluation_only,
        )

    def decide_nomination(
        self,
        *,
        document_id: str,
        page_id: str,
        nominated_family: DocumentClass | None,
        structured: bool,
        claim_related: bool,
        non_claim: bool,
        confidence: float,
        supporting_evidence: tuple[str, ...],
        standard_evidence: StandardFormEvidence | None = None,
        evaluation_only: bool = False,
    ) -> DocumentRoutingDecision:
        if non_claim:
            top = family = DocumentClass.NON_CLAIM
            subtype = DocumentClass.OTHER_NON_CLAIM
        elif nominated_family in {DocumentClass.CMS1500, DocumentClass.UB04}:
            top, family, subtype = (
                DocumentClass.CLAIM,
                DocumentClass.STANDARD_CLAIM,
                nominated_family,
            )
        elif structured and claim_related:
            top = DocumentClass.CLAIM
            family = DocumentClass.NON_STANDARD_CLAIM
            subtype = DocumentClass.OTHER_CLAIM_FORM
        else:
            top = family = subtype = DocumentClass.UNKNOWN
        classification = DocumentClassification(
            document_id=document_id,
            page_id=page_id,
            top_level_class=top,
            document_family=family,
            document_subtype=subtype,
            structured=structured,
            claim_related=claim_related,
            standard_candidate=nominated_family in {DocumentClass.CMS1500, DocumentClass.UB04},
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            ambiguity_reason=(
                "NO_VERIFIED_TAXONOMY_SUBTYPE" if top == DocumentClass.UNKNOWN else None
            ),
            classifier_version="legacy-nomination-bridge-v1",
        )
        return self.decide_classification(
            classification, standard_evidence, evaluation_only=evaluation_only
        )
