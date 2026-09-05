from packages.document_routing.router import MultiSignalRoute, RoutingEvidence
from packages.domain.common import DomainModel


class HierarchicalRoutingEvidence(DomainModel):
    legacy_route: MultiSignalRoute
    structured: bool
    claim_related: bool
    confidence: float
    supporting_codes: tuple[str, ...]
    contradicting_codes: tuple[str, ...] = ()


def from_routing_evidence(evidence: RoutingEvidence) -> HierarchicalRoutingEvidence:
    conflict_families = (
        (evidence.route.value,)
        if evidence.route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}
        else tuple(evidence.conflicting_anchors)
    )
    return HierarchicalRoutingEvidence(
        legacy_route=evidence.route,
        structured=evidence.route
        in {
            MultiSignalRoute.CMS1500,
            MultiSignalRoute.UB04,
            MultiSignalRoute.UNKNOWN_STRUCTURED,
            MultiSignalRoute.OTHER_CLAIM_FORM,
        },
        claim_related=evidence.route not in {MultiSignalRoute.NON_CLAIM},
        confidence=evidence.confidence,
        supporting_codes=tuple(evidence.reason_codes),
        contradicting_codes=tuple(
            dict.fromkeys(
                code
                for family in conflict_families
                for code in evidence.conflicting_anchors.get(family, ())
            )
        ),
    )