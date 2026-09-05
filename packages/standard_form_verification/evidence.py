from pydantic import Field

from packages.document_taxonomy.taxonomy import DocumentClass
from packages.domain.common import DomainModel


class AnchorRegionEvidence(DomainModel):
    family: DocumentClass
    anchor: str
    region_id: str
    anchor_class: str
    match_type: str
    ocr_confidence: float = Field(ge=0, le=1)
    geometry_score: float = Field(ge=0, le=1)
    phrase_score: float = Field(ge=0, le=1)


class StandardFormEvidence(DomainModel):
    candidate_family: DocumentClass
    page_geometry_score: float = Field(default=0, ge=0, le=1)
    region_layout_scores: dict[str, float] = Field(default_factory=dict)
    service_grid_score: float = Field(default=0, ge=0, le=1)
    high_value_anchor_score: float = Field(default=0, ge=0, le=1)
    high_value_anchor_count: int = Field(default=0, ge=0)
    independent_region_count: int = Field(default=0, ge=0)
    spatial_relationship_score: float = Field(default=0, ge=0, le=1)
    structure_score: float = Field(default=0, ge=0, le=1)
    standard_score: float = Field(default=0, ge=0, le=1)
    family_margin: float = Field(default=0, ge=-1, le=1)
    template_registration_score: float | None = Field(default=None, ge=0, le=1)
    repeating_row_score: float = Field(default=0, ge=0, le=1)
    contradiction_codes: tuple[str, ...] = ()
    visual_probability: float | None = Field(default=None, ge=0, le=1)
    template_version: str | None = None
    canonical_identity_confirmed: bool = False
    identity_status: str = "UNKNOWN"
    identity_family: DocumentClass | None = None
    identity_policy_version: str = "strict-form-identity-v2"
    authorization_path: str | None = None
    matched_identity_anchors: tuple[str, ...] = ()
    missing_required_anchors: tuple[str, ...] = ()
    field_topology_score: float = Field(default=0, ge=0, le=1)
    per_anchor_evidence: tuple[AnchorRegionEvidence, ...] = ()


def evidence_from_router_features(
    candidate_family: DocumentClass,
    feature_bundle,
    routing_evidence,
    template_registration_score: float | None = None,
    template_version: str | None = None,
) -> StandardFormEvidence:
    del feature_bundle
    family = candidate_family.value
    opposing = "UB04" if family == "CMS1500" else "CMS1500"
    structure = routing_evidence.standard_structure
    geometry = routing_evidence.anchor_geometry_score.get(family, 0.0)
    anchors = routing_evidence.weighted_anchor_coverage.get(family, 0.0)
    anchor_records = tuple(
        AnchorRegionEvidence(
            family=candidate_family,
            anchor=item["anchor"],
            region_id=item["region_id"],
            anchor_class=item["anchor_class"],
            match_type=item["match_type"],
            ocr_confidence=item.get("ocr_confidence", 0.0),
            geometry_score=item["geometry_score"],
            phrase_score=item["phrase_score"],
        )
        for item in routing_evidence.anchor_geometry_evidence
        if item.get("family") == family
    )
    region_scores: dict[str, float] = {}
    for item in anchor_records:
        region_scores[item.region_id] = max(
            region_scores.get(item.region_id, 0.0), item.geometry_score
        )
    family_policy = routing_evidence.family_eligibility.get(family, {})
    contradictions = tuple(
        dict.fromkeys(
            [
                *routing_evidence.conflicting_anchors.get(family, []),
                *(
                    [f"{opposing}_IDENTITY_CONFIRMED"]
                    if routing_evidence.identity_state.get(opposing) == "CONFIRMED"
                    else []
                ),
            ]
        )
    )
    return StandardFormEvidence(
        candidate_family=candidate_family,
        page_geometry_score=structure.get("aspect_score", 0.0),
        region_layout_scores=region_scores,
        service_grid_score=(
            structure.get(family, 0.0)
            if candidate_family == DocumentClass.CMS1500
            else structure.get("service_table_score", 0.0)
        ),
        high_value_anchor_score=anchors,
        high_value_anchor_count=sum(
            item.anchor_class.startswith("HIGH") for item in anchor_records
        ),
        independent_region_count=len(region_scores),
        spatial_relationship_score=geometry,
        structure_score=structure.get(family, 0.0),
        standard_score=routing_evidence.scores.get(family, 0.0),
        family_margin=(
            routing_evidence.scores.get(family, 0.0)
            - routing_evidence.scores.get(opposing, 0.0)
        ),
        template_registration_score=template_registration_score,
        repeating_row_score=structure.get("v4_service_table_repetition", 0.0),
        template_version=template_version,
        canonical_identity_confirmed=routing_evidence.identity_state.get(family) == "CONFIRMED",
        identity_status=routing_evidence.identity_state.get(family, "UNKNOWN"),
        identity_family=(
            candidate_family
            if routing_evidence.identity_state.get(family) == "CONFIRMED"
            else None
        ),
        identity_policy_version=routing_evidence.identity_policy_version,
        authorization_path=family_policy.get("authorization_path"),
        matched_identity_anchors=tuple(
            routing_evidence.matched_anchors.get(f"{family}_IDENTITY", [])
        ),
        missing_required_anchors=tuple(
            routing_evidence.missing_required_anchors.get(family, [])
        ),
        field_topology_score=routing_evidence.field_topology_score.get(family, 0.0),
        contradiction_codes=contradictions,
        per_anchor_evidence=anchor_records,
    )