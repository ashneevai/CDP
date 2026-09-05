import statistics

from packages.document_taxonomy.taxonomy import DocumentClass

from .contracts import (
    FormIdentityVerification,
    StandardFormStatus,
    StandardFormVerification,
)
from .evidence import StandardFormEvidence

MIN_HIGH_VALUE_ANCHORS = 3
MIN_INDEPENDENT_REGIONS = 3
MIN_WEIGHTED_ANCHOR_COVERAGE = 0.42
MIN_GEOMETRY_SCORE = 0.45
MIN_STRUCTURE_SCORE = 0.35
MIN_EXPLICIT_STANDARD_SCORE = 0.55
MIN_TOPOLOGY_STANDARD_SCORE = 0.60
MIN_FAMILY_MARGIN = 0.15
MIN_TOPOLOGY_SCORE = 1.0


class CMS1500Verifier:
    policy_version = "cms1500-verifier-v2"

    def verify(self, evidence: StandardFormEvidence) -> StandardFormVerification:
        if evidence.candidate_family != DocumentClass.CMS1500:
            raise ValueError("CMS1500Verifier requires a CMS1500 nomination")
        return _result(evidence, self.policy_version)


def _result(evidence: StandardFormEvidence, policy: str) -> StandardFormVerification:
    unique_anchors = {item.anchor: item for item in evidence.per_anchor_evidence}
    high_value_count = sum(
        item.anchor_class.startswith("HIGH") for item in unique_anchors.values()
    )
    independent_regions = {item.region_id for item in unique_anchors.values()}
    class_weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 0.5}
    denominator = 14.0 if evidence.candidate_family == DocumentClass.CMS1500 else 19.5
    weighted_coverage = sum(
        class_weights[item.anchor_class.split("_", 1)[0]] * item.phrase_score
        for item in unique_anchors.values()
    ) / denominator
    geometry_score = (
        statistics.fmean(item.geometry_score for item in unique_anchors.values())
        if unique_anchors
        else 0.0
    )
    explicit = evidence.authorization_path == "EXPLICIT_IDENTITY"
    topology = evidence.authorization_path == "COMPLETE_TOPOLOGY"
    path_valid = (
        explicit
        and bool(evidence.matched_identity_anchors)
        and evidence.standard_score >= MIN_EXPLICIT_STANDARD_SCORE
    ) or (
        topology
        and not evidence.matched_identity_anchors
        and evidence.field_topology_score >= MIN_TOPOLOGY_SCORE
        and evidence.standard_score >= MIN_TOPOLOGY_STANDARD_SCORE
    )
    classes = {
        "FORM_IDENTITY_VERIFIED": (
            evidence.canonical_identity_confirmed
            and evidence.identity_status == "CONFIRMED"
            and evidence.identity_family == evidence.candidate_family
        ),
        "AUTHORIZATION_PATH_VALID": path_valid,
        "HIGH_VALUE_ANCHORS": high_value_count >= MIN_HIGH_VALUE_ANCHORS,
        "INDEPENDENT_REGIONS": len(independent_regions) >= MIN_INDEPENDENT_REGIONS,
        "WEIGHTED_ANCHOR_COVERAGE": weighted_coverage >= MIN_WEIGHTED_ANCHOR_COVERAGE,
        "SPATIAL_RELATIONSHIPS": geometry_score >= MIN_GEOMETRY_SCORE,
        "STANDARD_STRUCTURE": evidence.structure_score >= MIN_STRUCTURE_SCORE,
        "SERVICE_REGION_EVIDENCE": (
            evidence.region_layout_scores.get("revenue_service", 0.0) >= MIN_GEOMETRY_SCORE
            if evidence.candidate_family == DocumentClass.UB04
            else evidence.service_grid_score >= MIN_STRUCTURE_SCORE
        ),
        "FAMILY_MARGIN": evidence.family_margin >= MIN_FAMILY_MARGIN,
        "NO_CONTRADICTIONS": not evidence.contradiction_codes,
    }
    support = tuple(name for name, passed in classes.items() if passed)
    verified = all(classes.values())
    status = (
        StandardFormStatus.VERIFIED
        if verified
        else StandardFormStatus.AMBIGUOUS
        if len(support) >= 6 and not evidence.contradiction_codes
        else StandardFormStatus.NOT_VERIFIED
    )
    prefix = "CMS" if evidence.candidate_family == DocumentClass.CMS1500 else "UB"
    opposing = "UB" if prefix == "CMS" else "CMS"
    if verified:
        reasons = (f"{prefix}_VERIFIED_INDEPENDENT_IDENTITY",)
    elif evidence.contradiction_codes and any(
        opposing in code for code in evidence.contradiction_codes
    ):
        reasons = (f"{prefix}_{opposing}_CONTRADICTION",)
    elif evidence.contradiction_codes:
        reasons = (f"{prefix}_IDENTITY_OR_LAYOUT_CONTRADICTION",)
    elif not classes["FORM_IDENTITY_VERIFIED"]:
        reasons = (f"{prefix}_FORM_IDENTITY_NOT_VERIFIED",)
    else:
        reasons = (f"{prefix}_INSUFFICIENT_INDEPENDENT_EVIDENCE",)
    return StandardFormVerification(
        candidate_family=evidence.candidate_family,
        status=status,
        verification_score=len(support) / len(classes),
        supporting_evidence_classes=support,
        contradicting_evidence_classes=tuple(evidence.contradiction_codes),
        reason_codes=reasons,
        template_version=evidence.template_version,
        verification_policy_version=policy,
        eligible_for_fixed_extractor=verified,
        form_identity=FormIdentityVerification(
            status=(
                StandardFormStatus.VERIFIED
                if (
                    classes["FORM_IDENTITY_VERIFIED"]
                    and not evidence.contradiction_codes
                )
                else StandardFormStatus.NOT_VERIFIED
            ),
            family=evidence.identity_family,
            contradiction_codes=tuple(evidence.contradiction_codes),
            policy_version=evidence.identity_policy_version,
            authorization_path=evidence.authorization_path,
        ),
    )