from enum import StrEnum

from pydantic import Field, model_validator

from packages.document_taxonomy.taxonomy import DocumentClass
from packages.domain.common import DomainModel


class StandardFormStatus(StrEnum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    AMBIGUOUS = "AMBIGUOUS"


class FormIdentityVerification(DomainModel):
    status: StandardFormStatus = StandardFormStatus.NOT_VERIFIED
    family: DocumentClass | None = None
    contradiction_codes: tuple[str, ...] = ()
    policy_version: str = "strict-form-identity-v2"
    authorization_path: str | None = None

    @model_validator(mode="after")
    def verified_identity_is_family_specific(self):
        if self.status == StandardFormStatus.VERIFIED and self.family not in {
            DocumentClass.CMS1500,
            DocumentClass.UB04,
        }:
            raise ValueError("verified form identity requires a canonical family")
        if self.status == StandardFormStatus.VERIFIED and self.contradiction_codes:
            raise ValueError("contradicted identity cannot be verified")
        return self


class StandardFormVerification(DomainModel):
    candidate_family: DocumentClass
    status: StandardFormStatus
    verification_score: float = Field(ge=0, le=1)
    supporting_evidence_classes: tuple[str, ...] = ()
    contradicting_evidence_classes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    template_version: str | None = None
    verification_policy_version: str = "standard-form-verification-v2"
    eligible_for_fixed_extractor: bool = False
    form_identity: FormIdentityVerification = Field(default_factory=FormIdentityVerification)

    @model_validator(mode="after")
    def fixed_extractor_is_fail_closed(self):
        if self.candidate_family not in {DocumentClass.CMS1500, DocumentClass.UB04}:
            raise ValueError("standard verification supports only CMS1500 and UB04")
        if self.eligible_for_fixed_extractor:
            if self.status != StandardFormStatus.VERIFIED:
                raise ValueError("only VERIFIED may be eligible for a fixed extractor")
            if self.form_identity.status != StandardFormStatus.VERIFIED:
                raise ValueError("fixed extraction requires verified form identity")
            if self.form_identity.family != self.candidate_family:
                raise ValueError("fixed extraction requires family-consistent identity")
            if self.form_identity.contradiction_codes:
                raise ValueError("fixed extraction forbids identity contradictions")
        return self