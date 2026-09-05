from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable
from pathlib import Path

import yaml

from packages.criticality import CriticalityLevel
from packages.evidence_decision.contracts import ReferenceEvidence
from packages.reference_enrichment.contracts import ReferenceDecision, ReferenceLookupRequest, ReferenceResolution
from packages.reference_enrichment.providers import ReferenceProvider, configured_providers
from packages.reference_enrichment.service import ReferenceMatchingService

REFERENCE_POLICY_VERSION = "reference-evidence-v1"

_ALIASES = {
    "member_id": ("member_id", "insured_id_number", "subscriber_id"),
    "dob": ("dob", "patient_dob", "insured_dob"),
    "name": ("patient_name", "insured_name", "subscriber_name", "patient_first"),
    "zip": ("patient_zip", "insured_zip", "subscriber_zip"),
    "npi": ("npi", "billing_provider_npi", "rendering_provider_npi", "provider_npi"),
    "provider_name": ("provider_name", "billing_provider_name", "rendering_provider_name"),
}
_CODE_FIELDS = {
    "principal_diagnosis",
    "diagnosis_code",
    "cpt",
    "cpt_hcpcs",
    "hcpcs",
    "place_of_service",
}


def canonical_claim_attributes(values: dict[str, Any]) -> dict[str, Any]:
    """Add stable matcher keys without discarding the source field names."""
    result = {key: value for key, value in values.items() if value not in (None, "")}
    for canonical, aliases in _ALIASES.items():
        value = next((values.get(alias) for alias in aliases if values.get(alias)), None)
        if value is not None:
            result[canonical] = value
    return result


def identity_key(field_name: str, candidate: str | None, attributes: dict[str, Any]) -> str | None:
    if field_name in _CODE_FIELDS:
        return candidate
    if "provider" in field_name or field_name == "npi":
        return attributes.get("npi")
    return attributes.get("member_id")


def reference_evidence_from_resolution(resolution: ReferenceResolution) -> ReferenceEvidence | None:
    return reference_evidence_from_decision(resolution.decision)


def reference_evidence_from_decision(decision: ReferenceDecision) -> ReferenceEvidence | None:
    unavailable = {
        "REFERENCE_NOT_FOUND", "REFERENCE_PROVIDER_ERROR", "PENDING", "PROVIDER_UNAUTHORIZED",
        "CIRCULAR_LINEAGE_REJECTED", "INSUFFICIENT_MATCH_ATTRIBUTES",
    }
    if decision.decision in unavailable:
        return None
    verified = decision.evaluation_eligible and decision.decision.endswith("VERIFIED")
    return ReferenceEvidence(
        value=decision.normalized_reference_value,
        verified=verified,
        contradiction=decision.decision in {"REFERENCE_CONTRADICTION", "MULTIPLE_REFERENCE_MATCHES"},
        source=decision.reference_provider,
        version=decision.reference_dataset_version,
        reference_key=decision.source_record_id,
        matched_attributes=decision.matching_attributes,
        match_scores=decision.match_scores,
        conflicts=decision.contradictions,
        snapshot_timestamp=decision.snapshot_timestamp,
        snapshot_checksum=decision.snapshot_checksum,
    )


@dataclass
class ReferenceEvidenceService:
    """The single runtime/evaluation bridge from reference providers to decisions."""

    providers: list[ReferenceProvider]
    policy_version: str = REFERENCE_POLICY_VERSION
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    @classmethod
    def from_config(cls, path: str | Path) -> "ReferenceEvidenceService":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for provider in payload.get("providers", []):
            if provider.get("path") and not Path(provider["path"]).is_absolute():
                provider["path"] = str((config_path.parent / provider["path"]).resolve())
        return cls(
            configured_providers(payload),
            policy_version=str(payload.get("policy_version", REFERENCE_POLICY_VERSION)),
        )

    def resolve(
        self, *, document_id: str, page_number: int, document_family: str,
        field_name: str, criticality: CriticalityLevel, raw_value: str | None,
        normalized_value: str | None, claim_values: dict[str, Any],
    ) -> ReferenceResolution | None:
        attributes = canonical_claim_attributes(claim_values)
        candidate = normalized_value or raw_value
        key = identity_key(field_name, candidate, attributes)
        if not key or not self.providers:
            return None
        digest = hashlib.sha256(
            f"{document_id}|{page_number}|{field_name}|{key}|{self.policy_version}".encode()
        ).hexdigest()
        request = ReferenceLookupRequest(
            request_id=digest, identity_key=str(key), document_id=document_id,
            page_number=page_number, document_family=document_family, field_name=field_name,
            criticality="CRITICAL" if criticality in {CriticalityLevel.C2, CriticalityLevel.C3} else "NON_CRITICAL",
            current_candidate=candidate, available_claim_attributes=attributes,
            requested_at=self.clock(), policy_version=self.policy_version,
        )
        return ReferenceMatchingService(self.providers).match(
            request, raw_value=raw_value, normalized_value=normalized_value,
        )

    def evidence(self, **kwargs: Any) -> tuple[ReferenceEvidence | None, dict[str, Any] | None]:
        resolution = self.resolve(**kwargs)
        if resolution is None:
            return None, None
        return reference_evidence_from_resolution(resolution), resolution.model_dump(mode="json")
