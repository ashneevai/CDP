"""Strict upstream capture manifest contracts for evidence-bearing bundles."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from packages.domain.common import DomainModel


class AvailabilityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CapturedAsset(DomainModel):
    status: AvailabilityStatus
    asset_uri: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mime_type: str | None = None

    @model_validator(mode="after")
    def require_identity_only_for_available_assets(self):
        identity = (self.asset_uri, self.sha256, self.mime_type)
        if self.status == AvailabilityStatus.AVAILABLE and not all(identity):
            raise ValueError("AVAILABLE_ASSET_REQUIRES_URI_HASH_AND_MIME")
        if self.status == AvailabilityStatus.UNAVAILABLE and any(identity):
            raise ValueError("UNAVAILABLE_ASSET_MUST_NOT_HAVE_FABRICATED_IDENTITY")
        return self


class NormalizedPageLineage(DomainModel):
    page_id: str
    page_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_uri: str
    observation_version: str
    source_document_id: str | None = None
    source_page_index: int | None = Field(default=None, ge=0)


class SourceBundleCaptureRecord(DomainModel):
    claim_id: str
    bundle_id: str
    raw_bundle: CapturedAsset
    raw_documents: tuple[CapturedAsset, ...] = ()
    attachment_inventory_status: AvailabilityStatus
    normalized_pages: tuple[NormalizedPageLineage, ...]
    acquisition_reason_codes: tuple[str, ...] = ()
    schema_version: str = "phase8.27-source-bundle-capture-v1"

    @model_validator(mode="after")
    def require_source_links_when_raw_bundle_is_available(self):
        if self.raw_bundle.status == AvailabilityStatus.AVAILABLE and any(
            page.source_document_id is None or page.source_page_index is None
            for page in self.normalized_pages
        ):
            raise ValueError("AVAILABLE_BUNDLE_REQUIRES_PAGE_TO_DOCUMENT_LINEAGE")
        return self
