from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.reference_enrichment.contracts import ReferenceLookupRequest, ReferenceRecord


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    reference_domain: str
    version: str
    snapshot_timestamp: datetime
    records_file: str = "records.json"
    storage_format: str = "JSON"
    record_count: int | None = Field(default=None, ge=0)
    records_sha256: str
    authorized: bool = False
    independent_truth: bool = False
    non_circular_lineage: bool = False
    source_contract_id: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    supported_fields: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_artifact_sha256: str | None = None
    source_published_at: datetime | None = None
    effective_from: datetime | None = None
    effective_through: datetime | None = None
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def authorized_snapshots_require_governance(self) -> SnapshotManifest:
        if self.authorized and not all(
            (
                self.independent_truth,
                self.non_circular_lineage,
                self.source_contract_id,
                self.approved_by,
                self.approved_at,
            )
        ):
            raise ValueError("authorized snapshot lacks governance approval or lineage")
        if bool(self.source_url) != bool(self.source_artifact_sha256):
            raise ValueError("source URL and artifact checksum must be supplied together")

        def valid_sha256(value: str) -> bool:
            return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)

        if not valid_sha256(self.records_sha256):
            raise ValueError("records checksum must be a SHA-256 hex digest")
        if self.source_artifact_sha256 and not valid_sha256(self.source_artifact_sha256):
            raise ValueError("source artifact checksum must be a SHA-256 hex digest")
        temporal = (
            self.snapshot_timestamp,
            self.source_published_at,
            self.effective_from,
            self.effective_through,
            self.expires_at,
            self.approved_at,
        )
        if any(value is not None and value.tzinfo is None for value in temporal):
            raise ValueError("snapshot timestamps must include a timezone")
        if (
            self.effective_from
            and self.effective_through
            and self.effective_from > self.effective_through
        ):
            raise ValueError("effective_from must not follow effective_through")
        if self.expires_at and self.expires_at <= self.snapshot_timestamp:
            raise ValueError("expires_at must follow snapshot_timestamp")
        if self.storage_format not in {"JSON", "SQLITE"}:
            raise ValueError("storage_format must be JSON or SQLITE")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup_key(value: str, reference_domain: str) -> str:
    key = value.strip()
    domain = reference_domain.upper()
    if domain == "ICD10CM":
        return key.upper().replace(".", "")
    if domain == "PLACE_OF_SERVICE":
        return key.zfill(2)
    if domain == "HCPCS_LEVEL_II":
        return key.upper()
    return key


@dataclass(frozen=True)
class LocalSnapshotProvider:
    root: Path
    test_only: bool = True
    _verified_state: dict[str, object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def manifest(self) -> SnapshotManifest:
        return SnapshotManifest.model_validate_json(
            (self.root / "manifest.json").read_text("utf-8")
        )

    @property
    def name(self) -> str:
        return self.manifest.source_name

    @property
    def provider_type(self) -> str:
        return self.manifest.reference_domain.upper()

    @property
    def authorized(self) -> bool:
        return self.manifest.authorized

    def supports(self, field_name: str) -> bool:
        fields = self.manifest.supported_fields
        return not fields or field_name in fields

    def is_available(self, field_name: str, at: datetime | None = None) -> bool:
        try:
            manifest = self.manifest
            if manifest.supported_fields and field_name not in manifest.supported_fields:
                return False
            checked_at = (at or datetime.now(UTC)).astimezone(UTC)
            if manifest.effective_from and checked_at < manifest.effective_from.astimezone(UTC):
                return False
            if manifest.effective_through and checked_at > manifest.effective_through.astimezone(
                UTC
            ):
                return False
            if manifest.expires_at and checked_at >= manifest.expires_at.astimezone(UTC):
                return False
            records_path = self._records_path(manifest)
            self._verify_records(records_path, manifest.records_sha256)
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def _records_path(self, manifest: SnapshotManifest) -> Path:
        records_path = (self.root / manifest.records_file).resolve()
        if self.root.resolve() not in records_path.parents:
            raise ValueError("snapshot records path escapes snapshot root")
        return records_path

    def _verify_records(self, path: Path, expected_sha256: str) -> None:
        stat = path.stat()
        fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns, expected_sha256)
        if self._verified_state.get("fingerprint") == fingerprint:
            return
        if _sha256(path) != expected_sha256:
            raise ValueError("reference snapshot checksum mismatch")
        self._verified_state["fingerprint"] = fingerprint

    def _load_rows(self, path: Path, manifest: SnapshotManifest, identity_key: str) -> list[dict]:
        if manifest.storage_format == "JSON":
            rows = json.loads(path.read_text("utf-8"))
            return [row for row in rows if row.get("identity_key") == identity_key]
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            result = connection.execute(
                "SELECT payload_json FROM reference_records WHERE identity_key = ?",
                (identity_key,),
            ).fetchall()
        return [json.loads(row[0]) for row in result]

    def lookup(self, request: ReferenceLookupRequest) -> list[ReferenceRecord]:
        manifest = self.manifest
        if manifest.supported_fields and request.field_name not in manifest.supported_fields:
            return []
        requested_at = request.requested_at.astimezone(UTC)
        if manifest.effective_from and requested_at < manifest.effective_from.astimezone(UTC):
            raise RuntimeError("reference snapshot is not yet effective")
        if manifest.effective_through and requested_at > manifest.effective_through.astimezone(UTC):
            raise RuntimeError("reference snapshot is outside its effective period")
        if manifest.expires_at and requested_at >= manifest.expires_at.astimezone(UTC):
            raise RuntimeError("reference snapshot has expired")
        records_path = self._records_path(manifest)
        self._verify_records(records_path, manifest.records_sha256)
        rows = self._load_rows(
            records_path, manifest, _lookup_key(request.identity_key, manifest.reference_domain)
        )
        output: list[ReferenceRecord] = []
        for row in rows:
            payload = dict(row)
            payload.update(
                provider_name=manifest.source_name,
                provider_type=manifest.reference_domain.upper(),
                provider_authorized=manifest.authorized,
                dataset_version=manifest.version,
                snapshot_timestamp=manifest.snapshot_timestamp,
                snapshot_checksum=manifest.records_sha256,
                independent_truth=manifest.independent_truth,
                non_circular_lineage=manifest.non_circular_lineage,
            )
            payload.pop("identity_key", None)
            output.append(ReferenceRecord.model_validate(payload))
        return output
