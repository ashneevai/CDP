"""Immutable local authoritative snapshots and exact-match providers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class MatchStatus(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    CONFLICT = "CONFLICT"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class AuthoritativeRecord:
    source_system: str
    source_record_id: str
    snapshot_id: str
    dataset_version: str
    record_hash: str
    effective_from: date
    effective_to: date | None
    values: dict[str, Any]


@dataclass(frozen=True)
class AuthoritativeSnapshot:
    snapshot_id: str
    source_system: str
    dataset_version: str
    effective_date: date
    created_at: datetime
    record_count: int
    schema_version: str
    sha256: str
    records: tuple[AuthoritativeRecord, ...]


@dataclass(frozen=True)
class AuthoritativeMatchResult:
    status: MatchStatus
    authoritative_source: str
    snapshot_id: str | None = None
    snapshot_version: str | None = None
    matched_record_id: str | None = None
    matched_fields: tuple[str, ...] = ()
    conflicting_fields: tuple[str, ...] = ()
    matching_rule: str | None = None
    record_hash: str | None = None
    provenance_reference: str | None = None

    @property
    def can_create_e7(self) -> bool:
        return bool(
            self.status == MatchStatus.MATCH
            and self.snapshot_id
            and self.snapshot_version
            and self.matched_record_id
            and self.record_hash
            and self.provenance_reference
        )


def _date(value: object, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid {field}") from error


def load_snapshot(path: str | Path) -> AuthoritativeSnapshot:
    source_path = Path(path)
    raw = source_path.read_bytes()
    payload = json.loads(raw)
    required = {
        "snapshot_id",
        "source_system",
        "dataset_version",
        "effective_date",
        "created_at",
        "schema_version",
        "records",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"snapshot missing fields: {sorted(missing)}")
    if payload["schema_version"] != "1.0":
        raise ValueError("unsupported authoritative snapshot schema")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("snapshot records must be a list")
    ids = [str(record.get("source_record_id") or "") for record in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate or blank authoritative source_record_id")
    snapshot_id = str(payload["snapshot_id"])
    source_system = str(payload["source_system"])
    dataset_version = str(payload["dataset_version"])
    parsed = []
    for record in records:
        effective_from = _date(record.get("effective_from"), "effective_from")
        effective_to = (
            _date(record["effective_to"], "effective_to") if record.get("effective_to") else None
        )
        if effective_to and effective_to < effective_from:
            raise ValueError("record effective period is inverted")
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        parsed.append(
            AuthoritativeRecord(
                source_system=source_system,
                source_record_id=str(record["source_record_id"]),
                snapshot_id=snapshot_id,
                dataset_version=dataset_version,
                record_hash=hashlib.sha256(canonical).hexdigest(),
                effective_from=effective_from,
                effective_to=effective_to,
                values={
                    key: value
                    for key, value in record.items()
                    if key not in {"source_record_id", "effective_from", "effective_to"}
                },
            )
        )
    try:
        created_at = datetime.fromisoformat(str(payload["created_at"]))
    except ValueError as error:
        raise ValueError("invalid created_at") from error
    if created_at.tzinfo is None:
        raise ValueError("created_at must include timezone")
    declared_count = payload.get("record_count", len(parsed))
    if declared_count != len(parsed):
        raise ValueError("record_count does not match records")
    return AuthoritativeSnapshot(
        snapshot_id=snapshot_id,
        source_system=source_system,
        dataset_version=dataset_version,
        effective_date=_date(payload["effective_date"], "effective_date"),
        created_at=created_at,
        record_count=len(parsed),
        schema_version="1.0",
        sha256=hashlib.sha256(raw).hexdigest(),
        records=tuple(parsed),
    )


def _identity(value: object) -> str:
    return str(value or "").strip().upper()


def _name(value: object) -> str:
    return re.sub(r"[^A-Z]", "", _identity(value))


def _result(
    status: MatchStatus,
    snapshot: AuthoritativeSnapshot | None,
    record: AuthoritativeRecord | None = None,
    *,
    matched: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
    rule: str | None = None,
) -> AuthoritativeMatchResult:
    source = snapshot.source_system if snapshot else "UNAVAILABLE_AUTHORITATIVE_SOURCE"
    return AuthoritativeMatchResult(
        status=status,
        authoritative_source=source,
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        snapshot_version=snapshot.dataset_version if snapshot else None,
        matched_record_id=record.source_record_id if record else None,
        matched_fields=matched,
        conflicting_fields=conflicts,
        matching_rule=rule,
        record_hash=record.record_hash if record else None,
        provenance_reference=(
            f"{source}:{snapshot.snapshot_id}:{record.source_record_id}:{record.record_hash}"
            if snapshot and record
            else None
        ),
    )


class MemberEligibilityEvidenceProvider:
    def __init__(self, snapshot: AuthoritativeSnapshot | None) -> None:
        self.snapshot = snapshot

    def validate(
        self,
        *,
        member_id: str,
        patient_name: str | None = None,
        subscriber_name: str | None = None,
        relationship: str | None = None,
        dob: str | None = None,
    ) -> AuthoritativeMatchResult:
        if self.snapshot is None:
            return _result(MatchStatus.NOT_AVAILABLE, None)
        matches = [
            record
            for record in self.snapshot.records
            if _identity(record.values.get("member_id")) == _identity(member_id)
        ]
        if not matches:
            return _result(MatchStatus.NO_MATCH, self.snapshot, rule="EXACT_MEMBER_ID")
        if len(matches) != 1:
            return _result(MatchStatus.CONFLICT, self.snapshot, rule="NON_UNIQUE_MEMBER_ID")
        record = matches[0]
        comparisons = {
            "patient_name": (_name(patient_name), _name(record.values.get("patient_name"))),
            "subscriber_name": (
                _name(subscriber_name),
                _name(record.values.get("subscriber_name")),
            ),
            "relationship": (_identity(relationship), _identity(record.values.get("relationship"))),
            "dob": (_identity(dob), _identity(record.values.get("dob"))),
        }
        checked = tuple(
            key for key, (actual, expected) in comparisons.items() if actual and expected
        )
        conflicts = tuple(key for key in checked if comparisons[key][0] != comparisons[key][1])
        return _result(
            MatchStatus.CONFLICT if conflicts else MatchStatus.MATCH,
            self.snapshot,
            record,
            matched=tuple(key for key in checked if key not in conflicts) + ("member_id",),
            conflicts=conflicts,
            rule="EXACT_MEMBER_ID_AND_SEMANTIC_FIELDS",
        )


class ProviderMasterEvidenceProvider:
    def __init__(self, snapshot: AuthoritativeSnapshot | None) -> None:
        self.snapshot = snapshot

    def validate(
        self, *, npi: str, provider_name: str, provider_role: str | None = None
    ) -> AuthoritativeMatchResult:
        if self.snapshot is None:
            return _result(MatchStatus.NOT_AVAILABLE, None)
        matches = [
            record
            for record in self.snapshot.records
            if _identity(record.values.get("npi")) == _identity(npi)
        ]
        if not matches:
            return _result(MatchStatus.NO_MATCH, self.snapshot, rule="EXACT_NPI")
        if len(matches) != 1:
            return _result(MatchStatus.CONFLICT, self.snapshot, rule="NON_UNIQUE_NPI")
        record = matches[0]
        conflicts = []
        if _name(provider_name) != _name(record.values.get("provider_name")):
            conflicts.append("provider_name")
        if (
            provider_role
            and record.values.get("provider_role")
            and _identity(provider_role) != _identity(record.values["provider_role"])
        ):
            conflicts.append("provider_role")
        return _result(
            MatchStatus.CONFLICT if conflicts else MatchStatus.MATCH,
            self.snapshot,
            record,
            matched=("npi",) if conflicts else ("npi", "provider_name"),
            conflicts=tuple(conflicts),
            rule="EXACT_NPI_AND_EXACT_NORMALIZED_NAME",
        )


class LocalCodeReferenceProvider:
    def __init__(self, snapshot: AuthoritativeSnapshot | None) -> None:
        self.snapshot = snapshot

    def validate(self, *, code_system: str, code: str) -> AuthoritativeMatchResult:
        if self.snapshot is None:
            return _result(MatchStatus.NOT_AVAILABLE, None)
        matches = [
            record
            for record in self.snapshot.records
            if _identity(record.values.get("code_system")) == _identity(code_system)
            and _identity(record.values.get("code")) == _identity(code)
        ]
        if not matches:
            return _result(MatchStatus.NO_MATCH, self.snapshot, rule="EXACT_CODE_AND_SYSTEM")
        if len(matches) != 1:
            return _result(MatchStatus.CONFLICT, self.snapshot, rule="NON_UNIQUE_CODE")
        return _result(
            MatchStatus.MATCH,
            self.snapshot,
            matches[0],
            matched=("code_system", "code"),
            rule="EXACT_CODE_AND_SYSTEM",
        )
