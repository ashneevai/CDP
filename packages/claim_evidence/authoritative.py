"""Contracts for provenance-bearing authoritative claim evidence.

Frozen replay implementations must return NOT_AVAILABLE; live integrations are
outside evaluation and may not manufacture reference values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AuthoritativeMatchStatus(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class AuthoritativeEvidenceResult:
    status: AuthoritativeMatchStatus
    authority: str
    provenance_reference: str | None = None
    reason: str | None = None


class AuthoritativeEvidenceProvider(Protocol):
    def validate_member(self, member_id: str) -> AuthoritativeEvidenceResult: ...

    def validate_provider(self, npi: str, provider_name: str) -> AuthoritativeEvidenceResult: ...

    def validate_code(self, code_system: str, code: str) -> AuthoritativeEvidenceResult: ...


class UnavailableAuthoritativeEvidenceProvider:
    """Safe frozen-replay provider that never implies external agreement."""

    def _unavailable(self, authority: str) -> AuthoritativeEvidenceResult:
        return AuthoritativeEvidenceResult(
            AuthoritativeMatchStatus.NOT_AVAILABLE,
            authority,
            reason="NO_AUTHORIZED_REFERENCE_SNAPSHOT",
        )

    def validate_member(self, member_id: str) -> AuthoritativeEvidenceResult:
        return self._unavailable("MEMBER_ELIGIBILITY_MASTER")

    def validate_provider(self, npi: str, provider_name: str) -> AuthoritativeEvidenceResult:
        return self._unavailable("PROVIDER_MASTER")

    def validate_code(self, code_system: str, code: str) -> AuthoritativeEvidenceResult:
        return self._unavailable(f"LOCAL_{code_system.upper()}_REFERENCE")
