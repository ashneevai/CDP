from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Protocol

from .contracts import (
    EvidenceAuthority,
    EvidenceClass,
    EvidenceObservation,
    EvidenceOutcome,
    EvidenceRequest,
)


class IndependentEvidenceProvider(Protocol):
    provider_id: str

    def supports(self, request: EvidenceRequest) -> bool: ...

    def collect(self, request: EvidenceRequest) -> EvidenceObservation: ...


def _digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def _luhn(number: str) -> bool:
    total = 0
    parity = len(number) % 2
    for index, char in enumerate(number):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class NPIChecksumProvider:
    """Deterministic NPI structure/checksum evidence; not a registry lookup."""

    provider_id = "deterministic.npi_checksum.v1"
    _fields = {"npi", "provider_npi", "billing_provider_npi", "rendering_provider_npi"}

    def supports(self, request: EvidenceRequest) -> bool:
        return request.field_name.lower() in self._fields

    def collect(self, request: EvidenceRequest) -> EvidenceObservation:
        value = _digits(request.candidate_value)
        valid = len(value) == 10 and value[0] in {"1", "2"} and _luhn("80840" + value)
        return EvidenceObservation(
            provider_id=self.provider_id,
            evidence_class=EvidenceClass.DETERMINISTIC,
            authority=EvidenceAuthority.STRONG,
            outcome=EvidenceOutcome.SUPPORT if valid else EvidenceOutcome.CONTRADICT,
            reason_code="NPI_CHECKSUM_VALID" if valid else "NPI_CHECKSUM_INVALID",
            lineage_key="deterministic:npi_checksum",
            value=value or request.candidate_value,
            source_version="CMS_NPI_LUHN_V1",
        )


class DateConsistencyProvider:
    """Checks parseability and simple claim-date chronology when context exists."""

    provider_id = "deterministic.date_consistency.v1"
    _date_fields = {
        "dob", "patient_dob", "service_date", "date_of_service", "admission_date",
        "discharge_date", "statement_from", "statement_to",
    }

    @staticmethod
    def _parse(value: object) -> date | None:
        if value is None:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m%d%Y", "%m%d%y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    def supports(self, request: EvidenceRequest) -> bool:
        return request.field_name.lower() in self._date_fields

    def collect(self, request: EvidenceRequest) -> EvidenceObservation:
        parsed = self._parse(request.candidate_value)
        if parsed is None:
            return EvidenceObservation(
                provider_id=self.provider_id,
                evidence_class=EvidenceClass.DETERMINISTIC,
                authority=EvidenceAuthority.STRONG,
                outcome=EvidenceOutcome.CONTRADICT,
                reason_code="DATE_INVALID",
                lineage_key="deterministic:date_parse",
                value=request.candidate_value,
            )

        today = date.today()
        if request.field_name.lower() in {"dob", "patient_dob"} and parsed > today:
            outcome, reason = EvidenceOutcome.CONTRADICT, "DOB_IN_FUTURE"
        else:
            admission = self._parse(request.claim_context.get("admission_date"))
            discharge = self._parse(request.claim_context.get("discharge_date"))
            if admission and discharge and discharge < admission:
                outcome, reason = EvidenceOutcome.CONTRADICT, "DISCHARGE_BEFORE_ADMISSION"
            else:
                outcome, reason = EvidenceOutcome.SUPPORT, "DATE_CONSISTENT"
        return EvidenceObservation(
            provider_id=self.provider_id,
            evidence_class=EvidenceClass.CROSS_FIELD if request.claim_context else EvidenceClass.DETERMINISTIC,
            authority=EvidenceAuthority.STRONG,
            outcome=outcome,
            reason_code=reason,
            lineage_key="cross_field:date_chronology" if request.claim_context else "deterministic:date_parse",
            value=request.candidate_value,
        )


class AmountReconciliationProvider:
    """Supports claim totals when service-line amounts reconcile within a small tolerance."""

    provider_id = "cross_field.amount_reconciliation.v1"
    _fields = {"claim_total", "total_charge", "total_charges", "total_amount"}

    def supports(self, request: EvidenceRequest) -> bool:
        return request.field_name.lower() in self._fields

    def collect(self, request: EvidenceRequest) -> EvidenceObservation:
        try:
            candidate = Decimal(str(request.candidate_value).replace("$", "").replace(",", ""))
            lines = request.claim_context.get("service_line_amounts") or []
            line_total = sum(Decimal(str(value).replace("$", "").replace(",", "")) for value in lines)
        except (InvalidOperation, ValueError, TypeError):
            return EvidenceObservation(
                provider_id=self.provider_id,
                evidence_class=EvidenceClass.CROSS_FIELD,
                authority=EvidenceAuthority.STRONG,
                outcome=EvidenceOutcome.INCONCLUSIVE,
                reason_code="AMOUNT_RECONCILIATION_UNAVAILABLE",
                lineage_key="cross_field:service_line_totals",
                value=request.candidate_value,
            )
        if not lines:
            outcome, reason = EvidenceOutcome.UNAVAILABLE, "SERVICE_LINE_AMOUNTS_MISSING"
        elif abs(candidate - line_total) <= Decimal("0.01"):
            outcome, reason = EvidenceOutcome.SUPPORT, "CLAIM_TOTAL_RECONCILED"
        else:
            outcome, reason = EvidenceOutcome.CONTRADICT, "CLAIM_TOTAL_MISMATCH"
        return EvidenceObservation(
            provider_id=self.provider_id,
            evidence_class=EvidenceClass.CROSS_FIELD,
            authority=EvidenceAuthority.STRONG,
            outcome=outcome,
            reason_code=reason,
            lineage_key="cross_field:service_line_totals",
            value=request.candidate_value,
            metadata={"computed_line_total": str(line_total) if lines else None},
        )


DEFAULT_PROVIDERS: tuple[IndependentEvidenceProvider, ...] = (
    NPIChecksumProvider(),
    DateConsistencyProvider(),
    AmountReconciliationProvider(),
)
