"""Healthcare-claim specialist field policies for CMS-1500 style documents.

This module is deliberately deterministic.  It normalizes and validates
candidate values but never decides whether a field may be machine accepted.
Final authority remains EvidenceDecisionService.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Callable


@dataclass(frozen=True, slots=True)
class FieldPolicy:
    field_name: str
    aliases: tuple[str, ...]
    critical: bool
    validator: Callable[[str], bool]
    normalizer: Callable[[str], str]
    preferred_sources: tuple[str, ...]


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _alnum(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", _compact(value).upper())


def normalize_member_id(value: str) -> str:
    return _alnum(value)


def validate_member_id(value: str) -> bool:
    token = normalize_member_id(value)
    return 5 <= len(token) <= 24 and any(ch.isdigit() for ch in token)


def normalize_npi(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def validate_npi(value: str) -> bool:
    digits = normalize_npi(value)
    if len(digits) != 10:
        return False
    # CMS NPI validation uses the Luhn algorithm with the 80840 prefix.
    payload = "80840" + digits[:-1]
    total = 0
    parity = len(payload) % 2
    for index, char in enumerate(payload):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    check = (10 - total % 10) % 10
    return check == int(digits[-1])


def normalize_tax_id(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def validate_tax_id(value: str) -> bool:
    return len(normalize_tax_id(value)) == 9


def normalize_code(value: str) -> str:
    return re.sub(r"[^A-Z0-9.]", "", str(value or "").upper())


def validate_diagnosis(value: str) -> bool:
    token = normalize_code(value)
    return bool(re.fullmatch(r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?", token))


def validate_procedure(value: str) -> bool:
    token = normalize_code(value).replace(".", "")
    return bool(
        re.fullmatch(r"\d{5}", token)
        or re.fullmatch(r"[A-V]\d{4}", token)
        or re.fullmatch(r"\d{4}[A-Z]", token)
    )


def normalize_date(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    for fmt, length in (("%m%d%Y", 8), ("%m%d%y", 6)):
        if len(digits) != length:
            continue
        try:
            return datetime.strptime(digits, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return _compact(value)


def validate_date(value: str) -> bool:
    normalized = normalize_date(value)
    try:
        parsed = datetime.strptime(normalized, "%Y-%m-%d")
    except ValueError:
        return False
    return 1900 <= parsed.year <= 2100


def normalize_amount(value: str) -> str:
    text = str(value or "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d{1,2})?", text)
    if not match:
        return _compact(value)
    try:
        return f"{float(match.group()):.2f}"
    except ValueError:
        return _compact(value)


def validate_amount(value: str) -> bool:
    normalized = normalize_amount(value)
    try:
        amount = float(normalized)
    except ValueError:
        return False
    return 0 <= amount <= 100_000_000


def normalize_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", _compact(value).upper())


def validate_name(value: str) -> bool:
    token = normalize_name(value)
    return len(token) >= 3 and any(ch.isalpha() for ch in token)


def normalize_account(value: str) -> str:
    return _alnum(value)


def validate_account(value: str) -> bool:
    token = normalize_account(value)
    return 2 <= len(token) <= 40


CMS_FIELD_POLICIES: dict[str, FieldPolicy] = {
    "member_id": FieldPolicy(
        "member_id", ("insured_id_number", "insured_unique_id"), True,
        validate_member_id, normalize_member_id,
        ("spatial", "rapidocr", "paddleocr", "regional_ocr", "vlm"),
    ),
    "patient_name": FieldPolicy(
        "patient_name", ("patient_name",), True, validate_name, normalize_name,
        ("spatial", "rapidocr", "regional_ocr", "paddleocr", "vlm"),
    ),
    "dob": FieldPolicy(
        "dob", ("patient_dob",), True, validate_date, normalize_date,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr", "vlm"),
    ),
    "provider_npi": FieldPolicy(
        "provider_npi", ("billing_provider_npi", "rendering_provider_npi"), True,
        validate_npi, normalize_npi,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr", "reference"),
    ),
    "federal_tax_no": FieldPolicy(
        "federal_tax_no", ("federal_tax_id",), True, validate_tax_id, normalize_tax_id,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr"),
    ),
    "diagnosis": FieldPolicy(
        "diagnosis", ("diagnosis_codes",), True, validate_diagnosis, normalize_code,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr", "vlm"),
    ),
    "procedure": FieldPolicy(
        "procedure", ("procedure_code", "hcpcs", "cpt"), True,
        validate_procedure, normalize_code,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr", "vlm"),
    ),
    "service_date": FieldPolicy(
        "service_date", ("date_of_service",), True, validate_date, normalize_date,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr"),
    ),
    "total_charge": FieldPolicy(
        "total_charge", ("total_charge", "claim_total"), True,
        validate_amount, normalize_amount,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr"),
    ),
    "account_no": FieldPolicy(
        "account_no", ("patient_account_no", "patient_control_number"), False,
        validate_account, normalize_account,
        ("spatial", "regional_ocr", "rapidocr", "paddleocr", "vlm"),
    ),
}


def policy_for(field_name: str) -> FieldPolicy | None:
    if field_name in CMS_FIELD_POLICIES:
        return CMS_FIELD_POLICIES[field_name]
    for policy in CMS_FIELD_POLICIES.values():
        if field_name in policy.aliases:
            return policy
    return None
