from packages.claim_evidence.field_policy import evaluate_field_policy


def _row(field, evidence):
    return {
        "final_value": "1234567893",
        "wrong_crop_suspected": False,
        "deterministic_validation": {"passed": True, "evidence": evidence},
        "localization_evidence": {
            "confirmed": True,
            "positive_bounded_roi": True,
            "geometry_valid": True,
        },
    }


def test_npi_safe_checksum_combination_can_replace_unavailable_e2():
    assert evaluate_field_policy(
        "provider_npi", "C3", _row("provider_npi", ["CHECKSUM_VALID"])
    ).accepted


def test_unsafe_crop_fails_closed():
    row = _row("provider_npi", ["CHECKSUM_VALID"])
    row["wrong_crop_suspected"] = True
    assert not evaluate_field_policy("provider_npi", "C3", row).accepted


def test_identifier_without_cross_field_evidence_remains_blocked():
    assert not evaluate_field_policy(
        "member_id", "C3", _row("member_id", ["FORMAT_VALID"])
    ).accepted
