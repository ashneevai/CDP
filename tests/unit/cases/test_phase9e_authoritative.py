import hashlib
import json

import pytest

from evaluation.phase9e_authoritative_replay import run
from packages.claim_evidence.authoritative_snapshot import (
    LocalCodeReferenceProvider,
    MatchStatus,
    MemberEligibilityEvidenceProvider,
    ProviderMasterEvidenceProvider,
    load_snapshot,
)


def _snapshot(tmp_path, name, records, source="TEST_AUTHORITY"):
    path = tmp_path / name
    payload = {
        "snapshot_id": f"snapshot-{name}",
        "source_system": source,
        "dataset_version": "2026-09-01",
        "effective_date": "2026-09-01",
        "created_at": "2026-09-01T00:00:00Z",
        "record_count": len(records),
        "schema_version": "1.0",
        "records": records,
    }
    path.write_text(json.dumps(payload, sort_keys=True), "utf-8")
    return path


def _record(record_id="r1", **values):
    return {
        "source_record_id": record_id,
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        **values,
    }


def test_snapshot_hashes_actual_bytes_and_rejects_duplicate_ids(tmp_path):
    path = _snapshot(tmp_path, "member_snapshot.json", [_record(member_id="001-ABC")])
    snapshot = load_snapshot(path)
    assert snapshot.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot.records[0].record_hash

    duplicate = _snapshot(tmp_path, "duplicate.json", [_record("same"), _record("same")])
    with pytest.raises(ValueError, match="duplicate"):
        load_snapshot(duplicate)


def test_member_exact_match_preserves_leading_zeroes_and_has_provenance(tmp_path):
    snapshot = load_snapshot(
        _snapshot(
            tmp_path,
            "member.json",
            [
                _record(
                    member_id="001-ABC",
                    patient_name="Ana Lee",
                    subscriber_name="Ana Lee",
                    relationship="SELF",
                )
            ],
        )
    )
    provider = MemberEligibilityEvidenceProvider(snapshot)
    match = provider.validate(member_id="001-ABC", patient_name="ANA LEE")
    assert match.status == MatchStatus.MATCH
    assert match.can_create_e7 is True
    assert match.provenance_reference
    assert provider.validate(member_id="1-ABC").status == MatchStatus.NO_MATCH


def test_member_semantic_conflict_fails_closed_without_fuzzy_identity(tmp_path):
    snapshot = load_snapshot(
        _snapshot(
            tmp_path,
            "member.json",
            [
                _record(
                    member_id="M-001",
                    patient_name="Ana Lee",
                    subscriber_name="Sam Lee",
                    relationship="CHILD",
                )
            ],
        )
    )
    provider = MemberEligibilityEvidenceProvider(snapshot)
    conflict = provider.validate(member_id="M-001", subscriber_name="ANA LEE", relationship="SELF")
    assert conflict.status == MatchStatus.CONFLICT
    assert conflict.can_create_e7 is False
    assert set(conflict.conflicting_fields) == {"subscriber_name", "relationship"}


def test_provider_requires_exact_npi_and_exact_normalized_name(tmp_path):
    snapshot = load_snapshot(
        _snapshot(
            tmp_path,
            "provider.json",
            [
                _record(
                    npi="1234567893", provider_name="Summit Medical Group", provider_role="BILLING"
                )
            ],
            source="PROVIDER_MASTER",
        )
    )
    provider = ProviderMasterEvidenceProvider(snapshot)
    assert (
        provider.validate(npi="1234567893", provider_name="SUMMIT MEDICAL GROUP").status
        == MatchStatus.MATCH
    )
    assert (
        provider.validate(npi="1234567893", provider_name="SUMMIT MEDICAL GRP").status
        == MatchStatus.CONFLICT
    )
    assert (
        provider.validate(npi="1234567890", provider_name="SUMMIT MEDICAL GROUP").status
        == MatchStatus.NO_MATCH
    )


def test_code_reference_is_exact_and_unavailable_preserves_hitl(tmp_path):
    snapshot = load_snapshot(
        _snapshot(
            tmp_path,
            "codes.json",
            [_record(code_system="ICD10", code="A01.0")],
            source="ICD_REFERENCE",
        )
    )
    provider = LocalCodeReferenceProvider(snapshot)
    assert provider.validate(code_system="ICD10", code="A01.0").can_create_e7
    assert provider.validate(code_system="ICD10", code="A010").status == MatchStatus.NO_MATCH
    assert (
        LocalCodeReferenceProvider(None).validate(code_system="ICD10", code="A01.0").status
        == MatchStatus.NOT_AVAILABLE
    )


def test_phase9e_without_real_snapshots_is_fail_closed_needs_more_data(tmp_path):
    output, absent = tmp_path / "output", tmp_path / "absent-snapshots"
    report = run(output, absent)
    metrics = report["metrics"]
    assert report["verdict"] == "NEEDS_MORE_DATA"
    assert metrics["authoritative_snapshots_loaded"] == 0
    assert metrics["NOT_AVAILABLE"] == 60
    assert metrics["MATCH"] == metrics["E7_backed_fields_accepted"] == 0
    assert (
        metrics["blockers_removed"] == metrics["claims_advanced"] == metrics["claims_unlocked"] == 0
    )
    assert metrics["field_hitl_before"] == metrics["field_hitl_after"] == 0.48
    assert metrics["claim_stp_after"] == 0.0
    assert all(report["safety_gates"].values())


def test_phase9e_writes_all_required_artifacts(tmp_path):
    output = tmp_path / "output"
    run(output, tmp_path / "absent")
    assert {path.name for path in output.iterdir()} == {
        "authoritative_snapshot_inventory.json",
        "member_eligibility_metrics.json",
        "provider_master_metrics.json",
        "reference_validation_metrics.json",
        "e7_acceptance_analysis.json",
        "claim_unlock_distance.json",
        "regression_analysis.json",
        "comparative_report.json",
    }
