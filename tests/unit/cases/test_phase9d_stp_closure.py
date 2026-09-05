import json

from evaluation.phase9d_stp_closure import run
from packages.claim_evidence.authoritative import (
    AuthoritativeMatchStatus,
    UnavailableAuthoritativeEvidenceProvider,
)

EXPECTED = {
    "claim_stp_baseline.json",
    "claim_closure_matrix.json",
    "blocker_classification.json",
    "acceptance_authority_analysis.json",
    "authoritative_data_opportunity.json",
    "stp_ceiling_analysis.json",
    "claim_remediation_plan.json",
    "comparative_report.json",
}


def test_phase9d_classifies_every_post_phase9c_blocker(tmp_path):
    report = run(tmp_path)
    metrics = report["metrics"]
    assert report["verdict"] == "PASS"
    assert all(report["acceptance_gates"].values())
    assert metrics["remaining_blockers"] == 96
    classified = sum(
        metrics[name]
        for name in (
            "extraction_defect_blockers",
            "validation_gap_blockers",
            "acceptance_authority_gap_blockers",
            "authoritative_data_required_blockers",
            "source_evidence_required_blockers",
            "conflicting_evidence_blockers",
            "unreadable_source_blockers",
            "mandatory_hitl_blockers",
        )
    )
    assert classified == 96
    assert metrics["incorrect_but_accepted_fields"] == 0


def test_phase9d_keeps_achieved_and_scenario_stp_separate(tmp_path):
    run(tmp_path)
    ceilings = json.loads((tmp_path / "stp_ceiling_analysis.json").read_text("utf-8"))
    assert ceilings["ACHIEVED_STP"]["rate"] == 0.0
    assert ceilings["CURRENT_EVIDENCE_STP_CEILING"]["rate"] == 0.0
    assert ceilings["POTENTIAL_STP_IF_AUTHORITATIVE_DATA_AVAILABLE"]["rate"] == 0.55
    assert ceilings["FULL_SOURCE_STP_CEILING"]["rate"] == 1.0
    assert ceilings["POTENTIAL_STP_IF_AUTHORITATIVE_DATA_AVAILABLE"]["label"].startswith(
        "OPPORTUNITY_CEILING"
    )


def test_phase9d_matrix_has_one_owner_and_all_required_columns(tmp_path):
    run(tmp_path)
    matrix = json.loads((tmp_path / "claim_closure_matrix.json").read_text("utf-8"))
    assert len(matrix["rows"]) == 96
    required = {
        "claim_id",
        "field_name",
        "candidate_correct",
        "accepted",
        "primary_category",
        "remediation_owner",
        "available_evidence_classes",
        "acceptance_authority_status",
        "claim_unlock_distance",
    }
    assert all(required <= row.keys() for row in matrix["rows"])
    assert all(row["primary_category"] and row["remediation_owner"] for row in matrix["rows"])


def test_frozen_authoritative_provider_fails_closed():
    provider = UnavailableAuthoritativeEvidenceProvider()
    assert provider.validate_member("M1").status == AuthoritativeMatchStatus.NOT_AVAILABLE
    assert (
        provider.validate_provider("123", "NAME").status == AuthoritativeMatchStatus.NOT_AVAILABLE
    )
    assert provider.validate_code("ICD10", "A01").provenance_reference is None


def test_phase9d_writes_required_artifacts(tmp_path):
    run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED
