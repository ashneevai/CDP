import json
from types import SimpleNamespace

from evaluation.azure_live_shadow import run
from packages.llm_adjudication.azure import AdjudicationDecision, AdjudicationResult, RoutingTier


def record(**changes):
    row = {
        "package_id": "pkg",
        "source_asset_id": "asset",
        "source_page_id": "page",
        "cdp_page_id": "cdp",
        "field_id": "field",
        "form_type": "UB04",
        "reviewed_page_class": "UB04",
        "field_name": "provider_name",
        "field_type": "NAME",
        "criticality": "CRITICAL",
        "ground_truth": "JOHN SMITH",
        "ground_truth_state": "VALUE",
        "ground_truth_authority": "HUMAN_ADJUDICATED",
        "critical_dual_reviewed": True,
        "annotation_disagreed": True,
        "adjudication_complete": True,
        "local_candidates": ["JOHN SM1TH", "JOHN SMITH"],
        "local_decision": "HUMAN_REVIEW_REQUIRED",
        "local_hitl": True,
        "local_hitl_reason": "OCR_DISAGREEMENT",
        "claim_blocking": True,
        "crop_safe": True,
        "localization_confidence": 0.96,
        "claim_distance": 1,
        "binding": {
            "exact": True,
            "rendered_page_sha256": "a" * 64,
            "source_representation_id": "rep",
            "pipeline_execution_id": "run",
            "page_observation_id": "obs",
        },
    }
    row.update(changes)
    return row


def write_input(path, records):
    path.mkdir()
    (path / "trusted_shadow_cohort.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )


class FakeService:
    def __init__(self):
        self.router = SimpleNamespace(config=SimpleNamespace(api_version="2024-10-21"))

    def observe(self, **kwargs):
        return AdjudicationResult(
            decision=AdjudicationDecision.SELECT_CANDIDATE,
            candidate_id="candidate_1",
            selected_value="JOHN SMITH",
            reason_code="BEST",
            tier=RoutingTier.TEXT,
            deployment="gpt-4o",
            model="version",
            input_tokens=100,
            output_tokens=10,
            latency_ms=50,
            cost_status="PRICING_NOT_CONFIGURED",
            data_categories_sent=("TARGET_FIELD_CANDIDATES",),
        )


def test_zero_trusted_labels_produces_all_artifacts_without_live_call(tmp_path):
    inputs = tmp_path / "in"
    write_input(inputs, [])
    out = tmp_path / "out"
    result = run(inputs, out, live=True, service=FakeService())
    assert result["status"] == "NEEDS_MORE_TRUSTED_LABELS" and result["azure_calls"] == 0
    assert len(list(out.iterdir())) == 14


def test_untrusted_and_unbound_records_are_rejected(tmp_path):
    inputs = tmp_path / "in"
    write_input(
        inputs,
        [record(ground_truth_authority="HUMAN_SINGLE_REVIEW"), record(field_id="f2", binding={})],
    )
    result = run(inputs, tmp_path / "out")
    assert (
        result["annotation_summary"]["trusted_field_labels"] == 1
        and result["annotation_summary"]["rejected_untrusted"] == 1
        and result["binding_rate"] == 0
    )


def test_live_shadow_scores_candidate_precision_without_changing_canonical(tmp_path):
    inputs = tmp_path / "in"
    write_input(inputs, [record()])
    out = tmp_path / "out"
    result = run(inputs, out, live=True, service=FakeService())
    assert result["benchmark_pages"] == 1 and result["fields_routed_to_azure"] == 1
    assert (
        result["candidate_selection_precision"] == 1
        and result["critical_false_accept_potential"] == 0
    )
    assert (
        result["potential_field_hitl_reduction"] == 1 and result["potential_blockers_removed"] == 1
    )
    assert result["cost"]["pricing_status"] == "PRICING_NOT_CONFIGURED"
    assert (
        json.loads((out / "comparative_report.json").read_text())["canonical_decisions_changed"]
        == 0
    )
    safe = json.loads((out / "trusted_shadow_cohort.json").read_text())["records"][0]
    assert safe["ground_truth"] == "REDACTED" and "JOHN SMITH" not in json.dumps(safe)


def test_critical_label_requires_dual_review_and_disagreement_adjudication(tmp_path):
    inputs = tmp_path / "in"
    write_input(
        inputs,
        [record(critical_dual_reviewed=False), record(field_id="f2", adjudication_complete=False)],
    )
    result = run(inputs, tmp_path / "out")
    assert result["annotation_summary"]["trusted_field_labels"] == 0


def test_tier2_only_marks_visual_ambiguity(tmp_path):
    inputs = tmp_path / "in"
    write_input(inputs, [record(visual_ambiguity=True)])
    result = run(inputs, tmp_path / "out")
    assert result["tier2_candidate_cohort_size"] == 1
