import json

from evaluation.phase8_22_blocker_optimization import run
from packages.claim_evidence.independence import observations_are_independent
from packages.domain.common import BoundingBox
from packages.field_localization.deterministic_repair import repair_from_expected_zone
from packages.ocr.token_reconstruction import SpatialToken, reconstruct_field_tokens


def _box(x0, y0, x1, y1):
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, image_width=200, image_height=100)


def test_name_reconstruction_removes_labels_and_preserves_geometry_order():
    tokens = (
        SpatialToken("NAME", .99, _box(0, 0, 30, 10)),
        SpatialToken("DOE", .98, _box(70, 30, 100, 45)),
        SpatialToken("JANE", .97, _box(20, 30, 60, 45)),
        SpatialToken("123", .99, _box(110, 30, 140, 45)),
    )
    result = reconstruct_field_tokens("patient_name", tokens, region=_box(10, 20, 150, 60))
    assert result.value == "JANE DOE"
    assert all(token.text not in {"NAME", "123"} for token in result.selected_tokens)


def test_localization_repair_fails_closed_for_competing_rows():
    tokens = (
        SpatialToken("JANE", .99, _box(20, 20, 60, 30)),
        SpatialToken("MARY", .99, _box(20, 60, 60, 70)),
    )
    repair = repair_from_expected_zone(tokens, _box(0, 0, 100, 90))
    assert repair.outcome == "LOCALIZATION_UNCERTAIN"
    assert repair.bounding_box is None


def test_same_crop_never_counts_as_independent_evidence():
    left = {"invocation_id": "a", "crop_sha256": "same", "localization_region_id": "one"}
    right = {"invocation_id": "b", "crop_sha256": "same", "localization_region_id": "two"}
    assert observations_are_independent(left, right) is False


def test_phase8_22_frozen_replay_is_safe_complete_and_writes_six_artifacts(tmp_path):
    report = run(tmp_path)
    assert report["verdict"] == "PASS"
    assert report["metrics"]["critical_false_accepts"] == 0
    assert report["metrics"]["blockers_removed"] == 7
    assert report["metrics"]["raw_accuracy"] == {"before": .855, "after": .89}
    assert report["production_candidate_overwrites"] == 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "blocker_cohort_analysis.json", "localization_metrics.json",
        "name_field_metrics.json", "independent_evidence_metrics.json",
        "claim_unlock_analysis.json", "comparative_report.json",
    }
    cohort = json.loads((tmp_path / "blocker_cohort_analysis.json").read_text())
    assert len(cohort["fields"]) == 200
    assert cohort["no_single_targeted_cohort_can_unlock_claim"] is True
