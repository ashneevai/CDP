import asyncio
import json

from evaluation.phase8_21a_source_b_replay import run
from packages.ocr.ppocr_v5_provider import PPOCRv5Provider


def _provider() -> PPOCRv5Provider:
    return PPOCRv5Provider(backend=lambda _image: [{
        "rec_texts": ["X"], "rec_scores": [.9],
        "rec_polys": [[[1, 1], [10, 1], [10, 10], [1, 10]]],
    }])


def test_frozen_source_b_replay_is_complete_unique_and_budgeted(tmp_path):
    report = asyncio.run(run(output=tmp_path, provider=_provider()))
    rows = [json.loads(line) for line in (tmp_path / "field_replay.jsonl").read_text().splitlines()]
    cohort = json.loads((tmp_path / "frozen_cohort.json").read_text())

    assert len(rows) == 200
    assert len({row["evaluation_field_key"]["key_sha256"] for row in rows}) == 200
    assert cohort["claims"] == 20
    assert cohort["eligible_fields"] == 38
    assert cohort["challenged_fields"] == 11
    assert report["challenger_metrics"]["ppocr_challenge_rate"] <= .30
    assert all(row["candidate_authority"] == "REVIEW_ONLY" for row in rows)
    assert not any(row["production_value_overwritten"] for row in rows)


def test_replay_writes_required_artifacts_and_never_uses_missing_data_for_poor_results(tmp_path):
    report = asyncio.run(run(output=tmp_path, provider=_provider()))
    required = {
        "frozen_cohort.json", "field_replay.jsonl", "quality_band_metrics.json",
        "challenger_metrics.json", "claim_unlock_analysis.json", "comparative_report.json",
    }
    assert required == {path.name for path in tmp_path.iterdir()}
    assert report["verdict"] == "REJECT"
    assert report["acceptance_gates"]["replay_complete"] is True
    assert report["challenger_metrics"]["unmatched_challenger_observations"] == 0
