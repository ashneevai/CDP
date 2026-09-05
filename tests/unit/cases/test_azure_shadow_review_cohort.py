import json

from evaluation.build_azure_shadow_review_cohort import build


def test_review_queue_is_deterministic_package_diverse_and_untrusted(tmp_path):
    rows = []
    for cls in ("UB04", "UNKNOWN"):
        for index in range(120):
            rows.append(
                {
                    "package_id": f"package_{index % 20:02d}",
                    "source_asset_id": f"asset_{index}",
                    "source_page_id": f"{cls}_{index:03d}",
                    "candidate_class": cls,
                    "classification_confidence": 0.8,
                    "candidate_record_sha256": str(index),
                }
            )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": rows}), encoding="utf-8")
    first = build(source, tmp_path / "a.json")
    second = build(source, tmp_path / "b.json")
    assert first == second and first["selected_pages"] == 200
    assert first["candidate_class_counts"] == {"CMS1500": 0, "UB04": 100, "UNKNOWN": 100}
    assert not first["trusted_ground_truth"] and not first["scoring_eligible"]
    assert len({r["package_id"] for r in first["records"]}) == 20
