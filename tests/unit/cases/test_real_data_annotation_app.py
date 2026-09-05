import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from evaluation.annotation_app import real_data_review as review


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), "utf-8")


def setup_review(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    image = root / "Group A" / "page.001"
    image.parent.mkdir()
    Image.new("1", (8, 8), 1).save(image, "TIFF")
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    closure = tmp_path / "closure"
    events = tmp_path / "events.jsonl"
    dump(source, {"records": [{"sha256": sha, "relative_path": "Group A/page.001"}]})
    dump(
        closure / "source_inventory.json",
        {"assets": [{"asset_id": "asset", "package_id": "package", "sha256": sha}]},
    )
    dump(
        closure / "document_boundaries.json",
        {
            "candidates": [
                {
                    "asset_ids": ["asset"],
                    "candidate_document_id": "doc",
                    "boundary_state": "IDENTITY_CANDIDATE",
                }
            ]
        },
    )
    dump(
        closure / "page_classification.json",
        {
            "pages": [
                {
                    "asset_id": "asset",
                    "package_id": "package",
                    "page_id": "page",
                    "page_number": 1,
                    "classification": "UNKNOWN",
                    "classification_authority": "NONE",
                    "cdp_prediction": "SECRET",
                }
            ]
        },
    )
    monkeypatch.setattr(review, "SOURCE_ROOT", root)
    monkeypatch.setattr(review, "SOURCE_RECORDS", source)
    monkeypatch.setattr(review, "CLOSURE", closure)
    monkeypatch.setattr(review, "EVENTS", events)
    app = FastAPI()
    app.include_router(review.router)
    return TestClient(app), events, image


def test_package_navigation_and_allowlisted_frame(tmp_path, monkeypatch):
    client, _, _ = setup_review(tmp_path, monkeypatch)
    headers = {"X-Reviewer-ID": "reviewer"}
    assert "package" in client.get("/real-review/", headers=headers).text
    screen = client.get("/real-review/package/0", headers=headers)
    assert (
        screen.status_code == 200
        and "Candidate class" in screen.text
        and "SECRET" not in screen.text
    )
    rendered = client.get("/real-review/package/0/image", headers=headers)
    assert rendered.status_code == 200 and rendered.headers["content-type"] == "image/png"


def test_field_submission_persists_hashes_not_phi(tmp_path, monkeypatch):
    client, events, _ = setup_review(tmp_path, monkeypatch)
    headers = {"X-Reviewer-ID": "reviewer"}
    secret = "Jane Sensitive"
    response = client.post(
        "/real-review/package/0",
        headers=headers,
        data={
            "target": "FIELD",
            "action": "CONFIRM",
            "new_value": secret,
            "field_name": "patient_name",
            "source_region_sha256": "a" * 64,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    raw = events.read_text("utf-8")
    event = json.loads(raw)
    assert (
        secret not in raw
        and event["new_value_sha256"] == review._hash(secret)
        and event["prediction_visible"] is False
    )
    assert event["reviewer_id"] == "reviewer" and event["annotation_version"] == review.VERSION


def test_correction_has_previous_hash_and_reason(tmp_path, monkeypatch):
    client, events, _ = setup_review(tmp_path, monkeypatch)
    headers = {"X-Reviewer-ID": "reviewer"}
    base = {"target": "PAGE", "new_value": "CMS1500"}
    assert (
        client.post(
            "/real-review/package/0", headers=headers, data=base | {"action": "CONFIRM"}
        ).status_code
        == 200
    )
    response = client.post(
        "/real-review/package/0",
        headers=headers,
        data=base | {"action": "CORRECT", "new_value": "UB04", "reason_code": "WRONG_TYPE"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    rows = [json.loads(x) for x in events.read_text().splitlines()]
    assert (
        rows[1]["previous_value_sha256"] == review._hash("CMS1500")
        and rows[1]["correction_reason_code"] == "WRONG_TYPE"
    )


def test_source_file_must_still_match_frozen_hash(tmp_path, monkeypatch):
    _, _, image = setup_review(tmp_path, monkeypatch)
    record = review._records()[0]
    image.write_bytes(b"changed")
    with pytest.raises(Exception, match="allowlist"):
        review._source(record)


def test_runtime_candidates_overlay_closure_seed_by_source_page(tmp_path, monkeypatch):
    client, _, _ = setup_review(tmp_path, monkeypatch)
    real_eval = tmp_path / "real_eval"
    dump(
        real_eval / "page_classification_candidates.json",
        {
            "records": [
                {
                    "source_page_id": "page",
                    "candidate_class": "UB04",
                    "confidence_band": "MEDIUM",
                    "reason_codes": ["GRID_AND_ANCHOR"],
                    "anchor_evidence": [{"id": 1}],
                    "text_evidence": [{"id": 2}, {"id": 3}],
                    "layout_evidence": [],
                    "grid_line_evidence": [{"id": 4}],
                }
            ]
        },
    )
    dump(
        real_eval / "document_boundary_candidates.json",
        {
            "records": [
                {
                    "source_page_id": "page",
                    "candidate_document_id": "runtime-doc",
                    "boundary_state": "SPLIT_CANDIDATE",
                }
            ]
        },
    )
    monkeypatch.setattr(review, "REAL_EVAL", real_eval)
    screen = client.get("/real-review/package/0", headers={"X-Reviewer-ID": "reviewer"})
    assert screen.status_code == 200
    assert "UB04" in screen.text and "MEDIUM" in screen.text and "GRID_AND_ANCHOR" in screen.text
    assert "anchor_evidence=1" in screen.text and "text_evidence=2" in screen.text
    assert "SPLIT_CANDIDATE" in screen.text and "&mdash;" in screen.text


def test_missing_runtime_candidates_fall_back_to_closure_seed(tmp_path, monkeypatch):
    client, _, _ = setup_review(tmp_path, monkeypatch)
    monkeypatch.setattr(review, "REAL_EVAL", tmp_path / "missing")
    screen = client.get("/real-review/package/0", headers={"X-Reviewer-ID": "reviewer"})
    assert "UNKNOWN" in screen.text and "IDENTITY_CANDIDATE" in screen.text
