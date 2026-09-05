import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from evaluation.annotation_app import azure_shadow_review as azure
from evaluation.annotation_app import real_data_review as real


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup(tmp_path, monkeypatch, candidate_class="UB04", page_count=1):
    root = tmp_path / "source"
    image = root / "pkg" / "a.tif"
    image.parent.mkdir(parents=True)
    page = Image.new("L", (400, 520), 255)
    draw = ImageDraw.Draw(page)
    for y in range(20, 500, 20):
        draw.line((10, y, 390, y), fill=0, width=2)
    page.save(image, save_all=True, append_images=[page.copy() for _ in range(page_count - 1)])
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    closure = tmp_path / "closure"
    source = tmp_path / "source.json"
    dump(source, {"records": [{"sha256": sha, "relative_path": "pkg/a.tif"}]})
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
                    "boundary_state": "CANDIDATE_END",
                }
            ]
        },
    )
    pages = [
        {
            "asset_id": "asset",
            "package_id": "package",
            "page_id": f"page-{number}",
            "page_number": number,
            "classification": "UNKNOWN",
        }
        for number in range(1, page_count + 1)
    ]
    dump(closure / "page_classification.json", {"pages": pages})
    queue = tmp_path / "queue.json"
    dump(
        queue,
        {
            "records": [
                {
                    "package_id": "package",
                    "source_asset_id": "asset",
                    "source_page_id": row["page_id"],
                    "candidate_class": candidate_class,
                    "classification_confidence": 0.8,
                    "candidate_record_sha256": f"{number:064x}",
                    "source_quality_band": "UNKNOWN",
                }
                for number, row in enumerate(pages, 1)
            ]
        },
    )
    monkeypatch.setattr(real, "SOURCE_ROOT", root)
    monkeypatch.setattr(real, "SOURCE_RECORDS", source)
    monkeypatch.setattr(real, "CLOSURE", closure)
    monkeypatch.setattr(real, "REAL_EVAL", tmp_path / "none")
    monkeypatch.setattr(azure, "QUEUE", queue)
    monkeypatch.setattr(azure, "PAGE_EVENTS", tmp_path / "pages.jsonl")
    monkeypatch.setattr(azure, "ANNOTATIONS", tmp_path / "annotations.jsonl")
    monkeypatch.setattr(azure, "ADJUDICATIONS", tmp_path / "adjudications.jsonl")
    app = FastAPI()
    app.include_router(azure.router)
    return TestClient(app)


def headers(reviewer="reviewer"):
    return {"X-Reviewer-ID": reviewer}


@pytest.mark.parametrize("candidate_class", ["UB04", "CMS1500", "UNKNOWN"])
def test_candidate_class_is_selected_but_get_does_not_review(
    tmp_path, monkeypatch, candidate_class
):
    client = setup(tmp_path, monkeypatch, candidate_class)
    response = client.get("/real-review/fast-track/0", headers=headers())
    assert response.status_code == 200
    assert f'<option value="{candidate_class}" selected>{candidate_class}</option>' in response.text
    assert not azure.PAGE_EVENTS.exists()


def test_high_measured_quality_defaults_high_and_boundary_defaults_unknown(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        azure,
        "_quality",
        lambda _: {
            "quality_score": 0.964,
            "contrast": 0.75,
            "dynamic_range": 255,
            "skew": 0.5,
            "noise": 0.05,
            "reason_codes": [],
        },
    )
    screen = client.get("/real-review/fast-track/0", headers=headers()).text
    assert '<option value="HIGH" selected>HIGH</option>' in screen
    assert '<option value="UNKNOWN" selected>UNKNOWN</option>' in screen
    assert "Suggested boundary: CANDIDATE_END" in screen
    assert 'name="source_region_sha256"' not in screen
    assert "CDP and Azure predictions are hidden" in screen
    assert "requestSubmit()" in screen and "#page-review-form" in screen


def test_page_review_persists_and_redirects_to_next_unreviewed(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch, page_count=2)
    response = client.post(
        "/real-review/fast-track/0/page-review",
        headers=headers(),
        data={
            "action": "CONFIRM",
            "reviewed_class": "UB04",
            "reviewed_quality_band": "HIGH",
            "boundary_action": "UNKNOWN",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/real-review/fast-track/1"
    event = json.loads(azure.PAGE_EVENTS.read_text())
    assert event["reviewed_class"] == "UB04"
    assert event["reviewed_quality_band"] == "HIGH"
    assert event["action"] == "CONFIRM"


def test_field_crop_is_smaller_and_hash_is_deterministic(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    url = "/real-review/fast-track/0/field-crop/NPI"
    first = client.get(url, headers=headers())
    second = client.get(url, headers=headers())
    assert first.status_code == 200 and first.content == second.content
    crop = Image.open(BytesIO(first.content))
    assert crop.width < 400 and crop.height < 520
    assert hashlib.sha256(first.content).hexdigest() == hashlib.sha256(second.content).hexdigest()


def test_annotation_uses_server_crop_hash_and_client_cannot_spoof(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    crop = client.get("/real-review/fast-track/0/field-crop/NPI", headers=headers("a")).content
    expected = hashlib.sha256(crop).hexdigest()
    response = client.post(
        "/real-review/fast-track/0/annotation",
        headers=headers("a"),
        data={
            "field_name": "NPI",
            "annotator_role": "ANNOTATOR_A",
            "state": "VALUE",
            "value": "1234567893",
            "source_region_sha256": "f" * 64,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    event = json.loads(azure.ANNOTATIONS.read_text())
    assert event["source_region_sha256"] == expected
    assert event["source_region_sha256"] != "f" * 64
    assert event["crop_box"] != [0, 0, 400, 520]
    assert event["localization_method"] == "TEMPLATE_SCALE"
    assert event["localization_version"].startswith("ub04@")
    assert len(event["source_page_sha256"]) == 64
    assert event["prediction_visible"] is False


def test_missing_crop_fails_closed(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch, candidate_class="UNKNOWN")
    assert (
        client.get("/real-review/fast-track/0/field-crop/NPI", headers=headers()).status_code == 409
    )
    response = client.post(
        "/real-review/fast-track/0/annotation",
        headers=headers("a"),
        data={
            "field_name": "NPI",
            "annotator_role": "ANNOTATOR_A",
            "state": "VALUE",
            "value": "1234567893",
        },
    )
    assert response.status_code == 409
    assert not azure.ANNOTATIONS.exists()


def test_blind_dual_annotation_requires_independent_reviewers(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    data = {
        "field_name": "NPI",
        "state": "VALUE",
        "value": "1234567893",
    }
    assert (
        client.post(
            "/real-review/fast-track/0/annotation",
            headers=headers("a"),
            data=data | {"annotator_role": "ANNOTATOR_A"},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert (
        client.post(
            "/real-review/fast-track/0/annotation",
            headers=headers("a"),
            data=data | {"annotator_role": "ANNOTATOR_B"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/real-review/fast-track/0/annotation",
            headers=headers("b"),
            data=data | {"annotator_role": "ANNOTATOR_B"},
            follow_redirects=False,
        ).status_code
        == 303
    )
    progress = azure._annotation_progress()
    assert progress["dual_reviewed_critical"] == 1
    assert progress["agreements"][0]["state"] == "DUAL_REVIEW_AGREED"
    assert len(progress["agreements"][0]["annotation_ids"]) == 2


def test_disagreement_requires_independent_adjudicator(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    base = {"field_name": "NPI", "state": "VALUE"}
    for reviewer, role, value in [("a", "ANNOTATOR_A", "123"), ("b", "ANNOTATOR_B", "456")]:
        client.post(
            "/real-review/fast-track/0/annotation",
            headers=headers(reviewer),
            data=base | {"annotator_role": role, "value": value},
        )
    assert azure._annotation_progress()["pending_adjudications"] == 1
    assert (
        client.get("/real-review/fast-track/0/adjudication/NPI", headers=headers("a")).status_code
        == 403
    )
    assert (
        client.get(
            "/real-review/fast-track/0/adjudication/NPI", headers=headers("judge")
        ).status_code
        == 200
    )
    result = client.post(
        "/real-review/fast-track/0/adjudication/NPI",
        headers=headers("judge"),
        data={"state": "VALUE", "value": "456"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert json.loads(azure.ADJUDICATIONS.read_text())["authority"] == "HUMAN_ADJUDICATED"


def test_fast_track_routes_precede_real_review_catch_all():
    from evaluation.annotation_app.app import app as integrated_app

    included = [
        route.original_router
        for route in integrated_app.routes
        if hasattr(route, "original_router")
    ]
    assert included.index(azure.router) < included.index(real.router)
