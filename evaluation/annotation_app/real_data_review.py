"""PHI-safe package/page review router for the real Source-B archive."""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from PIL import Image

from packages.real_data_evaluation.governance import ReviewAction, ReviewEvent, ReviewTarget

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.getenv("CDP_REAL_SOURCE_ROOT", ROOT / "evaluation_data/source_b_1000_claims"))
SOURCE_RECORDS = ROOT / "evaluation_results/closure1000/source_inventory.json"
CLOSURE = ROOT / "evaluation_results/closure"
REAL_EVAL = ROOT / "evaluation_results/real_eval"
EVENTS = Path(os.getenv("CDP_REAL_REVIEW_EVENTS", ROOT / ".runs/real_review_events.jsonl"))
VERSION = "real-source-b-review-v1"
router = APIRouter(prefix="/real-review", tags=["real-data-review"])


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(503, f"review inventory unavailable: {path.name}")
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(" ".join(value.strip().upper().split()).encode()).hexdigest()


def _reviewer(request: Request) -> str:
    value = request.headers.get("X-Reviewer-ID") or os.getenv("REAL_DATA_REVIEWER_ID")
    if not value:
        raise HTTPException(401, "reviewer identity required")
    return value


def _optional(path: Path, key: str) -> list[dict[str, Any]]:
    return _load(path).get(key, []) if path.is_file() else []


def _records() -> list[dict[str, Any]]:
    source = {r["sha256"]: r for r in _load(SOURCE_RECORDS)["records"]}
    assets = {r["asset_id"]: r for r in _load(CLOSURE / "source_inventory.json")["assets"]}
    seed_bounds = {
        a: r
        for r in _load(CLOSURE / "document_boundaries.json")["candidates"]
        for a in r["asset_ids"]
    }
    candidates = {
        r["source_page_id"]: r
        for r in _optional(REAL_EVAL / "page_classification_candidates.json", "records")
    }
    boundary_rows = _optional(REAL_EVAL / "document_boundary_candidates.json", "records")
    if not boundary_rows:
        boundary_rows = _optional(REAL_EVAL / "document_boundary_candidates.json", "candidates")
    runtime_bounds = {r["source_page_id"]: r for r in boundary_rows if r.get("source_page_id")}
    out = []
    for page in _load(CLOSURE / "page_classification.json")["pages"]:
        asset = assets.get(page["asset_id"])
        src = source.get(asset["sha256"]) if asset else None
        if asset and src:
            out.append(
                page
                | asset
                | src
                | {
                    "candidate": candidates.get(page["page_id"], {}),
                    "boundary": runtime_bounds.get(
                        page["page_id"], seed_bounds.get(page["asset_id"], {})
                    ),
                }
            )
    return sorted(out, key=lambda r: (r["package_id"], r["relative_path"], r["page_number"]))


def _record(package: str, index: int):
    pages = [r for r in _records() if r["package_id"] == package]
    if not pages or index not in range(len(pages)):
        raise HTTPException(404, "review page not found")
    return pages, pages[index]


def _source(record):
    root = SOURCE_ROOT.resolve()
    path = (root / record["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as e:
        raise HTTPException(404, "invalid source record") from e
    if not path.is_file() or _digest(path) != record["sha256"]:
        raise HTTPException(409, "source does not match allowlist")
    return path


def _previous(record, target, field):
    if not EVENTS.is_file():
        return None
    result = None
    for line in EVENTS.read_text("utf-8").splitlines():
        event = json.loads(line)
        if (
            event.get("package_id"),
            event.get("source_page_id"),
            event.get("target"),
            event.get("field_name"),
        ) == (record["package_id"], record["page_id"], target, field):
            result = event.get("new_value_sha256")
    return result


def _append(event):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")


@router.get("/", response_class=HTMLResponse)
def queue(request: Request):
    reviewer = html.escape(_reviewer(request))
    packages = sorted({r["package_id"] for r in _records()})
    links = "".join(f"<li><a href='/real-review/{p}/0'>{p}</a></li>" for p in packages)
    return f"<h1>Real Source-B review</h1><p>Reviewer: {reviewer}; {len(packages)} packages</p><ol>{links}</ol>"


@router.get("/{package}/{index}/image")
def image(package: str, index: int, request: Request):
    _reviewer(request)
    _, record = _record(package, index)
    try:
        with Image.open(_source(record)) as im:
            im.seek(record["page_number"] - 1)
            data = io.BytesIO()
            im.convert("L").save(data, "PNG")
    except (OSError, EOFError) as e:
        raise HTTPException(409, "TIFF frame unreadable") from e
    return Response(data.getvalue(), media_type="image/png")


@router.get("/{package}/{index}", response_class=HTMLResponse)
def page(package: str, index: int, request: Request):
    reviewer = html.escape(_reviewer(request))
    pages, r = _record(package, index)
    prev = max(0, index - 1)
    nxt = min(len(pages) - 1, index + 1)
    boundary = r["boundary"]
    candidate = r["candidate"]
    candidate_class = candidate.get("candidate_class", r.get("classification", "UNKNOWN"))
    confidence = candidate.get("confidence_band", "UNAVAILABLE")
    reasons = ", ".join(candidate.get("reason_codes", [])) or "UNAVAILABLE"
    names = ("anchor_evidence", "text_evidence", "layout_evidence", "grid_line_evidence")
    evidence = ", ".join(f"{name}={len(candidate.get(name, []))}" for name in names)
    return f"""<meta charset=utf-8><h1>{package} &mdash; {index + 1}/{len(pages)}</h1><p><a href=/real-review/>Packages</a> <a href=/real-review/{package}/{prev}>Previous</a> <a href=/real-review/{package}/{nxt}>Next</a></p><p>Candidate class: <b>{html.escape(candidate_class)}</b>; confidence: {html.escape(confidence)}; reasons: {html.escape(reasons)}; evidence counts: {html.escape(evidence)}; boundary: {html.escape(boundary.get("boundary_state", "UNKNOWN"))}; reviewer: {reviewer}</p><img style="max-width:100%;max-height:68vh" src=/real-review/{package}/{index}/image><form method=post><label>Target <select name=target><option>PAGE</option><option>DOCUMENT</option><option>ATTACHMENT</option><option>FIELD</option></select></label><label> Action <select name=action><option>CONFIRM</option><option>CORRECT</option><option>UNKNOWN</option><option>SPLIT_DOCUMENT</option><option>MERGE_DOCUMENT</option></select></label><p><label>Class/type/attachment semantic (or blind field value) <input name=new_value autocomplete=off></label></p><p><label>Field name <input name=field_name></label> <label>Region SHA256 <input name=source_region_sha256 pattern="[0-9a-f]{{64}}"></label></p><p><label>Reason <input name=reason_code></label></p><button>Append review</button></form><p>Field review is blind: no CDP field prediction is displayed. Clear text is hashed in memory and never persisted.</p>"""


@router.post("/{package}/{index}")
def submit(
    package: str,
    index: int,
    request: Request,
    target: str = Form(...),
    action: str = Form(...),
    new_value: str = Form(""),
    field_name: str = Form(""),
    source_region_sha256: str = Form(""),
    reason_code: str = Form(""),
):
    reviewer = _reviewer(request)
    pages, r = _record(package, index)
    try:
        t = ReviewTarget(target)
        a = ReviewAction(action)
    except ValueError as e:
        raise HTTPException(400, "unknown target or action") from e
    if (
        a in {ReviewAction.CORRECT, ReviewAction.SPLIT_DOCUMENT, ReviewAction.MERGE_DOCUMENT}
        and not reason_code.strip()
    ):
        raise HTTPException(400, "reason required")
    if t == ReviewTarget.FIELD and not new_value.strip():
        raise HTTPException(400, "blind field value required")
    field = field_name.strip() or None
    new = _hash(new_value) if new_value.strip() else None
    event = ReviewEvent(
        event_id=str(uuid.uuid4()),
        target=t,
        action=a,
        archive_id="archive_sha256_8c09d4ec4de3fef8bf41771ae45bc69b69300221781a9eea9183936cfcbe85f3",
        package_id=r["package_id"],
        source_asset_id=r["asset_id"],
        source_page_id=r["page_id"],
        document_id=r["boundary"].get("candidate_document_id"),
        field_name=field,
        reviewer_id=reviewer,
        reviewed_at=datetime.now(UTC),
        annotation_version=VERSION,
        previous_value_sha256=_previous(r, t.value, field),
        new_value_sha256=new,
        correction_reason_code=reason_code.strip() or None,
        prediction_visible=False,
        source_region_sha256=source_region_sha256.strip() or None,
    )
    _append(event)
    return RedirectResponse(f"/real-review/{package}/{min(index + 1, len(pages) - 1)}", 303)
