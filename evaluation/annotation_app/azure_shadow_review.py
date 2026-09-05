"""Fast-track Azure shadow human review inside the existing annotation app."""

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

from evaluation.annotation_app import real_data_review
from packages.domain.enums import ClaimFormType
from packages.image_quality import assess_image_quality
from packages.templates.registry import TemplateRegistry

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "evaluation_results/azure_live_shadow/review_cohort_candidates.json"
PAGE_EVENTS = Path(
    os.getenv("AZURE_SHADOW_PAGE_REVIEWS", ROOT / ".runs/azure_shadow/page_reviews.jsonl")
)
ANNOTATIONS = Path(
    os.getenv("AZURE_SHADOW_ANNOTATIONS", ROOT / ".runs/azure_shadow/annotations.jsonl")
)
ADJUDICATIONS = Path(
    os.getenv("AZURE_SHADOW_ADJUDICATIONS", ROOT / ".runs/azure_shadow/adjudications.jsonl")
)
PAGE_CLASSES = {"CMS1500", "UB04", "ATTACHMENT", "SUPPORTING_DOCUMENT", "NON_CLAIM", "UNKNOWN"}
QUALITY = {"HIGH", "MEDIUM", "LOW", "UNREADABLE", "UNKNOWN"}
PAGE_ACTIONS = {"CONFIRM", "CORRECT", "UNKNOWN", "SKIP"}
BOUNDARY_ACTIONS = {"CONFIRM_DOCUMENT_START", "CONFIRM_DOCUMENT_END", "SPLIT", "MERGE", "UNKNOWN"}
STATES = {"VALUE", "NOT_PRESENT", "UNREADABLE", "NOT_APPLICABLE"}
FIELDS = (
    "member_id",
    "subscriber_id",
    "patient_name",
    "insured_name",
    "provider_name",
    "NPI",
    "patient_DOB",
    "service_date",
    "total_charge",
    "principal_diagnosis",
)
CRITICAL = {
    "member_id",
    "subscriber_id",
    "NPI",
    "patient_DOB",
    "service_date",
    "total_charge",
    "principal_diagnosis",
}
router = APIRouter(prefix="/real-review/fast-track", tags=["azure-shadow-review"])
TEMPLATES = TemplateRegistry.load_from_directory()
FIELD_REGION_ALIASES = {
    "CMS1500": {
        "member_id": "insured_id_number",
        "subscriber_id": "insured_id_number",
        "patient_name": "patient_name",
        "insured_name": "insured_name",
        "provider_name": "billing_provider_info",
        "NPI": "billing_provider_info",
        "patient_DOB": "patient_dob",
        "service_date": "date_from",
        "total_charge": "total_charge",
        "principal_diagnosis": "diagnosis_codes",
    },
    "UB04": {
        "member_id": "insured_unique_id",
        "subscriber_id": "insured_unique_id",
        "patient_name": "patient_name",
        "insured_name": "insured_name",
        "provider_name": "provider_name_address",
        "NPI": "provider_npi",
        "patient_DOB": "patient_dob",
        "service_date": "service_date",
        "total_charge": "total_charges",
        "principal_diagnosis": "principal_diagnosis",
    },
}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _reviewer(request: Request) -> str:
    return real_data_review._reviewer(request)


def _queue() -> list[dict[str, Any]]:
    if not QUEUE.exists():
        raise HTTPException(503, "fast-track queue unavailable")
    source = {r["page_id"]: r for r in real_data_review._records()}
    return [
        source[r["source_page_id"]] | {"queue_candidate": r}
        for r in json.loads(QUEUE.read_text("utf-8"))["records"]
        if r["source_page_id"] in source
    ]


def _item(index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _queue()
    if index not in range(len(items)):
        raise HTTPException(404, "queue page not found")
    return items, items[index]


def _page_latest() -> dict[str, dict]:
    latest = {}
    for row in _rows(PAGE_EVENTS):
        latest[row["source_page_id"]] = row
    return latest


def _annotation_latest() -> dict[tuple[str, str, str], dict]:
    latest = {}
    for row in _rows(ANNOTATIONS):
        latest[(row["source_page_id"], row["field_name"], row["annotator_role"])] = row
    return latest


def _adjudication_latest() -> dict[tuple[str, str], dict]:
    latest = {}
    for row in _rows(ADJUDICATIONS):
        latest[(row["source_page_id"], row["field_name"])] = row
    return latest


def _normalized_annotation(row: dict) -> tuple[str, str]:
    return row["state"], " ".join(str(row.get("value") or "").upper().split())


def _annotation_progress() -> dict[str, Any]:
    annotations = _annotation_latest()
    adjudications = _adjudication_latest()
    pairs = {(page, field) for page, field, _ in annotations if field in CRITICAL}
    agreements = []
    pending = 0
    dual = 0
    for page, field in pairs:
        a = annotations.get((page, field, "ANNOTATOR_A"))
        b = annotations.get((page, field, "ANNOTATOR_B"))
        if not a or not b or a["annotator_id"] == b["annotator_id"]:
            continue
        dual += 1
        if _normalized_annotation(a) == _normalized_annotation(b):
            agreements.append(
                {
                    "state": "DUAL_REVIEW_AGREED",
                    "source_page_id": page,
                    "field_name": field,
                    "annotation_ids": [a["annotation_id"], b["annotation_id"]],
                }
            )
        elif (page, field) not in adjudications:
            pending += 1
    return {
        "total": len(annotations),
        "critical": sum(row.get("critical", False) for row in annotations.values()),
        "dual_reviewed_critical": dual,
        "pending_adjudications": pending,
        "agreements": agreements,
    }


def _progress(items: list[dict]) -> dict:
    latest = _page_latest()
    reviewed = [r for r in latest.values() if r["action"] != "SKIP"]
    classes = {name: sum(r["reviewed_class"] == name for r in reviewed) for name in PAGE_CLASSES}
    return {
        "total": len(items),
        "reviewed": len(reviewed),
        "remaining": len(items) - len(reviewed),
        "confirmed_claim_form_pages": classes["CMS1500"] + classes["UB04"],
        "unknown": classes["UNKNOWN"],
        "classes": classes,
    }


def _quality(record: dict[str, Any]) -> dict[str, Any]:
    """Compute deterministic quality evidence for the exact source frame."""
    try:
        with Image.open(real_data_review._source(record)) as image:
            image.seek(record["page_number"] - 1)
            frame = image.copy()
        evidence = assess_image_quality(frame)
    except (OSError, EOFError) as exc:
        raise HTTPException(409, "TIFF frame unreadable for quality assessment") from exc
    extrema = frame.convert("L").getextrema()
    return {
        "blur": round(evidence.blur_score, 3),
        "contrast": round(evidence.contrast, 4),
        "noise": round(evidence.noise_estimate, 4),
        "skew": round(evidence.skew_degrees, 3),
        "text_density": round(evidence.text_density, 4),
        "dynamic_range": extrema[1] - extrema[0],
        "quality_score": round(evidence.quality_score, 4),
        "reason_codes": evidence.reason_codes,
    }


def _recommended_quality(evidence: dict[str, Any]) -> str:
    unreadable = {"UNREADABLE", "NO_CONTENT", "EMPTY_PAGE", "NO_TEXT", "TEXT_NOT_DETECTED"}
    if unreadable.intersection(evidence.get("reason_codes", [])):
        return "UNREADABLE"
    score = evidence.get("quality_score", 0)
    if (
        score >= 0.90
        and evidence.get("contrast", 0) >= 0.50
        and evidence.get("dynamic_range", 0) >= 180
        and abs(evidence.get("skew", 999)) <= 2
        and evidence.get("noise", 999) <= 0.10
    ):
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "UNKNOWN"


def _options(values, selected: str) -> str:
    return "".join(
        f'<option value="{html.escape(value)}"'
        f"{' selected' if value == selected else ''}>{html.escape(value)}</option>"
        for value in values
    )


def _frame(record: dict[str, Any]) -> Image.Image:
    try:
        with Image.open(real_data_review._source(record)) as image:
            image.seek(record["page_number"] - 1)
            return image.convert("L")
    except (OSError, EOFError) as exc:
        raise HTTPException(409, "TIFF frame unreadable") from exc


def _png_bytes(image: Image.Image) -> bytes:
    data = io.BytesIO()
    image.convert("L").save(data, "PNG", optimize=False, compress_level=9)
    return data.getvalue()


def _reviewed_form(record: dict[str, Any]) -> str:
    review = _page_latest().get(record["page_id"])
    reviewed = review.get("reviewed_class") if review else None
    if reviewed in {"CMS1500", "UB04"}:
        return reviewed
    return record["queue_candidate"]["candidate_class"]


def _field_crop(record: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name not in FIELDS:
        raise HTTPException(404, "unsupported field crop")
    form = _reviewed_form(record)
    alias = FIELD_REGION_ALIASES.get(form, {}).get(field_name)
    if not alias:
        raise HTTPException(409, "FIELD CROP UNAVAILABLE")
    enum = ClaimFormType.CMS1500 if form == "CMS1500" else ClaimFormType.UB04
    try:
        template = TEMPLATES.latest_for_form_type(enum)
    except KeyError as exc:
        raise HTTPException(409, "FIELD CROP UNAVAILABLE") from exc
    region = template.field_region(alias)
    if region:
        reference_box = (region.x0, region.y0, region.x1, region.y1)
    else:
        table = template.service_line_region
        column = (
            next((item for item in table.columns if item.field_name == alias), None)
            if table
            else None
        )
        if not table or not column:
            raise HTTPException(409, "FIELD CROP UNAVAILABLE")
        reference_box = (column.x0, table.table_y0, column.x1, table.table_y1)
    frame = _frame(record)
    ref = template.reference_dimensions
    box = (
        max(0, round(reference_box[0] * frame.width / ref.width_px)),
        max(0, round(reference_box[1] * frame.height / ref.height_px)),
        min(frame.width, round(reference_box[2] * frame.width / ref.width_px)),
        min(frame.height, round(reference_box[3] * frame.height / ref.height_px)),
    )
    if box[2] <= box[0] or box[3] <= box[1] or box == (0, 0, frame.width, frame.height):
        raise HTTPException(409, "FIELD CROP UNAVAILABLE")
    crop_bytes = _png_bytes(frame.crop(box))
    return {
        "bytes": crop_bytes,
        "crop_box": list(box),
        "source_region_sha256": hashlib.sha256(crop_bytes).hexdigest(),
        "source_page_sha256": hashlib.sha256(_png_bytes(frame)).hexdigest(),
        "localization_method": "TEMPLATE_SCALE",
        "localization_version": f"{template.template_id}@{template.version}",
    }


def _next_unreviewed(items: list[dict], current: int) -> int:
    done = set(_page_latest())
    for offset in range(1, len(items) + 1):
        index = (current + offset) % len(items)
        if items[index]["page_id"] not in done:
            return index
    return current


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    reviewer = html.escape(_reviewer(request))
    items = _queue()
    p = _progress(items)
    a = _annotation_progress()
    classes = p["classes"]
    return f"""<meta charset=utf-8><h1>Azure shadow trusted review</h1>
<p>Reviewer {reviewer}</p>
<p>Reviewed {p["reviewed"]} / {p["total"]}; CMS1500 {classes["CMS1500"]};
UB04 {classes["UB04"]}; Supporting {classes["SUPPORTING_DOCUMENT"]};
Attachment {classes["ATTACHMENT"]}; Non-claim {classes["NON_CLAIM"]};
Unknown {classes["UNKNOWN"]}; Remaining {p["remaining"]}.</p>
<p>Field annotations {a["total"]}; critical annotations {a["critical"]};
dual-reviewed critical fields {a["dual_reviewed_critical"]};
pending adjudications {a["pending_adjudications"]}.</p>
<p><a href="/real-review/fast-track/0">Start / resume queue</a></p>"""


@router.get("/{index}/image")
def image(index: int, request: Request):
    _reviewer(request)
    _, r = _item(index)
    try:
        with Image.open(real_data_review._source(r)) as im:
            im.seek(r["page_number"] - 1)
            data = io.BytesIO()
            im.convert("L").save(data, "PNG")
    except (OSError, EOFError) as exc:
        raise HTTPException(409, "TIFF frame unreadable") from exc
    return Response(data.getvalue(), media_type="image/png")


@router.get("/{index}/field-crop/{field_name}")
def field_crop(index: int, field_name: str, request: Request):
    _reviewer(request)
    _, record = _item(index)
    crop = _field_crop(record, field_name)
    return Response(crop["bytes"], media_type="image/png")


@router.get("/{index}", response_class=HTMLResponse)
def page(index: int, request: Request):
    html.escape(_reviewer(request))
    items, r = _item(index)
    p = _progress(items)
    annotations = _annotation_progress()
    candidate = r["queue_candidate"]
    nxt = min(len(items) - 1, index + 1)
    prev = max(0, index - 1)
    unreviewed = _next_unreviewed(items, index)
    boundary = r.get("boundary") or {}
    measured_quality = _quality(r)
    quality_json = html.escape(json.dumps(measured_quality, sort_keys=True))
    candidate_class = candidate.get("candidate_class", "UNKNOWN")
    selected_class = candidate_class if candidate_class in PAGE_CLASSES else "UNKNOWN"
    proposed_quality = candidate.get("source_quality_band", "UNKNOWN")
    selected_quality = (
        proposed_quality
        if proposed_quality in QUALITY and proposed_quality != "UNKNOWN"
        else _recommended_quality(measured_quality)
    )
    prior_review = _page_latest().get(r["page_id"], {})
    selected_boundary = prior_review.get("boundary_action", "UNKNOWN")
    if selected_boundary not in BOUNDARY_ACTIONS:
        selected_boundary = "UNKNOWN"
    class_opts = _options(sorted(PAGE_CLASSES), selected_class)
    quality_opts = _options(sorted(QUALITY), selected_quality)
    boundary_opts = _options(sorted(BOUNDARY_ACTIONS), selected_boundary)
    field_opts = _options(FIELDS, FIELDS[0])
    classes = p["classes"]
    return f"""<meta charset=utf-8><style>body{{font:15px Arial;margin:20px}}label{{display:block;margin:8px}}img{{max-width:95vw;max-height:62vh}}#field-crop{{display:block;max-width:80vw;max-height:25vh;border:1px solid #555}}button,select,input,textarea{{padding:6px}}</style><h1>Fast-track {index + 1}/{len(items)}</h1><p>Reviewed {p["reviewed"]} / {p["total"]}; CMS1500 {classes["CMS1500"]}; UB04 {classes["UB04"]}; Supporting {classes["SUPPORTING_DOCUMENT"]}; Attachment {classes["ATTACHMENT"]}; Non-claim {classes["NON_CLAIM"]}; Unknown {classes["UNKNOWN"]}; Remaining {p["remaining"]}.</p><p>Field annotations {annotations["total"]}; critical annotations {annotations["critical"]}; dual-reviewed critical fields {annotations["dual_reviewed_critical"]}; pending adjudications {annotations["pending_adjudications"]}.</p><p><a id=prev href=/real-review/fast-track/{prev}>Previous</a> | <a id=next href=/real-review/fast-track/{nxt}>Next</a> | <a href=/real-review/fast-track/{unreviewed}>Next unreviewed</a></p><p>Package: {html.escape(r["package_id"])}<br>Asset: {html.escape(r["asset_id"])}<br>Page: {html.escape(r["page_id"])}; frame {r["page_number"]}<br>Proposed class: <b>{html.escape(candidate_class)}</b>; confidence {candidate["classification_confidence"]}<br>Proposed quality: {html.escape(proposed_quality)}<br>Measured quality: {quality_json}<br>Suggested boundary: {html.escape(str(boundary.get("boundary_state", "UNKNOWN")))}</p><img src=/real-review/fast-track/{index}/image><form id=page-review-form method=post action=/real-review/fast-track/{index}/page-review><label>Action <select name=action><option>CONFIRM</option><option>CORRECT</option><option>UNKNOWN</option><option>SKIP</option></select></label><label>Reviewed class <select name=reviewed_class>{class_opts}</select></label><label>Quality <select name=reviewed_quality_band>{quality_opts}</select></label><label>Boundary <select name=boundary_action>{boundary_opts}</select></label><label>Correction reason <input name=correction_reason></label><button type=submit>Save and next unreviewed</button></form><hr><h2>Blind field annotation</h2><p>CDP and Azure predictions are hidden.</p><form id=field-annotation-form method=post action=/real-review/fast-track/{index}/annotation><label>Field <select id=field-name name=field_name>{field_opts}</select></label><p>Field crop &mdash; <span id=field-label>{FIELDS[0]}</span></p><img id=field-crop alt="FIELD CROP UNAVAILABLE" src=/real-review/fast-track/{index}/field-crop/{FIELDS[0]}><label>Role <select name=annotator_role><option>ANNOTATOR_A</option><option>ANNOTATOR_B</option></select></label><label>State <select name=state><option>VALUE</option><option>NOT_PRESENT</option><option>UNREADABLE</option><option>NOT_APPLICABLE</option></select></label><label>Value <input name=value autocomplete=off></label><label>Notes <textarea name=notes></textarea></label><button type=submit>Save blind annotation</button></form><script>const field=document.querySelector('#field-name'),crop=document.querySelector('#field-crop'),label=document.querySelector('#field-label');field.addEventListener('change',()=>{{label.textContent=field.value;crop.src='/real-review/fast-track/{index}/field-crop/'+encodeURIComponent(field.value)}});document.addEventListener('keydown',e=>{{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;if(e.key.toLowerCase()==='n')location=document.querySelector('#next').href;if(e.key.toLowerCase()==='p')location=document.querySelector('#prev').href;if(e.key.toLowerCase()==='u')document.querySelector('[name=action]').value='UNKNOWN';if(e.key.toLowerCase()==='c')document.querySelector('[name=action]').value='CONFIRM';if(e.key.toLowerCase()==='s')document.querySelector('#page-review-form').requestSubmit()}})</script>"""


@router.post("/{index}/page-review")
def page_review(
    index: int,
    request: Request,
    action: str = Form(...),
    reviewed_class: str = Form(...),
    reviewed_quality_band: str = Form(...),
    boundary_action: str = Form(...),
    correction_reason: str = Form(""),
):
    reviewer = _reviewer(request)
    items, r = _item(index)
    if (
        action not in PAGE_ACTIONS
        or reviewed_class not in PAGE_CLASSES
        or reviewed_quality_band not in QUALITY
        or boundary_action not in BOUNDARY_ACTIONS
    ):
        raise HTTPException(400, "unsupported review decision")
    if action == "CORRECT" and not correction_reason.strip():
        raise HTTPException(400, "correction reason required")
    _append(
        PAGE_EVENTS,
        {
            "review_id": str(uuid.uuid4()),
            "reviewer_id": reviewer,
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "candidate_class": r["queue_candidate"]["candidate_class"],
            "reviewed_class": reviewed_class,
            "candidate_quality_band": r["queue_candidate"]["source_quality_band"],
            "reviewed_quality_band": reviewed_quality_band,
            "boundary_action": boundary_action,
            "correction_reason": correction_reason.strip() or None,
            "source_page_id": r["page_id"],
            "source_asset_id": r["asset_id"],
            "package_id": r["package_id"],
            "candidate_evidence_sha256": r["queue_candidate"]["candidate_record_sha256"],
        },
    )
    return RedirectResponse(f"/real-review/fast-track/{_next_unreviewed(items, index)}", 303)


@router.post("/{index}/annotation")
def annotation(
    index: int,
    request: Request,
    field_name: str = Form(...),
    annotator_role: str = Form(...),
    state: str = Form(...),
    value: str = Form(""),
    notes: str = Form(""),
):
    reviewer = _reviewer(request)
    _, r = _item(index)
    crop = _field_crop(r, field_name)
    if (
        field_name not in FIELDS
        or annotator_role not in {"ANNOTATOR_A", "ANNOTATOR_B"}
        or state not in STATES
    ):
        raise HTTPException(400, "unsupported annotation")
    if (state == "VALUE") != bool(value.strip()):
        raise HTTPException(400, "VALUE requires value; non-VALUE forbids it")
    latest = _annotation_latest()
    other_role = "ANNOTATOR_B" if annotator_role == "ANNOTATOR_A" else "ANNOTATOR_A"
    other = latest.get((r["page_id"], field_name, other_role))
    if other and other["annotator_id"] == reviewer:
        raise HTTPException(409, "critical dual review requires independent annotators")
    prior = latest.get((r["page_id"], field_name, annotator_role))
    if prior and prior["annotator_id"] != reviewer:
        raise HTTPException(409, "annotator role already owned by another reviewer")
    _append(
        ANNOTATIONS,
        {
            "annotation_id": str(uuid.uuid4()),
            "annotator_id": reviewer,
            "annotator_role": annotator_role,
            "timestamp": datetime.now(UTC).isoformat(),
            "package_id": r["package_id"],
            "source_page_id": r["page_id"],
            "field_name": field_name,
            "critical": field_name in CRITICAL,
            "state": state,
            "value": value.strip() or None,
            "value_sha256": hashlib.sha256(value.strip().upper().encode()).hexdigest()
            if value.strip()
            else None,
            "source_region_sha256": crop["source_region_sha256"],
            "crop_box": crop["crop_box"],
            "localization_method": crop["localization_method"],
            "localization_version": crop["localization_version"],
            "source_page_sha256": crop["source_page_sha256"],
            "notes": notes.strip() or None,
            "authority": "HUMAN_SINGLE_REVIEW",
            "prediction_visible": False,
        },
    )
    return RedirectResponse(f"/real-review/fast-track/{index}", 303)


@router.get("/{index}/adjudication/{field_name}", response_class=HTMLResponse)
def adjudication_screen(index: int, field_name: str, request: Request):
    reviewer = _reviewer(request)
    _, row = _item(index)
    latest = _annotation_latest()
    a = latest.get((row["page_id"], field_name, "ANNOTATOR_A"))
    b = latest.get((row["page_id"], field_name, "ANNOTATOR_B"))
    if not a or not b or (a["state"], a["value_sha256"]) == (b["state"], b["value_sha256"]):
        raise HTTPException(409, "adjudication requires a dual-review disagreement")
    if reviewer in {a["annotator_id"], b["annotator_id"]}:
        raise HTTPException(403, "adjudicator must be independent")
    options = "".join(f"<option>{state}</option>" for state in sorted(STATES))
    return f"""<meta charset=utf-8><h1>Independent adjudication: {html.escape(field_name)}</h1><img style="max-width:95vw;max-height:65vh" src=/real-review/fast-track/{index}/image><p>Annotator A: {html.escape(a["state"])} / {html.escape(str(a["value"]))}</p><p>Annotator B: {html.escape(b["state"])} / {html.escape(str(b["value"]))}</p><form method=post><label>Final state <select name=state>{options}</select></label><label>Final value <input name=value autocomplete=off></label><button>Finalize</button></form>"""


@router.post("/{index}/adjudication/{field_name}")
def adjudicate(
    index: int, field_name: str, request: Request, state: str = Form(...), value: str = Form("")
):
    reviewer = _reviewer(request)
    _, row = _item(index)
    latest = _annotation_latest()
    a = latest.get((row["page_id"], field_name, "ANNOTATOR_A"))
    b = latest.get((row["page_id"], field_name, "ANNOTATOR_B"))
    if not a or not b or reviewer in {a["annotator_id"], b["annotator_id"]}:
        raise HTTPException(403, "independent dual-review adjudication required")
    if (a["state"], a["value_sha256"]) == (b["state"], b["value_sha256"]):
        raise HTTPException(409, "matching annotations do not require adjudication")
    if state not in STATES or ((state == "VALUE") != bool(value.strip())):
        raise HTTPException(400, "invalid final state/value")
    _append(
        ADJUDICATIONS,
        {
            "adjudication_id": str(uuid.uuid4()),
            "adjudicator_id": reviewer,
            "timestamp": datetime.now(UTC).isoformat(),
            "package_id": row["package_id"],
            "source_page_id": row["page_id"],
            "field_name": field_name,
            "annotation_a_id": a["annotation_id"],
            "annotation_b_id": b["annotation_id"],
            "final_state": state,
            "final_value": value.strip() or None,
            "final_value_sha256": hashlib.sha256(value.strip().upper().encode()).hexdigest()
            if value.strip()
            else None,
            "authority": "HUMAN_ADJUDICATED",
            "prediction_visible": False,
        },
    )
    return RedirectResponse(f"/real-review/fast-track/{index}", 303)
