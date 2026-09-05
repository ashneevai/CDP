"""Lightweight local UI for TUNING_TRUTH_V1 field, crop, and UB-row annotation."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from evaluation.annotation_app.azure_shadow_review import router as azure_shadow_review_router
from evaluation.annotation_app.real_data_review import router as real_data_review_router
from evaluation.tuning_truth.contracts import (
    FieldCropTruth,
    FieldTruth,
    NormalizedBBox,
    ReviewStatus,
    TruthStatus,
    UB04ServiceLineTruth,
    Visibility,
)
from evaluation.tuning_truth.quality import validate_dataset
from evaluation.tuning_truth.schema import field_policy

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation_results/phase7a15"
MANIFEST = RESULTS / "annotation_sample_manifest.json"
TASKS = RESULTS / "annotation_tasks.jsonl"
FIELDS = RESULTS / "field_truth.jsonl"
CROPS = RESULTS / "crop_truth.jsonl"
UB_LINES = RESULTS / "ub_service_line_truth.jsonl"
QUALITY = RESULTS / "annotation_quality.json"

app = FastAPI(title="CDP Tuning Truth V1", docs_url=None, redoc_url=None)
app.include_router(azure_shadow_review_router)
app.include_router(real_data_review_router)


def _reviewer(request: Request) -> str:
    reviewer = request.headers.get("X-Reviewer-ID") or os.getenv("TUNING_TRUTH_REVIEWER_ID")
    if not reviewer:
        raise HTTPException(401, "X-Reviewer-ID header or TUNING_TRUTH_REVIEWER_ID is required")
    return reviewer


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _upsert(path: Path, row: dict[str, Any], key_fields: tuple[str, ...]) -> None:
    rows = _jsonl(path)
    key = tuple(row[field] for field in key_fields)
    rows = [item for item in rows if tuple(item[field] for field in key_fields) != key]
    rows.append(row)
    _write_jsonl(path, sorted(rows, key=lambda item: tuple(item[field] for field in key_fields)))


def _tasks() -> list[dict[str, Any]]:
    return _jsonl(TASKS)


def _task(index: int) -> dict[str, Any]:
    tasks = _tasks()
    if index < 0 or index >= len(tasks):
        raise HTTPException(404, "annotation task not found")
    return tasks[index]


def _refresh_quality() -> None:
    fields = [FieldTruth.model_validate(row) for row in _jsonl(FIELDS)]
    crops = [FieldCropTruth.model_validate(row) for row in _jsonl(CROPS)]
    lines = [UB04ServiceLineTruth.model_validate(row) for row in _jsonl(UB_LINES)]
    QUALITY.write_text(json.dumps(validate_dataset(fields, crops, lines), indent=2) + "\n", "utf-8")


def _normalized(value: str) -> str:
    return " ".join(value.strip().upper().split())


@app.get("/", response_class=HTMLResponse)
def queue(request: Request) -> str:
    reviewer = html.escape(_reviewer(request))
    tasks = _tasks()
    field_done = {(r["document_id"], r["page_id"], r["field_name"]) for r in _jsonl(FIELDS)}
    line_done = {(r["document_id"], r["page_id"]) for r in _jsonl(UB_LINES)}
    complete = sum(
        (task["document_id"], task["page_id"], task.get("field_name")) in field_done
        if task["task_type"] == "FIELD_AND_CROP"
        else (task["document_id"], task["page_id"]) in line_done
        for task in tasks
    )
    first = next(
        (
            i
            for i, task in enumerate(tasks)
            if (
                (task["document_id"], task["page_id"], task.get("field_name")) not in field_done
                if task["task_type"] == "FIELD_AND_CROP"
                else (task["document_id"], task["page_id"]) not in line_done
            )
        ),
        0,
    )
    return f"""<!doctype html><meta charset=utf-8><title>Tuning Truth V1</title>
<style>body{{font:15px Arial;margin:30px;max-width:900px}}.meter{{height:20px;background:#ddd}}
.meter div{{height:100%;background:#1769aa;width:{100 * complete / max(1, len(tasks)):.2f}%}}</style>
<h1>TUNING_TRUTH_V1 annotation</h1><p>Reviewer: <b>{reviewer}</b></p>
<p>{complete} / {len(tasks)} tasks verified. Suggestions are never truth until submitted.</p>
<div class=meter><div></div></div><p><a href='/task/{first}'>Continue annotation</a></p>"""


@app.get("/image/{index}")
def image(index: int, request: Request):
    _reviewer(request)
    task = _task(index)
    path = (ROOT / task["image_path"]).resolve()
    # Only exact image paths already frozen into the tuning-only task queue are served.
    if not path.is_file() or path.as_posix() != (ROOT / task["image_path"]).resolve().as_posix():
        raise HTTPException(404, "selected tuning image unavailable")
    return FileResponse(path)


@app.get("/task/{index}", response_class=HTMLResponse)
def task_screen(index: int, request: Request) -> str:
    reviewer = html.escape(_reviewer(request))
    task = _task(index)
    previous = max(0, index - 1)
    following = min(len(_tasks()) - 1, index + 1)
    title = task.get("field_name", "UB service lines")
    if task["task_type"] == "UB_SERVICE_LINES":
        controls = """<fieldset><legend>Draw row, enter values, then add row</legend>
<input id=x1 type=number min=0 max=1 step=.0001 placeholder=x1><input id=y1 type=number min=0 max=1 step=.0001 placeholder=y1>
<input id=x2 type=number min=0 max=1 step=.0001 placeholder=x2><input id=y2 type=number min=0 max=1 step=.0001 placeholder=y2><br>
<input id=row_index type=number min=1 placeholder='row index'><input id=revenue_code placeholder='revenue code'>
<input id=description placeholder=description><input id=hcpcs placeholder=HCPCS><input id=service_date placeholder='service date'>
<input id=units placeholder=units><input id=charge placeholder=charge>
<button type=button onclick=addRow()>Add row</button></fieldset>
<label>Rows JSON <textarea id=rows_json name=rows_json rows=12 cols=100>[]</textarea></label>
<label>Total charge <input name=total_charge></label>"""
    else:
        statuses = "".join(f"<option>{status.value}</option>" for status in TruthStatus)
        controls = f"""<label>Status <select name=truth_status>{statuses}</select></label>
<label>Truth value <input name=truth_value size=60></label>
<label>Visibility <select name=visibility>{"".join(f"<option>{v.value}</option>" for v in Visibility if v != Visibility.UNANNOTATED)}</select></label>
<fieldset><legend>Normalized value box (draw on image or type)</legend>
<input id=x1 name=x1 type=number min=0 max=1 step=.0001 placeholder=x1>
<input id=y1 name=y1 type=number min=0 max=1 step=.0001 placeholder=y1>
<input id=x2 name=x2 type=number min=0 max=1 step=.0001 placeholder=x2>
<input id=y2 name=y2 type=number min=0 max=1 step=.0001 placeholder=y2></fieldset>
<label><input type=checkbox name=multi_line value=true> Multi-line value</label>"""
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:14px Arial;margin:20px}}#wrap{{position:relative;display:inline-block;max-width:95vw}}
#page{{max-width:95vw;max-height:68vh}}#box{{position:absolute;border:3px solid #e22;pointer-events:none}}
label{{display:block;margin:9px}}input,select,textarea{{padding:5px}}</style>
<p><a href='/task/{previous}'>&larr; Previous</a> | <a href='/'>Queue</a> | <a href='/task/{following}'>Next &rarr;</a></p>
<h2>{html.escape(task["form_family"])}: {html.escape(title)}</h2>
<p>{html.escape(task["document_id"])} | {html.escape(task["source_dataset"])} | {html.escape(task["quality_bucket"])} | reviewer {reviewer}</p>
<div id=wrap><img id=page src='/image/{index}'><div id=box hidden></div></div>
<form method=post action='/task/{index}'>{controls}
<label>Annotation confidence <input name=confidence type=number min=0 max=1 step=.05 value=1></label>
<label><input name=visual_verified type=checkbox value=true required> I visually verified this annotation against the page.</label>
<button>Save verified annotation and continue</button></form>
<script>
const img=document.querySelector('#page'), box=document.querySelector('#box'); let start=null;
img.onpointerdown=e=>{{const r=img.getBoundingClientRect();start=[e.clientX-r.left,e.clientY-r.top];e.preventDefault()}};
img.onpointerup=e=>{{if(!start)return;const r=img.getBoundingClientRect(),end=[e.clientX-r.left,e.clientY-r.top];
const a=[Math.min(start[0],end[0]),Math.min(start[1],end[1])],b=[Math.max(start[0],end[0]),Math.max(start[1],end[1])];
for(const [id,v] of [['x1',a[0]/r.width],['y1',a[1]/r.height],['x2',b[0]/r.width],['y2',b[1]/r.height]]){{const el=document.querySelector('#'+id);if(el)el.value=v.toFixed(4)}};
Object.assign(box.style,{{left:a[0]+'px',top:a[1]+'px',width:(b[0]-a[0])+'px',height:(b[1]-a[1])+'px'}});box.hidden=false;start=null}};
function addRow(){{const area=document.querySelector('#rows_json');if(!area)return;const rows=JSON.parse(area.value||'[]');
const v=id=>document.querySelector('#'+id).value;rows.push({{row_index:Number(v('row_index')),bbox:{{x1:Number(v('x1')),y1:Number(v('y1')),x2:Number(v('x2')),y2:Number(v('y2'))}},revenue_code:v('revenue_code'),description:v('description'),hcpcs:v('hcpcs'),service_date:v('service_date'),units:v('units'),charge:v('charge')}});area.value=JSON.stringify(rows,null,2)}}
</script>"""


@app.post("/task/{index}")
def submit(
    index: int,
    request: Request,
    visual_verified: bool = Form(False),
    truth_status: str = Form("UNSUPPORTED"),
    truth_value: str = Form(""),
    visibility: str = Form("UNANNOTATED"),
    x1: float | None = Form(None),
    y1: float | None = Form(None),
    x2: float | None = Form(None),
    y2: float | None = Form(None),
    multi_line: bool = Form(False),
    confidence: float = Form(1),
    rows_json: str = Form("[]"),
    total_charge: str = Form(""),
):
    reviewer = _reviewer(request)
    if not visual_verified:
        raise HTTPException(400, "visual verification is required")
    task = _task(index)
    if task["task_type"] == "UB_SERVICE_LINES":
        try:
            rows = json.loads(rows_json)
            record = UB04ServiceLineTruth(
                document_id=task["document_id"],
                page_id=task["page_id"],
                rows=rows,
                expected_row_count=len(rows),
                total_charge=total_charge,
                review_status=ReviewStatus.VERIFIED,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = record.model_dump(mode="json") | {"reviewer_id": reviewer}
        _upsert(UB_LINES, payload, ("document_id", "page_id"))
    else:
        status = TruthStatus(truth_status)
        has_box = None not in (x1, y1, x2, y2)
        if status == TruthStatus.PRESENT and (not truth_value.strip() or not has_box):
            raise HTTPException(400, "PRESENT values require truth text and a value bbox")
        required, criticality, blocks_stp = field_policy(task["field_name"])
        field = FieldTruth(
            document_id=task["document_id"],
            page_id=task["page_id"],
            form_family=task["form_family"],
            field_name=task["field_name"],
            truth_value=truth_value.strip(),
            normalized_truth_value=_normalized(truth_value),
            required=required,
            criticality=criticality,
            blocks_stp=blocks_stp,
            truth_status=status,
            review_status=ReviewStatus.VERIFIED,
            preannotation_source=None,
        )
        crop = FieldCropTruth(
            document_id=task["document_id"],
            page_id=task["page_id"],
            form_family=task["form_family"],
            field_name=task["field_name"],
            value_bbox=NormalizedBBox(x1=x1, y1=y1, x2=x2, y2=y2) if has_box else None,
            expected_text=truth_value.strip(),
            visibility=Visibility(visibility),
            multi_line=multi_line,
            annotation_confidence=confidence,
            review_status=ReviewStatus.VERIFIED,
        )
        key = ("document_id", "page_id", "field_name")
        _upsert(FIELDS, field.model_dump(mode="json") | {"reviewer_id": reviewer}, key)
        _upsert(CROPS, crop.model_dump(mode="json") | {"reviewer_id": reviewer}, key)
    _refresh_quality()
    return RedirectResponse(f"/task/{min(index + 1, len(_tasks()) - 1)}", status_code=303)
