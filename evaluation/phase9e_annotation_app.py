"""Private, prediction-blind Phase 9E annotation UI.

Set ``CDP_ACCURACY_PRIVATE_DIR`` to a directory containing
``private_sample_manifest.jsonl``. The app never imports prediction or OCR
modules and writes one isolated file per annotator role.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from evaluation.phase9e_accuracy import DOCUMENT_TYPES, TRUTH_STATUSES, _jsonl


PRIVATE = Path(os.environ.get("CDP_ACCURACY_PRIVATE_DIR", ".private/phase9e")).resolve()
MANIFEST = PRIVATE / "private_sample_manifest.jsonl"
CANONICAL_FIELDS = (
    "member_id", "patient_name", "patient_dob", "insured_name", "provider_name",
    "provider_npi", "federal_tax_id", "payer_id", "account_number", "diagnosis_codes",
    "procedure_codes", "service_dates", "total_charge",
)
app = FastAPI(title="CDP Accuracy Qualification V1", docs_url=None, redoc_url=None)


def _identity(role: str | None, annotator: str | None) -> tuple[str, str]:
    role = role or os.environ.get("CDP_ANNOTATION_ROLE")
    annotator = annotator or os.environ.get("CDP_ANNOTATOR_ID")
    if role not in {"A", "B", "ADJUDICATOR"} or not annotator:
        raise HTTPException(
            401,
            "Set CDP_ANNOTATION_ROLE and CDP_ANNOTATOR_ID, or supply the corresponding X- headers",
        )
    return role, annotator


def _rows() -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise HTTPException(503, "private sample manifest unavailable")
    return _jsonl(MANIFEST)


def _output(role: str) -> Path:
    return PRIVATE / f"annotations_{role.lower()}.jsonl"


def _saved(role: str) -> dict[str, dict[str, Any]]:
    path = _output(role)
    return {row["document_id"]: row for row in _jsonl(path)} if path.is_file() else {}


def _write(role: str, row: dict[str, Any]) -> None:
    rows = _saved(role)
    rows[row["document_id"]] = row
    path = _output(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in rows.values()), "utf-8")


@app.get("/", response_class=HTMLResponse)
def queue(x_annotation_role: str | None = Header(None), x_annotator_id: str | None = Header(None)) -> str:
    role, annotator = _identity(x_annotation_role, x_annotator_id)
    rows, done = _rows(), _saved(role)
    first = next((index for index, row in enumerate(rows) if row["document_id"] not in done), 0)
    return (f"<!doctype html><meta charset=utf-8><title>CDP Accuracy V1</title>"
            f"<h1>Blind annotation — role {html.escape(role)}</h1>"
            f"<p>Annotator {html.escape(annotator)} · {len(done)}/{len(rows)} complete</p>"
            f"<p>Predictions, OCR, routes, confidence, and dispositions are intentionally unavailable.</p>"
            f"<a href='/document/{first}'>Resume</a>")


@app.get("/image/{index}")
def image(index: int, x_annotation_role: str | None = Header(None), x_annotator_id: str | None = Header(None)):
    _identity(x_annotation_role, x_annotator_id)
    row = _rows()[index]
    path = Path(row["path"]).resolve()
    if not path.is_file():
        raise HTTPException(404, "image unavailable")
    return FileResponse(path)


@app.get("/document/{index}", response_class=HTMLResponse)
def document(index: int, x_annotation_role: str | None = Header(None), x_annotator_id: str | None = Header(None)) -> str:
    role, annotator = _identity(x_annotation_role, x_annotator_id)
    rows = _rows()
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "document unavailable")
    row = rows[index]
    fields = "".join(
        f"<tr><td>{html.escape(name)}</td><td><select name='{name}__status'>"
        + "".join(f"<option>{status}</option>" for status in sorted(TRUTH_STATUSES))
        + f"</select></td><td><input name='{name}__value' autocomplete=off></td></tr>"
        for name in CANONICAL_FIELDS
    )
    types = "".join(f"<option>{value}</option>" for value in sorted(DOCUMENT_TYPES))
    previous, following = max(0, index - 1), min(len(rows) - 1, index + 1)
    return f"""<!doctype html><meta charset=utf-8><title>Blind annotation</title>
<style>body{{font:14px Arial;margin:15px}}img{{max-width:96vw;max-height:62vh;transform:rotate(var(--r,0deg));transform-origin:center}}table{{border-collapse:collapse}}td{{padding:4px;border:1px solid #bbb}}input{{width:36em}}</style>
<p><a href='/document/{previous}'>Previous</a> · <a href='/'>Progress</a> · <a href='/document/{following}'>Next</a>
<button onclick="z+=.15;page.style.zoom=z">Zoom +</button><button onclick="z=Math.max(.25,z-.15);page.style.zoom=z">Zoom −</button><button onclick="r+=90;page.style.setProperty('--r',r+'deg')">Rotate</button></p>
<p>Role {role}; annotator {html.escape(annotator)}; document {index + 1}/{len(rows)}; package {html.escape(row['package_id'])}</p>
<img id=page src='/image/{index}'><form method=post action='/document/{index}'>
<label>Document type <select name=document_type>{types}</select></label>
<table><tr><th>Field</th><th>Status</th><th>Independent truth value</th></tr>{fields}</table>
<label>Notes <textarea name=notes></textarea></label><button>Save and continue</button></form>
<script>let z=1,r=0</script>"""


@app.post("/document/{index}")
async def submit(index: int, request: Request,
                 x_annotation_role: str | None = Header(None), x_annotator_id: str | None = Header(None),
                 ):
    role, annotator = _identity(x_annotation_role, x_annotator_id)
    if role == "ADJUDICATOR":
        raise HTTPException(403, "adjudication uses the separate disagreement queue")
    form = await request.form()
    document_type = str(form.get("document_type") or "")
    notes = str(form.get("notes") or "")
    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(400, "invalid document type")
    source = _rows()[index]
    fields = {name: {"status": form.get(f"{name}__status"), "value": form.get(f"{name}__value", "")}
              for name in CANONICAL_FIELDS}
    _write(role, {"document_id": source["document_id"], "package_id": source["package_id"],
                  "document_type": document_type, "fields": fields, "annotator_id": annotator,
                  "notes": notes})
    return RedirectResponse(f"/document/{min(index + 1, len(_rows()) - 1)}", status_code=303)
