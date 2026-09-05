"""FastAPI server to host the Claims Intelligence Governance Console (evaluation_ui)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

DIST_DIR = Path(__file__).resolve().parent / "dist"
TRANSFORMATION_DIR = Path(__file__).resolve().parent.parent / "transformation_ui"

INGESTION_API_URL = os.getenv("INGESTION_API_URL", "http://localhost:8000")
HUMAN_REVIEW_API_URL = os.getenv("HUMAN_REVIEW_API_URL", "http://localhost:8100")

app = FastAPI(title="IDP Evaluation & Governance Console UI", version="1.0.0")

client = httpx.AsyncClient(timeout=120.0)


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_api(request: Request, path: str):
    url = f"{INGESTION_API_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("transfer-encoding", None)
    
    body = await request.body()
    
    try:
        req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=body
        )
        resp = await client.send(req, stream=True)
        
        # Clean response headers
        resp_headers = dict(resp.headers)
        resp_headers.pop("content-length", None)
        resp_headers.pop("content-encoding", None)
        
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=resp_headers
        )
    except Exception as exc:
        return StreamingResponse(
            iter([f'{{"error": "Ingestion API proxy error: {str(exc)}"}}'.encode()]),
            status_code=502,
            media_type="application/json"
        )


@app.api_route("/review-api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_review_api(request: Request, path: str):
    url = f"{HUMAN_REVIEW_API_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
        
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("transfer-encoding", None)
    
    body = await request.body()
    
    try:
        req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=body
        )
        resp = await client.send(req, stream=True)
        
        resp_headers = dict(resp.headers)
        resp_headers.pop("content-length", None)
        resp_headers.pop("content-encoding", None)
        
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=resp_headers
        )
    except Exception as exc:
        return StreamingResponse(
            iter([f'{{"error": "Human Review API proxy error: {str(exc)}"}}'.encode()]),
            status_code=502,
            media_type="application/json"
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

if (DIST_DIR / "reports").exists():
    app.mount("/reports", StaticFiles(directory=DIST_DIR / "reports"), name="reports")

if TRANSFORMATION_DIR.exists():
    app.mount("/transformation-files", StaticFiles(directory=TRANSFORMATION_DIR), name="transformation_files")


@app.get("/transformation")
def read_transformation():
    return FileResponse(TRANSFORMATION_DIR / "index.html")


@app.get("/styles.css")
def read_transformation_css():
    return FileResponse(TRANSFORMATION_DIR / "styles.css")


@app.get("/app.js")
def read_transformation_js():
    return FileResponse(TRANSFORMATION_DIR / "app.js")


@app.get("/transformation-files/styles.css")
def read_transformation_css_files():
    return FileResponse(TRANSFORMATION_DIR / "styles.css")


@app.get("/transformation-files/app.js")
def read_transformation_js_files():
    return FileResponse(TRANSFORMATION_DIR / "app.js")


@app.get("/reports/evaluation.json")
def get_evaluation_report():
    report_file = DIST_DIR / "reports" / "evaluation.json"
    if report_file.exists():
        return FileResponse(report_file)
    return {"error": "Report file not found"}


@app.get("/")
def read_root():
    index = DIST_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>IDP Claims Evaluation UI</title></head>
  <body>
    <main>
      <h1>IDP Claims Evaluation UI</h1>
      <p>Frontend assets are not built. Run <code>npm run build</code> in
      <code>apps/evaluation_ui</code>.</p>
    </main>
  </body>
</html>
""",
        status_code=200,
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8180)
