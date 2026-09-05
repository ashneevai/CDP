"""Resumable, PHI-safe page classification and document-boundary candidates.

OCR text exists only in memory long enough to run ``MultiSignalRouter``.  The
checkpoint and result artifacts contain hashes, numeric scores, canonical
anchor names, and reason codes only.  Every result is a review candidate; this
module never creates ground truth or confirms a document boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from packages.document_routing import MultiSignalRoute, MultiSignalRouter
from packages.domain.common import BoundingBox
from packages.domain.enums import ClaimFormType
from packages.ocr import OCRExecutionService, OCRRequest, RapidOCRProvider
from packages.templates.registry import TemplateRegistry
from workers.page_detection.grid_signature import compute_grid_signature, signature_similarity
from workers.page_detection.router import GRID_AMBIGUITY_MARGIN, GRID_CONFIDENT_THRESHOLD
from workers.page_detection.text_extraction import TextLine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "evaluation_data/source_b_1000_claims"
DEFAULT_OUTPUT = ROOT / "evaluation_data/real_eval_runtime"
RUNNER_VERSION = "real-archive-candidates-v1"


class ResumeMismatchError(RuntimeError):
    """Raised rather than mixing checkpoints from different source/config runs."""


class ObservationProvider(Protocol):
    async def __call__(self, image: Image.Image, page: PageRef) -> Observation: ...


@dataclass(frozen=True)
class PageRef:
    archive_id: str
    package_id: str
    asset_id: str
    page_id: str
    page_number: int
    asset_page_count: int
    asset_sequence: int | None
    asset_path: Path
    asset_sha256: str
    page_sha256: str


@dataclass(frozen=True)
class Observation:
    lines: tuple[TextLine, ...]
    latency_ms: float
    engine: str
    engine_version: str
    cache_hit: bool = False


def _digest(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pages(source: Path, archive_sha256: str) -> Iterable[tuple[PageRef, Image.Image]]:
    """Yield decoded pages in deterministic package/asset/frame order."""
    archive_id = f"archive_sha256_{archive_sha256}"
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        asset_sha = _file_digest(path)
        with Image.open(path) as image:
            source_package = str(image.tag_v2.get(45016, "")).strip() or path.stem
            package_id = f"package_{_digest(archive_id, source_package)[:24]}"
            asset_id = f"asset_{_digest(archive_id, asset_sha)[:24]}"
            page_count = int(getattr(image, "n_frames", 1))
            suffix = path.suffix[1:]
            sequence = int(suffix) if suffix.isdigit() else None
            for index in range(page_count):
                image.seek(index)
                page = image.copy()
                page_digest = hashlib.sha256()
                page_digest.update(f"{page.mode}|{page.width}|{page.height}|".encode())
                page_digest.update(page.tobytes())
                page_sha = page_digest.hexdigest()
                page_id = f"page_{_digest(asset_id, index + 1, page_sha)[:24]}"
                yield (
                    PageRef(
                        archive_id,
                        package_id,
                        asset_id,
                        page_id,
                        index + 1,
                        page_count,
                        sequence,
                        path,
                        asset_sha,
                        page_sha,
                    ),
                    page,
                )


class RapidOCRPageObserver:
    """Production OCR adapter; recognized values are never returned as artifacts."""

    def __init__(self, provider: RapidOCRProvider | None = None) -> None:
        self.provider = provider or RapidOCRProvider()
        self.execution = OCRExecutionService(benchmark_mode=True)

    async def __call__(self, image: Image.Image, page: PageRef) -> Observation:
        box = BoundingBox(
            x0=0,
            y0=0,
            x1=image.width,
            y1=image.height,
            image_width=image.width,
            image_height=image.height,
        )
        request = OCRRequest(
            document_id=page.asset_id,
            page_number=page.page_number,
            field_name="__classification_observation__",
            field_type="routing_evidence",
            form_type=ClaimFormType.UNSTRUCTURED,
            image=image,
            bounding_box=box,
            scope="FULL_PAGE",
            policy_allows_full_page=True,
            document_sha256=page.asset_sha256,
            page_sha256=page.page_sha256,
            source_representation_id=page.asset_id,
            preprocessing_profile=None,
        )
        result = await self.execution.execute(self.provider, request)
        tokens = result.candidates[0].tokens if result.candidates else ()
        lines = tuple(
            TextLine(
                token.text,
                token.bounding_box.x0,
                token.bounding_box.y0,
                token.bounding_box.x1,
                token.bounding_box.y1,
                token.confidence,
            )
            for token in tokens
        )
        return Observation(
            lines, result.latency_ms, result.provider, result.provider_version, result.cache_hit
        )


def _candidate_class(route: MultiSignalRoute) -> str:
    return {
        MultiSignalRoute.CMS1500: "CMS1500",
        MultiSignalRoute.UB04: "UB04",
        MultiSignalRoute.OTHER_CLAIM_FORM: "OTHER_CLAIM_FORM",
        MultiSignalRoute.NON_CLAIM: "NON_CLAIM",
        MultiSignalRoute.UNKNOWN_STRUCTURED: "SUPPORTING_DOCUMENT",
        MultiSignalRoute.UNKNOWN_UNSTRUCTURED: "UNKNOWN",
    }[route]


def _confidence_band(route: MultiSignalRoute, evidence: Any) -> str:
    # Existing eligibility/reason gates are reused; these are review-routing
    # states, explicitly not calibrated accuracy thresholds.
    if route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
        return (
            "HIGH_CONFIDENCE_CANDIDATE" if evidence.eligibility[route.value] else "REVIEW_REQUIRED"
        )
    if (
        route is MultiSignalRoute.NON_CLAIM
        and "MULTIPLE_NEGATIVE_ANCHORS_LOW_HEALTHCARE_DENSITY" in evidence.reason_codes
    ):
        return "HIGH_CONFIDENCE_CANDIDATE"
    if route is MultiSignalRoute.UNKNOWN_UNSTRUCTURED:
        return "UNKNOWN"
    return "REVIEW_REQUIRED"


def _safe_record(
    page: PageRef,
    image: Image.Image,
    observation: Observation,
    router: MultiSignalRouter,
    routing_evidence: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence = routing_evidence or router.route(image, list(observation.lines))
    routing_ms = (time.perf_counter() - started) * 1000
    matched = {
        family: sorted(anchors) for family, anchors in evidence.matched_anchors.items() if anchors
    }
    return {
        "schema_version": "1.0",
        "candidate_only": True,
        "trusted_ground_truth": False,
        "archive_id": page.archive_id,
        "package_id": page.package_id,
        "source_asset_id": page.asset_id,
        "source_page_id": page.page_id,
        "source_page_number": page.page_number,
        "source_asset_page_count": page.asset_page_count,
        "source_asset_sequence": page.asset_sequence,
        "source_page_sha256": page.page_sha256,
        "candidate_class": _candidate_class(evidence.route),
        "classification_confidence": evidence.confidence,
        "confidence_band": _confidence_band(evidence.route, evidence),
        "template_registration": {
            "state": "NOT_EXECUTED",
            "candidate_only": True,
        },
        "anchor_evidence": {
            "matched_canonical_anchors": matched,
            "exact_count": evidence.exact_anchor_count,
            "normalized_count": evidence.normalized_anchor_count,
            "fuzzy_count": evidence.fuzzy_anchor_count,
            "high_value_count": evidence.high_value_anchor_count,
            "weighted_coverage": evidence.weighted_anchor_coverage,
            "geometry_score": evidence.anchor_geometry_score,
        },
        "layout_evidence": evidence.standard_structure,
        "grid_line_evidence": {
            "grid_score": evidence.grid_score,
            "horizontal_line_score": evidence.horizontal_line_score,
            "vertical_line_score": evidence.vertical_line_score,
        },
        "text_evidence": {
            "token_count": len(observation.lines),
            "mean_token_confidence": (
                sum(line.confidence for line in observation.lines) / len(observation.lines)
                if observation.lines
                else 0.0
            ),
            "recognized_text_persisted": False,
        },
        "reason_codes": evidence.reason_codes,
        "scores": evidence.scores,
        "router_version": evidence.router_version,
        "form_identity": {
            "identity_state": evidence.identity_state,
            "field_topology_score": evidence.field_topology_score,
            "missing_required_anchors": evidence.missing_required_anchors,
            "conflicting_anchors": evidence.conflicting_anchors,
            "localization_allowed": evidence.localization_allowed,
            "identity_policy_version": evidence.identity_policy_version,
            "family_eligibility": evidence.family_eligibility,
            "identity_anchor_evidence": [
                item.model_dump(mode="json") for item in evidence.identity_anchor_evidence
            ],
        },
        "ocr": {
            "engine": observation.engine,
            "engine_version": observation.engine_version,
            "latency_ms": observation.latency_ms,
            "cache_hit": observation.cache_hit,
            "raw_value_persisted": False,
        },
        "routing_latency_ms": routing_ms,
        "review_state": "UNREVIEWED",
    }


def load_visual_references() -> dict[str, tuple[str, Image.Image]]:
    registry = TemplateRegistry.load_from_directory()
    references = {}
    for family, form_type in (("CMS1500", ClaimFormType.CMS1500), ("UB04", ClaimFormType.UB04)):
        template = registry.latest_for_form_type(form_type)
        image = registry.load_reference_image(template)
        if image is None:
            filename = "cms1500_v02_12.png" if family == "CMS1500" else "ub04_v2014.png"
            fallback = ROOT / "config" / "templates" / "reference_images" / filename
            if fallback.is_file():
                with Image.open(fallback) as source:
                    image = source.convert("L").copy()
        if image is not None:
            references[family] = (f"{template.template_id}@{template.version}", image)
    return references


def _visual_record(
    page: PageRef, image: Image.Image, references: dict[str, tuple[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    signature = compute_grid_signature(image)
    scores = {
        family: signature_similarity(signature, reference)
        for family, (_, reference) in references.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_family, best_score = ranked[0] if ranked else ("UNKNOWN", 0.0)
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    eligible = (
        len(scores) == 2
        and best_family in {"CMS1500", "UB04"}
        and best_score >= GRID_CONFIDENT_THRESHOLD
        and margin >= GRID_AMBIGUITY_MARGIN
    )
    candidate_class = best_family if eligible else "UNKNOWN"
    reasons = (
        ["GRID_LAYOUT_SIGNATURE_MATCH", "EXISTING_GRID_THRESHOLDS_SATISFIED"]
        if eligible
        else ["VISUAL_EVIDENCE_INSUFFICIENT", "OCR_ESCALATION_OR_REVIEW_REQUIRED"]
    )
    return {
        "schema_version": "1.0",
        "candidate_only": True,
        "trusted_ground_truth": False,
        "archive_id": page.archive_id,
        "package_id": page.package_id,
        "source_asset_id": page.asset_id,
        "source_page_id": page.page_id,
        "source_page_number": page.page_number,
        "source_asset_page_count": page.asset_page_count,
        "source_asset_sequence": page.asset_sequence,
        "source_page_sha256": page.page_sha256,
        "candidate_class": candidate_class,
        "classification_confidence": best_score,
        "confidence_band": "HIGH_CONFIDENCE_CANDIDATE" if eligible else "REVIEW_REQUIRED",
        "template_registration": {
            "state": "GRID_SIGNATURE_CANDIDATE_ONLY",
            "candidate_only": True,
            "reference_templates": {
                family: template_id for family, (template_id, _) in references.items()
            },
        },
        "anchor_evidence": {"state": "NOT_EXECUTED"},
        "layout_evidence": {
            "method": "GRID_LAYOUT_SIGNATURE",
            "template_similarity": scores,
            "best_family": best_family,
            "best_score": best_score,
            "runner_up_score": runner_up,
            "margin": margin,
            "threshold": GRID_CONFIDENT_THRESHOLD,
            "ambiguity_margin": GRID_AMBIGUITY_MARGIN,
            "threshold_source": "workers.page_detection.router",
        },
        "grid_line_evidence": {
            "signature_bins": len(signature.row_profile),
            "row_profile_nonzero_bins": int((signature.row_profile > 0).sum()),
            "column_profile_nonzero_bins": int((signature.col_profile > 0).sum()),
            "raw_profiles_persisted": False,
        },
        "text_evidence": {"state": "NOT_EXECUTED", "recognized_text_persisted": False},
        "reason_codes": reasons,
        "scores": scores,
        "visual_latency_ms": (time.perf_counter() - started) * 1000,
        "review_state": "UNREVIEWED",
    }


def _header(source_fingerprint: str, router: MultiSignalRouter, mode: str) -> dict[str, Any]:
    return {
        "record_type": "CHECKPOINT_HEADER",
        "runner_version": RUNNER_VERSION,
        "source_fingerprint": source_fingerprint,
        "mode": mode,
        "router_version": router.config.get("router_version", "unknown"),
        "phi_safe": True,
    }


def _load_checkpoint(path: Path, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    if not rows or rows[0] != expected:
        raise ResumeMismatchError("checkpoint source, runner, or router does not match")
    records = rows[1:]
    return records, {record["source_page_id"] for record in records}


def build_document_boundaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create reviewable starts from asset edges and class transitions only."""
    ordered = sorted(
        records,
        key=lambda row: (
            row["package_id"],
            row["source_asset_sequence"] or 0,
            row["source_asset_id"],
            row["source_page_number"],
        ),
    )
    candidates = []
    previous: dict[str, Any] | None = None
    for row in ordered:
        reasons = []
        if previous is None or previous["package_id"] != row["package_id"]:
            reasons.append("PACKAGE_START")
        if previous is None or previous["source_asset_id"] != row["source_asset_id"]:
            reasons.append("TIFF_ASSET_START")
        if (
            previous
            and previous["package_id"] == row["package_id"]
            and previous["candidate_class"] != row["candidate_class"]
        ):
            reasons.append("FORM_CLASS_TRANSITION")
        if row["candidate_class"] == "NON_CLAIM":
            reasons.append("SEPARATOR_PAGE_CANDIDATE")
        if reasons:
            candidates.append(
                {
                    "boundary_candidate_id": f"boundary_{_digest(row['source_page_id'], *reasons)[:24]}",
                    "package_id": row["package_id"],
                    "source_asset_id": row["source_asset_id"],
                    "source_page_id": row["source_page_id"],
                    "boundary_state": "CANDIDATE",
                    "candidate_document_start": True,
                    "confirmed": False,
                    "reason_codes": reasons,
                    "review_state": "UNREVIEWED",
                }
            )
        previous = row
    return candidates


async def run(
    source: Path,
    output: Path,
    archive_sha256: str,
    observer: ObservationProvider | None = None,
    *,
    limit: int | None = None,
    mode: str = "ocr",
    visual_references: dict[str, tuple[str, Image.Image]] | None = None,
) -> dict[str, Any]:
    """Run or resume classification, then materialize deterministic JSON artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    router = MultiSignalRouter.load()
    if mode not in {"ocr", "visual"}:
        raise ValueError("mode must be 'ocr' or 'visual'")
    source_fingerprint = _digest(archive_sha256, RUNNER_VERSION, mode)
    header = _header(source_fingerprint, router, mode)
    checkpoint = output / f".page_classification_{mode}_checkpoint.jsonl"
    records, completed = _load_checkpoint(checkpoint, header)
    if not checkpoint.exists():
        checkpoint.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")
    observe = (observer or RapidOCRPageObserver()) if mode == "ocr" else observer
    reference_images = (
        visual_references if visual_references is not None else load_visual_references()
    )
    references = {
        family: (template_id, compute_grid_signature(image))
        for family, (template_id, image) in reference_images.items()
    }
    processed_now = 0
    discovered_pages = 0
    with checkpoint.open("a", encoding="utf-8") as stream:
        for page, image in discover_pages(source, archive_sha256):
            discovered_pages += 1
            if page.page_id in completed:
                continue
            try:
                if mode == "visual":
                    record = _visual_record(page, image, references)
                else:
                    if observe is None:
                        raise RuntimeError("OCR observer is unavailable")
                    observation = await observe(image, page)
                    record = _safe_record(page, image, observation, router)
            except Exception as error:  # noqa: BLE001 - each page must fail closed and resume
                record = {
                    "schema_version": "1.0",
                    "candidate_only": True,
                    "trusted_ground_truth": False,
                    "archive_id": page.archive_id,
                    "package_id": page.package_id,
                    "source_asset_id": page.asset_id,
                    "source_page_id": page.page_id,
                    "source_page_number": page.page_number,
                    "source_asset_page_count": page.asset_page_count,
                    "source_asset_sequence": page.asset_sequence,
                    "source_page_sha256": page.page_sha256,
                    "candidate_class": "UNKNOWN",
                    "classification_confidence": 0.0,
                    "confidence_band": "UNKNOWN",
                    "reason_codes": ["OBSERVATION_FAILED"],
                    "failure": {"error_type": type(error).__name__, "message_persisted": False},
                    "review_state": "UNREVIEWED",
                }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            records.append(record)
            completed.add(page.page_id)
            processed_now += 1
            if limit is not None and processed_now >= limit:
                break
    records.sort(key=lambda row: row["source_page_id"])
    classifications = {
        "schema_version": "1.0",
        "runner_version": RUNNER_VERSION,
        "candidate_only": True,
        "trusted_ground_truth": False,
        "source_fingerprint": source_fingerprint,
        "mode": mode,
        "records": records,
        "counts": dict(Counter(row["candidate_class"] for row in records)),
        "processed_pages": len(records),
        "complete": limit is None and len(records) == discovered_pages,
        "recognized_text_persisted": False,
    }
    boundaries = {
        "schema_version": "1.0",
        "candidate_only": True,
        "complete": limit is None and len(records) == discovered_pages,
        "confirmed_document_count": 0,
        "package_count": len({row["package_id"] for row in records}),
        "records": build_document_boundaries(records),
    }
    (output / "page_classification_candidates.json").write_text(
        json.dumps(classifications, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "document_boundary_candidates.json").write_text(
        json.dumps(boundaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"classification": classifications, "boundaries": boundaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("visual", "ocr"), default="ocr")
    args = parser.parse_args()
    asyncio.run(
        run(args.source, args.output, args.archive_sha256, limit=args.limit, mode=args.mode)
    )


if __name__ == "__main__":
    main()
