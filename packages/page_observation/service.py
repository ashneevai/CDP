from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .cache import PageObservationCache
from .contracts import (
    ImageQualityEvidence,
    ObservationToken,
    PageObservation,
    StructuralLine,
    StructuralRegion,
)
from .reading_order import line_clustered_reading_order


class OCRLine(Protocol):
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


class FullPageExtractor(Protocol):
    model_version: str

    def extract(self, image: Image.Image) -> list[OCRLine]: ...


def _group_positions(mask: np.ndarray, axis: int, threshold: float) -> list[int]:
    indices = np.flatnonzero((mask > 0).mean(axis=axis) >= threshold)
    groups: list[list[int]] = []
    for value in indices.tolist():
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def _skew_degrees(binary: np.ndarray) -> float:
    points = np.column_stack(np.where(binary > 0))
    if len(points) < 20:
        return 0.0
    angle = float(cv2.minAreaRect(points[:, ::-1].astype(np.float32))[-1])
    if angle > 45:
        angle -= 90
    return max(-15.0, min(15.0, angle))


def _writing_signal(components: tuple[tuple[int, int, int, int], ...]) -> tuple[str, float | None]:
    if len(components) < 5:
        return "UNKNOWN", None
    heights = np.asarray([box[3] - box[1] for box in components], dtype=np.float32)
    widths = np.asarray([box[2] - box[0] for box in components], dtype=np.float32)
    variation = min(1.0, float(heights.std() / max(1.0, heights.mean())))
    wide_ratio = float(np.mean(widths > heights * 3.0))
    likelihood = max(0.0, min(1.0, .65 * variation + .35 * wide_ratio))
    kind = "HANDWRITTEN" if likelihood >= .65 else "PRINTED" if likelihood <= .35 else "MIXED"
    return kind, likelihood


class PageObservationService:
    version = "page-observation-service-v1"

    def __init__(self, extractor: FullPageExtractor, *, preprocessing_version: str,
                 cache: PageObservationCache | None = None,
                 benchmark_mode: bool | None = None):
        self._extractor = extractor
        self._preprocessing_version = preprocessing_version
        self._cache = cache or PageObservationCache()
        self._benchmark_mode = (
            os.getenv("BENCHMARK_MODE", "").strip().casefold() in {"1", "true", "yes", "on"}
            if benchmark_mode is None
            else benchmark_mode
        )

    def observe(self, page_id: str, image: Image.Image, *, page_sha256: str | None = None,
                source_channel: str = "UNKNOWN", resolution_dpi: float | None = None):
        rgb = image.convert("RGB")
        digest = page_sha256 or hashlib.sha256(rgb.tobytes()).hexdigest()
        model_version = getattr(self._extractor, "model_version", "unknown")
        quality_context = f"{self._preprocessing_version}:{source_channel}:{resolution_dpi}"
        key = self._cache.key(digest, model_version, quality_context)
        if not self._benchmark_mode and (cached := self._cache.get(key)):
            return cached
        lines = self._extractor.extract(rgb)
        tokens = tuple(
            ObservationToken(
                token_id=f"ocr-{index}", text=line.text.strip(),
                bbox=(line.x0, line.y0, line.x1, line.y1), confidence=line.confidence,
                line_index=index, reading_order=index,
            )
            for index, line in enumerate(line_clustered_reading_order(lines))
            if line.text.strip()
        )
        gray = np.asarray(rgb.convert("L"))
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, rgb.width // 30), 1)))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, rgb.height // 40))))
        ys = _group_positions(horizontal, 1, .15)
        xs = _group_positions(vertical, 0, .15)
        horizontal_lines = tuple(StructuralLine(
            line_id=f"h-{i}", orientation="HORIZONTAL", bbox=(0, y, rgb.width, y + 1),
            confidence=float((horizontal[y] > 0).mean()),
        ) for i, y in enumerate(ys))
        vertical_lines = tuple(StructuralLine(
            line_id=f"v-{i}", orientation="VERTICAL", bbox=(x, 0, x + 1, rgb.height),
            confidence=float((vertical[:, x] > 0).mean()),
        ) for i, x in enumerate(xs))
        count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8))
        components = tuple(
            (int(x), int(y), int(x + w), int(y + h))
            for x, y, w, h, area in stats[1:count]
            if 8 <= area <= rgb.width * rgb.height * .02
        )
        checkboxes = tuple(box for box in components if .7 <=
            (box[2] - box[0]) / max(1, box[3] - box[1]) <= 1.3 and 6 <= box[2] - box[0] <= 80)
        table_regions: tuple[StructuralRegion, ...] = ()
        if len(xs) >= 3 and len(ys) >= 3:
            table_regions = (StructuralRegion(
                region_id="grid-0", kind="TABLE_GRID",
                bbox=(min(xs), min(ys), max(xs), max(ys)), confidence=min(1, (len(xs)+len(ys))/20),
            ),)
        contrast = float(gray.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = float(np.clip(np.median(np.abs(laplacian)) / 32.0, 0.0, 1.0))
        skew = _skew_degrees(binary)
        foreground = float((binary > 0).mean())
        dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        background = gray[gray >= np.percentile(gray, 75)]
        background_uniformity = float(
            np.clip(1.0 - (background.std() / 64.0), 0.0, 1.0)
        ) if background.size else 0.0
        edges = cv2.Canny(gray, 80, 180)
        edge_density = float((edges > 0).mean())
        binarization_quality = float(np.clip(
            .5 * min(1.0, dynamic_range / 128.0)
            + .5 * (1.0 - min(1.0, abs(foreground - .12) / .25)),
            0.0, 1.0,
        ))
        writing_type, handwriting_likelihood = _writing_signal(components)
        quality_score = (
            .25 * min(1.0, blur / 160.0) + .20 * min(1.0, contrast / 64.0)
            + .15 * (1.0 - noise) + .10 * background_uniformity
            + .15 * binarization_quality + .15 * (1.0 - min(1.0, abs(skew) / 8.0))
        )
        quality = (
            "UNREADABLE" if quality_score < .25 or foreground < .001
            else "LOW" if quality_score < .50
            else "MEDIUM" if quality_score < .75
            else "HIGH"
        )
        anchors = tuple(sorted({
            re.sub(r"[^A-Z0-9 ]", "", token.text.upper()).strip() for token in tokens
            if token.confidence >= .5
        }))
        observation = PageObservation(
            page_id=page_id, page_sha256=digest, width=rgb.width, height=rgb.height,
            aspect_ratio=rgb.width / rgb.height,
            image_quality=ImageQualityEvidence(blur_score=blur, contrast_score=contrast,
                foreground_ratio=foreground, quality_bucket=quality, skew_degrees=skew,
                noise_estimate=noise, resolution_width=rgb.width,
                resolution_height=rgb.height, resolution_dpi=resolution_dpi,
                writing_type=writing_type, handwriting_likelihood=handwriting_likelihood,
                source_channel=source_channel, dynamic_range=dynamic_range,
                background_uniformity=background_uniformity,
                orientation_degrees=0, binarization_quality=binarization_quality,
                text_density=foreground, edge_density=edge_density),
            ocr_tokens=tokens, text_lines=tuple(token.text for token in tokens),
            word_boxes=tuple(token.bbox for token in tokens), horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines, connected_components=components,
            checkbox_candidates=checkboxes, table_regions=table_regions,
            anchor_candidates=anchors, structural_regions=table_regions,
            ocr_model_version=model_version, preprocessing_version=self._preprocessing_version,
        )
        if not self._benchmark_mode:
            self._cache.put(key, observation)
        return observation
