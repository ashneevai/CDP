from __future__ import annotations

import hashlib
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


class PageObservationService:
    version = "page-observation-service-v1"

    def __init__(self, extractor: FullPageExtractor, *, preprocessing_version: str,
                 cache: PageObservationCache | None = None):
        self._extractor = extractor
        self._preprocessing_version = preprocessing_version
        self._cache = cache or PageObservationCache()

    def observe(self, page_id: str, image: Image.Image, *, page_sha256: str | None = None):
        rgb = image.convert("RGB")
        digest = page_sha256 or hashlib.sha256(rgb.tobytes()).hexdigest()
        model_version = getattr(self._extractor, "model_version", "unknown")
        key = self._cache.key(digest, model_version, self._preprocessing_version, page_id)
        if cached := self._cache.get(key):
            return cached
        lines = self._extractor.extract(rgb)
        tokens = tuple(
            ObservationToken(
                token_id=f"ocr-{index}", text=line.text.strip(),
                bbox=(line.x0, line.y0, line.x1, line.y1), confidence=line.confidence,
                line_index=index, reading_order=index,
                normalized_text=re.sub(r"\s+", " ", line.text).strip().casefold(),
                engine=getattr(self._extractor, "engine_name", "unknown"),
                model_name=getattr(self._extractor, "model_name", "unknown"),
                model_version=model_version,
                preprocessing_variant=self._preprocessing_version,
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
        foreground = float((binary > 0).mean())
        quality = "degraded" if blur < 40 or contrast < 25 else "clean"
        anchors = tuple(sorted({
            re.sub(r"[^A-Z0-9 ]", "", token.text.upper()).strip() for token in tokens
            if token.confidence >= .5
        }))
        observation = PageObservation(
            page_id=page_id, page_sha256=digest, width=rgb.width, height=rgb.height,
            aspect_ratio=rgb.width / rgb.height,
            image_quality=ImageQualityEvidence(blur_score=blur, contrast_score=contrast,
                foreground_ratio=foreground, quality_bucket=quality),
            ocr_tokens=tokens, text_lines=tuple(token.text for token in tokens),
            word_boxes=tuple(token.bbox for token in tokens), horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines, connected_components=components,
            checkbox_candidates=checkboxes, table_regions=table_regions,
            anchor_candidates=anchors, structural_regions=table_regions,
            ocr_model_version=model_version, preprocessing_version=self._preprocessing_version,
            ocr_provenance={
                "engine": getattr(self._extractor, "engine_name", "unknown"),
                "model_name": getattr(self._extractor, "model_name", "unknown"),
                "model_version": model_version,
            },
        )
        self._cache.put(key, observation)
        return observation
