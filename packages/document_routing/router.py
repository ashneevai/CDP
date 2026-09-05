from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml
from PIL import Image
from pydantic import Field, model_validator

from packages.domain.common import DomainModel

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/document_routing.yaml"


class TextGeometry(Protocol):
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


@dataclass
class _JoinedGeometry:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


def _ordered_phrase_candidates(lines: list[TextGeometry], anchor: str) -> list[TextGeometry]:
    """Join bounded adjacent OCR tokens while retaining their union geometry."""
    ordered = sorted(lines, key=lambda line: (round(line.y0 / 18), line.x0, line.y0))
    width = max(1, len(_routing_tokens(anchor)))
    candidates: list[TextGeometry] = list(ordered)
    for size in range(2, min(width + 2, 6)):
        for index in range(len(ordered) - size + 1):
            window = ordered[index : index + size]
            # Identity/anchor phrases may join only adjacent observations on one
            # visual row. This prevents reconstruction across columns or zones.
            centers = [(item.y0 + item.y1) / 2 for item in window]
            gaps = [window[offset + 1].x0 - window[offset].x1 for offset in range(size - 1)]
            if max(centers) - min(centers) > 32 or any(gap > 160 for gap in gaps):
                continue
            candidates.append(
                _JoinedGeometry(
                    " ".join(item.text for item in window),
                    min(item.x0 for item in window),
                    min(item.y0 for item in window),
                    max(item.x1 for item in window),
                    max(item.y1 for item in window),
                    min(float(getattr(item, "confidence", 0.0)) for item in window),
                )
            )
    return candidates


class MultiSignalRoute(StrEnum):
    CMS1500 = "CMS1500"
    UB04 = "UB04"
    OTHER_CLAIM_FORM = "OTHER_CLAIM_FORM"
    UNKNOWN_STRUCTURED = "UNKNOWN_STRUCTURED"
    UNKNOWN_UNSTRUCTURED = "UNKNOWN_UNSTRUCTURED"
    NON_CLAIM = "NON_CLAIM"


class IdentityAnchorEvidence(DomainModel):
    family: str
    canonical_anchor: str
    match_type: str
    ocr_confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[float, float, float, float]
    expected_identity_zone: tuple[float, float, float, float]
    zone_score: float = Field(ge=0, le=1)
    context_classification: str
    negation_reference_veto: bool
    policy_version: str
    reason_codes: tuple[str, ...] = ()


class RoutingEvidence(DomainModel):
    route: MultiSignalRoute
    confidence: float = Field(ge=0, le=1)
    scores: dict[str, float]
    best_score: float
    second_best_score: float
    margin: float
    grid_score: float
    horizontal_line_score: float
    vertical_line_score: float
    healthcare_label_density: float
    matched_anchors: dict[str, list[str]]
    reason_codes: list[str]
    router_version: str = "2.0"
    exact_anchor_count: int = 0
    normalized_anchor_count: int = 0
    fuzzy_anchor_count: int = 0
    high_value_anchor_count: int = 0
    medium_value_anchor_count: int = 0
    weighted_anchor_coverage: dict[str, float] = Field(default_factory=dict)
    anchor_geometry_score: dict[str, float] = Field(default_factory=dict)
    standard_structure: dict[str, float] = Field(default_factory=dict)
    anchor_geometry_evidence: list[dict] = Field(default_factory=list)
    anchor_combinations: list[dict] = Field(default_factory=list)
    eligibility: dict[str, bool] = Field(default_factory=dict)
    family_eligibility: dict[str, dict] = Field(default_factory=dict)
    identity_state: dict[str, str] = Field(default_factory=dict)
    field_topology_score: dict[str, float] = Field(default_factory=dict)
    conflicting_anchors: dict[str, list[str]] = Field(default_factory=dict)
    missing_required_anchors: dict[str, list[str]] = Field(default_factory=dict)
    identity_anchor_evidence: list[IdentityAnchorEvidence] = Field(default_factory=list)
    identity_policy_version: str = "strict-form-identity-v2"
    localization_allowed: bool = False

    @model_validator(mode="after")
    def localization_requires_canonical_authorization(self):
        if not self.localization_allowed:
            return self
        if self.route not in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
            raise ValueError("localization requires a canonical route")
        family = self.route.value
        opposing = "UB04" if family == "CMS1500" else "CMS1500"
        gates = self.family_eligibility.get(family, {}).get("authorization_gates", {})
        if (
            self.identity_state.get(family) != "CONFIRMED"
            or self.identity_state.get(opposing) == "CONFIRMED"
            or self.conflicting_anchors.get(family)
            or not gates
            or not all(gates.values())
        ):
            raise ValueError("localization requires every identity authorization gate")
        return self

    @property
    def winning_score(self) -> float:
        return self.best_score

    @property
    def runner_up_score(self) -> float:
        return self.second_best_score


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _routing_tokens(value: str) -> list[str]:
    tokens = _normalize(value).split()
    substitutions = {
        "patlent": "patient",
        "diagnos1s": "diagnosis",
        "biil": "bill",
        "blll": "bill",
        "hcpcs": "hcpcs",
    }
    return [substitutions.get(token, token) for token in tokens]


def _phrase_match(anchor: str, text: str) -> tuple[str | None, float]:
    wanted = _routing_tokens(anchor)
    observed = _routing_tokens(text)
    if not wanted or not observed:
        return None, 0.0
    normalized_anchor, normalized_text = " ".join(wanted), " ".join(observed)
    if normalized_anchor in normalized_text:
        raw_exact = _normalize(anchor) in _normalize(text)
        return ("EXACT" if raw_exact else "NORMALIZED"), 1.0
    width = len(wanted)
    # Short/generic labels are never fuzzy routing authority.
    if width == 1 or len(normalized_anchor) < 8:
        return None, 0.0
    threshold = 0.82 if len(normalized_anchor) >= 18 else 0.88
    best = max(
        (
            SequenceMatcher(
                None, normalized_anchor, " ".join(observed[index : index + width])
            ).ratio()
            for index in range(max(1, len(observed) - width + 1))
        ),
        default=0.0,
    )
    return ("FUZZY", best) if best >= threshold else (None, best)


def _anchor_found(anchor: str, text: str) -> bool:
    return _phrase_match(anchor, text)[0] is not None


def _bbox_score(line: TextGeometry, zone: list[float], width: int, height: int) -> float:
    cx = ((line.x0 + line.x1) / 2) / max(width, 1)
    cy = ((line.y0 + line.y1) / 2) / max(height, 1)
    x0, y0, x1, y1 = zone
    tolerance = 0.08
    if x0 - tolerance <= cx <= x1 + tolerance and y0 - tolerance <= cy <= y1 + tolerance:
        return 1.0 if x0 <= cx <= x1 and y0 <= cy <= y1 else 0.65
    return 0.0


def _line_scores(image: Image.Image) -> tuple[float, float, float]:
    gray = np.asarray(image.convert("L"))
    if max(gray.shape) > 1400:
        scale = 1400 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, gray.shape[1] // 25), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, gray.shape[0] // 35))),
    )
    h = min(1.0, np.count_nonzero(horizontal) / max(gray.size * 0.018, 1))
    v = min(1.0, np.count_nonzero(vertical) / max(gray.size * 0.012, 1))
    return h, v, min(1.0, 0.55 * h + 0.45 * v)


def _structure_scores(image: Image.Image, h: float, v: float, grid: float) -> dict[str, float]:
    ratio = image.width / max(image.height, 1)
    aspect = max(0.0, 1 - abs(ratio - 0.77) / 0.35)
    gray = np.asarray(image.convert("L"))
    start = int(gray.shape[0] * 0.18)
    end = int(gray.shape[0] * 0.78)
    band = gray[start:end]
    binary = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rows = np.count_nonzero(binary, axis=1) / max(binary.shape[1], 1)
    repeated = float(np.mean(rows > 0.28)) if len(rows) else 0.0
    service = min(1.0, repeated / 0.08)
    return {
        "grid_score": grid,
        "horizontal_line_score": h,
        "vertical_line_score": v,
        "service_table_score": service,
        "template_similarity": 0.0,
        "aspect_score": aspect,
        "CMS1500": min(1.0, 0.38 * grid + 0.32 * h + 0.12 * v + 0.18 * aspect),
        "UB04": min(1.0, 0.28 * grid + 0.20 * h + 0.20 * v + 0.20 * service + 0.12 * aspect),
    }


class MultiSignalRouter:
    def __init__(self, config: dict) -> None:
        self.config = config

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> MultiSignalRouter:
        return cls(yaml.safe_load(Path(path).read_text("utf-8")))

    @staticmethod
    def _nearby_context(candidate: TextGeometry, lines: list[TextGeometry]) -> str:
        center = (candidate.y0 + candidate.y1) / 2
        nearby = [
            line
            for line in lines
            if abs(((line.y0 + line.y1) / 2) - center) <= 36
            and line.x1 >= candidate.x0 - 500
            and line.x0 <= candidate.x1 + 500
        ]
        return " ".join(line.text for line in sorted(nearby, key=lambda line: line.x0))

    @staticmethod
    def _region_id(family: str, anchor: str) -> str:
        regions = {
            "CMS1500": {
                "health insurance claim form": "identity_header",
                "patients name": "patient_identity",
                "insured id number": "insured_identity",
                "diagnosis or nature of illness": "diagnosis",
                "federal tax id": "provider_billing",
            },
            "UB04": {
                "type of bill": "institutional_header",
                "patient control": "patient_control",
                "statement covers": "statement_period",
                "principal diagnosis": "diagnosis",
                "revenue code": "revenue_service",
                "hcpcs": "revenue_service",
                "service date": "revenue_service",
                "units": "revenue_service",
                "total charges": "revenue_service",
                "medical record": "patient_control",
            },
        }
        return regions.get(family, {}).get(anchor, anchor.replace(" ", "_"))

    def _identity_observations(
        self, image: Image.Image, lines: list[TextGeometry]
    ) -> tuple[dict[str, list[str]], list[IdentityAnchorEvidence], dict[str, list[str]]]:
        policy = self.config["form_identity"]
        minimum_ocr_confidence = float(policy.get("minimum_ocr_confidence", 0.0))
        vetoes = tuple(_normalize(item) for item in policy.get("context_vetoes", ()))
        matched: dict[str, list[str]] = {"CMS1500_IDENTITY": [], "UB04_IDENTITY": []}
        observations: list[IdentityAnchorEvidence] = []
        invalid: dict[str, list[str]] = {"CMS1500": [], "UB04": []}
        anchors = self.config["anchors"]
        for family, key in (("CMS1500", "CMS1500_IDENTITY"), ("UB04", "UB04_IDENTITY")):
            zone = policy[family]["identity_zone"]
            for anchor in anchors[key]:
                matches: list[tuple[TextGeometry, str, float, str, tuple[str, ...]]] = []
                for candidate in _ordered_phrase_candidates(lines, anchor):
                    # Detector artifacts below the governed confidence floor
                    # are neither positive identity evidence nor contradiction
                    # authority. This prevents a zero-confidence token outside
                    # the header from vetoing an otherwise complete form.
                    if float(getattr(candidate, "confidence", 0.0)) < minimum_ocr_confidence:
                        continue
                    match_type, phrase_score = _phrase_match(anchor, candidate.text)
                    if match_type is None:
                        continue
                    context = _normalize(self._nearby_context(candidate, lines))
                    found_vetoes = tuple(veto for veto in vetoes if veto in context)
                    matches.append((candidate, match_type, phrase_score, context, found_vetoes))
                if not matches:
                    continue
                valid_found = False
                invalid_context_found = False
                for candidate, match_type, phrase_score, _context, found_vetoes in matches:
                    zone_score = _bbox_score(candidate, zone, image.width, image.height)
                    reasons: list[str] = []
                    context_classification = "CANONICAL_IDENTITY"
                    veto = bool(found_vetoes)
                    if veto:
                        context_classification = "NEGATED_OR_REFERENTIAL"
                        reasons.extend(f"IDENTITY_CONTEXT_VETO:{item}" for item in found_vetoes)
                    if zone_score < 1.0:
                        context_classification = "OUTSIDE_IDENTITY_ZONE"
                        reasons.append("IDENTITY_OUTSIDE_EXPECTED_ZONE")
                    if match_type == "FUZZY":
                        context_classification = "UNBOUNDED_OCR_VARIANT"
                        reasons.append("FUZZY_IDENTITY_NOT_AUTHORIZING")
                    invalid_context_found = (
                        invalid_context_found
                        or bool(found_vetoes)
                        or (match_type in {"EXACT", "NORMALIZED"} and zone_score < 1.0)
                    )
                    authorizing = (
                        not veto and zone_score == 1.0 and match_type in {"EXACT", "NORMALIZED"}
                    )
                    if authorizing:
                        valid_found = True
                        reasons.append("IDENTITY_ANCHOR_VALID")
                    observations.append(
                        IdentityAnchorEvidence(
                            family=family,
                            canonical_anchor=anchor,
                            match_type=match_type,
                            ocr_confidence=float(getattr(candidate, "confidence", 0.0)),
                            bounding_box=(candidate.x0, candidate.y0, candidate.x1, candidate.y1),
                            expected_identity_zone=tuple(zone),
                            zone_score=zone_score,
                            context_classification=context_classification,
                            negation_reference_veto=veto,
                            policy_version=policy["policy_version"],
                            reason_codes=tuple(reasons),
                        )
                    )
                if valid_found:
                    matched[key].append(anchor)
                elif invalid_context_found:
                    invalid[family].append(f"INVALID_IDENTITY_CONTEXT:{anchor}")
        return matched, observations, invalid

    def route(self, image: Image.Image, lines: list[TextGeometry]) -> RoutingEvidence:
        anchors = self.config["anchors"]
        identity_matched, identity_evidence, identity_invalid = self._identity_observations(
            image, lines
        )
        matched: dict[str, list[str]] = {
            name: [
                anchor
                for anchor in values
                if any(
                    _phrase_match(anchor, candidate.text)[0] is not None
                    for candidate in _ordered_phrase_candidates(lines, anchor)
                )
            ]
            for name, values in anchors.items()
            if name not in {"CMS1500_IDENTITY", "UB04_IDENTITY", "CMS1500", "UB04"}
        }
        matched.update(identity_matched)

        geometry_evidence: list[dict] = []
        match_counts: Counter[str] = Counter()
        family_counts: dict[str, Counter] = {}
        geometry_scores: dict[str, float] = {}
        weighted: dict[str, float] = {}
        independent_regions: dict[str, set[str]] = {}
        weight_value = {"high": 3.0, "medium": 2.0, "low": 0.5}
        for family in ("CMS1500", "UB04"):
            family_geometry: list[float] = []
            numerator = denominator = 0.0
            family_count: Counter[str] = Counter()
            family_matched: list[str] = []
            regions: set[str] = set()
            classes = self.config.get("anchor_weights", {}).get(family, {})
            for anchor_class, values in classes.items():
                for anchor in values:
                    weight = weight_value[anchor_class]
                    denominator += weight
                    candidates = []
                    for candidate in _ordered_phrase_candidates(lines, anchor):
                        match_type, phrase_score = _phrase_match(anchor, candidate.text)
                        if match_type:
                            zone = self.config.get("anchor_zones", {}).get(family, {}).get(anchor)
                            zone_score = (
                                _bbox_score(candidate, zone, image.width, image.height)
                                if zone
                                else 0.0
                            )
                            candidates.append(
                                (
                                    phrase_score * zone_score,
                                    candidate,
                                    match_type,
                                    phrase_score,
                                    zone_score,
                                )
                            )
                    if not candidates:
                        continue
                    _, candidate, match_type, phrase_score, zone_score = max(
                        candidates, key=lambda item: item[0]
                    )
                    if zone_score <= 0:
                        continue
                    region_id = self._region_id(family, anchor)
                    numerator += weight * phrase_score
                    family_geometry.append(zone_score)
                    family_matched.append(anchor)
                    regions.add(region_id)
                    match_counts[match_type] += 1
                    match_counts[anchor_class] += 1
                    family_count[match_type] += 1
                    family_count[anchor_class] += 1
                    geometry_evidence.append(
                        {
                            "family": family,
                            "anchor": anchor,
                            "region_id": region_id,
                            "expected_zone": self.config.get("anchor_zones", {})
                            .get(family, {})
                            .get(anchor),
                            "observed_bbox": [
                                candidate.x0,
                                candidate.y0,
                                candidate.x1,
                                candidate.y1,
                            ],
                            "ocr_confidence": float(getattr(candidate, "confidence", 0.0)),
                            "zone_match": True,
                            "geometry_score": zone_score,
                            "match_type": match_type,
                            "phrase_score": phrase_score,
                            "anchor_class": anchor_class.upper() + "_DISCRIMINATION",
                        }
                    )
            matched[family] = family_matched
            weighted[family] = numerator / max(denominator, 1)
            family_counts[family] = family_count
            independent_regions[family] = regions
            geometry_scores[family] = statistics.fmean(family_geometry) if family_geometry else 0.0

        h, v, grid = _line_scores(image)
        structure = _structure_scores(image, h, v, grid)
        identity_present = {
            "CMS1500": bool(matched["CMS1500_IDENTITY"]),
            "UB04": bool(matched["UB04_IDENTITY"]),
        }
        healthcare = len(matched["healthcare"]) / len(anchors["healthcare"])
        negative = len(matched["negative"]) / max(len(anchors["negative"]), 1)
        combinations = []
        combination_bonus = {"CMS1500": 0.0, "UB04": 0.0}
        for family, items in self.config.get("anchor_combinations", {}).items():
            detected = set(matched[family])
            for item in items:
                present = [anchor for anchor in item["anchors"] if anchor in detected]
                score = len(present) / len(item["anchors"])
                geometry_valid = geometry_scores[family] >= self.config["minimum_geometry_score"]
                combinations.append(
                    {
                        "family": family,
                        "combination_id": item["id"],
                        "required_or_weighted_anchors": item["anchors"],
                        "anchors_detected": present,
                        "geometry_valid": geometry_valid,
                        "combination_score": score,
                    }
                )
                if geometry_valid and score >= 0.66:
                    combination_bonus[family] = max(combination_bonus[family], score)
        topology = {
            family: statistics.fmean(
                item["combination_score"] for item in combinations if item["family"] == family
            )
            for family in ("CMS1500", "UB04")
        }
        required = {
            family: sorted(
                {
                    anchor
                    for item in self.config.get("anchor_combinations", {}).get(family, [])
                    for anchor in item["anchors"]
                }
            )
            for family in ("CMS1500", "UB04")
        }
        missing = {
            family: sorted(set(required[family]) - set(matched[family])) for family in required
        }
        noncanonical = matched.get("noncanonical_claim", [])

        raw_standard = {}
        for family in ("CMS1500", "UB04"):
            raw_standard[family] = min(
                1.0,
                0.34 * weighted[family]
                + 0.16 * geometry_scores[family]
                + 0.25 * structure[family]
                + 0.15 * float(identity_present[family])
                + 0.10 * combination_bonus[family],
            )
        scores = {
            "CMS1500": min(
                1.0,
                raw_standard["CMS1500"]
                + self.config.get("identity_discrimination_bonus", 0)
                * float(identity_present["CMS1500"]),
            ),
            "UB04": min(
                1.0,
                raw_standard["UB04"]
                + self.config.get("identity_discrimination_bonus", 0)
                * float(identity_present["UB04"]),
            ),
            "OTHER_CLAIM_FORM": min(
                1.0, 0.50 * healthcare + 0.32 * grid + 0.18 * min(1, len(lines) / 18)
            ),
            "UNKNOWN_STRUCTURED": min(1.0, 0.32 * grid + 0.18 * min(1, len(lines) / 18)),
            "UNKNOWN_UNSTRUCTURED": min(
                1.0, 0.45 * (1 - grid) + 0.25 * healthcare + 0.30 * min(1, len(lines) / 12)
            ),
            "NON_CLAIM": min(
                1.0, 0.70 * negative + 0.20 * (1 - healthcare) + 0.10 * (len(lines) < 5)
            ),
        }
        standard = sorted(
            ((name, scores[name]) for name in ("CMS1500", "UB04")),
            key=lambda item: item[1],
            reverse=True,
        )
        standard_margin = standard[0][1] - standard[1][1]

        raw_identity_state = {
            family: (
                "REJECTED"
                if identity_invalid[family]
                else "CONFIRMED"
                if identity_present[family]
                else "UNKNOWN"
            )
            for family in ("CMS1500", "UB04")
        }
        conflicts: dict[str, list[str]] = {}
        family_eligibility: dict[str, dict] = {}
        eligible = {"CMS1500": False, "UB04": False}
        authorization_path: dict[str, str | None] = {"CMS1500": None, "UB04": None}
        for family in ("CMS1500", "UB04"):
            opposing = "UB04" if family == "CMS1500" else "CMS1500"
            conflicts[family] = list(noncanonical) + list(identity_invalid[family])
            if raw_identity_state[opposing] == "CONFIRMED":
                conflicts[family].append(f"{opposing}_IDENTITY_CONFLICT")
            policy = self.config["form_identity"][family]
            common = {
                "minimum_high_value_anchors": family_counts[family]["high"]
                >= policy["minimum_high_value_anchors"],
                "minimum_independent_regions": len(independent_regions[family])
                >= policy["minimum_independent_regions"],
                "minimum_weighted_anchor_coverage": weighted[family]
                >= self.config["minimum_weighted_anchor_coverage"],
                "minimum_geometry_score": geometry_scores[family]
                >= self.config["minimum_geometry_score"],
                "minimum_structure_score": structure[family]
                >= self.config["minimum_structure_score"],
                "minimum_family_margin": (scores[family] - scores[opposing])
                >= self.config["minimum_standard_margin"],
                "no_conflicting_or_noncanonical_evidence": not conflicts[family],
                "opposing_family_not_confirmed": raw_identity_state[opposing] != "CONFIRMED",
            }
            explicit_gates = {
                **common,
                "canonical_identity_in_expected_zone": raw_identity_state[family] == "CONFIRMED",
                "minimum_absolute_standard_score": scores[family]
                >= self.config["minimum_identity_backed_standard_score"],
            }
            topology_gates = {
                **common,
                "identity_header_absent_without_rejection": raw_identity_state[family] == "UNKNOWN",
                "complete_configured_topology": topology[family]
                >= policy["minimum_topology_score"],
                "required_grid_structure": grid >= self.config["minimum_structure_score"],
                "minimum_absolute_standard_score": scores[family]
                >= max(
                    self.config["minimum_standard_score"],
                    self.config["minimum_structure_backed_score"],
                ),
            }
            explicit_gates = {key: bool(value) for key, value in explicit_gates.items()}
            topology_gates = {key: bool(value) for key, value in topology_gates.items()}
            if all(explicit_gates.values()):
                eligible[family] = True
                authorization_path[family] = "EXPLICIT_IDENTITY"
                selected = explicit_gates
            elif all(topology_gates.values()):
                eligible[family] = True
                authorization_path[family] = "COMPLETE_TOPOLOGY"
                selected = topology_gates
            else:
                selected = explicit_gates if identity_present[family] else topology_gates
            family_eligibility[family] = {
                "eligible": eligible[family],
                "authorization_path": authorization_path[family],
                "authorization_gates": selected,
                "explicit_identity_gates": explicit_gates,
                "topology_gates": topology_gates,
                "high_value_anchor_count": family_counts[family]["high"],
                "independent_regions": sorted(independent_regions[family]),
            }

        identity_state = {
            family: (
                "CONFIRMED" if eligible[family] else "REJECTED" if conflicts[family] else "UNKNOWN"
            )
            for family in ("CMS1500", "UB04")
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, second = ranked[0], ranked[1]
        family = standard[0][0]
        reasons = ["MULTI_SIGNAL_ROUTER", f"BEST:{best[0]}"]
        if eligible[family]:
            route = MultiSignalRoute(family)
            reasons.extend(
                [
                    (
                        f"{family}_IDENTITY_CONFIRMED"
                        if authorization_path[family] == "EXPLICIT_IDENTITY"
                        else f"{family}_TOPOLOGY_CONFIRMED"
                    ),
                    f"{family}_AUTHORIZATION_PATH:{authorization_path[family]}",
                    f"{family}_WEIGHTED_ANCHORS",
                    f"STANDARD_MARGIN:{standard_margin:.3f}",
                    f"{family}_GEOMETRY_CONFIRMED",
                ]
            )
            if family == "UB04" and structure["service_table_score"] >= 0.35:
                reasons.append("UB04_SERVICE_TABLE_CONFIRMED")
        elif (
            scores["NON_CLAIM"] >= self.config["non_claim_score"]
            and len(matched["negative"]) >= 2
            and healthcare <= 0.20
        ):
            route = MultiSignalRoute.NON_CLAIM
            reasons.append("MULTIPLE_NEGATIVE_ANCHORS_LOW_HEALTHCARE_DENSITY")
        elif (noncanonical or healthcare >= 0.20 or len(matched[family]) >= 2) and (
            grid >= 0.20 or len(lines) >= 3
        ):
            route = MultiSignalRoute.OTHER_CLAIM_FORM
            reasons.extend(
                [
                    "CLAIM_FORM_NONCANONICAL",
                    f"{family}_REJECT_CONFLICTING_EVIDENCE"
                    if conflicts[family]
                    else f"{family}_REJECT_MISSING_CANONICAL_ANCHORS",
                ]
            )
        elif scores["UNKNOWN_STRUCTURED"] >= self.config["minimum_structured_score"]:
            route = MultiSignalRoute.UNKNOWN_STRUCTURED
            reasons.extend(["STANDARD_EVIDENCE_INSUFFICIENT", "UNKNOWN_STRUCTURED_CONFIRMED"])
        else:
            route = MultiSignalRoute.UNKNOWN_UNSTRUCTURED
            reasons.append("UNKNOWN_UNSTRUCTURED_CONFIRMED")

        localization_allowed = (
            route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}
            and eligible[route.value]
            and identity_state[route.value] == "CONFIRMED"
            and not conflicts[route.value]
            and raw_identity_state["UB04" if route == MultiSignalRoute.CMS1500 else "CMS1500"]
            != "CONFIRMED"
        )
        return RoutingEvidence(
            route=route,
            confidence=best[1],
            scores=scores,
            best_score=best[1],
            second_best_score=second[1],
            margin=best[1] - second[1],
            grid_score=grid,
            horizontal_line_score=h,
            vertical_line_score=v,
            healthcare_label_density=healthcare,
            matched_anchors=matched,
            reason_codes=reasons,
            router_version=self.config.get("router_version", "strict-form-identity-v2"),
            exact_anchor_count=match_counts["EXACT"],
            normalized_anchor_count=match_counts["NORMALIZED"],
            fuzzy_anchor_count=match_counts["FUZZY"],
            high_value_anchor_count=match_counts["high"],
            medium_value_anchor_count=match_counts["medium"],
            weighted_anchor_coverage=weighted,
            anchor_geometry_score=geometry_scores,
            standard_structure=structure,
            anchor_geometry_evidence=geometry_evidence,
            anchor_combinations=combinations,
            eligibility=eligible,
            family_eligibility=family_eligibility,
            identity_state=identity_state,
            field_topology_score=topology,
            conflicting_anchors=conflicts,
            missing_required_anchors=missing,
            identity_anchor_evidence=identity_evidence,
            identity_policy_version=self.config["form_identity"]["policy_version"],
            localization_allowed=localization_allowed,
        )
