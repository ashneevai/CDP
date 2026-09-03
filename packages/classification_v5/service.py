from __future__ import annotations

from collections import defaultdict

from packages.classification_v5.contracts import (
    ClassificationSignal,
    DocumentFamily,
    PageClassificationV5,
    PageKind,
    StructureKind,
)


class ClassificationV5Service:
    """Conservative ensemble classifier. Produces routing context only.

    The service never extracts or accepts fields. Low-confidence and low-margin
    outcomes fail closed to generic/unknown routes.
    """

    def __init__(self, min_confidence: float = 0.82, min_margin: float = 0.12) -> None:
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def classify(self, page_id: str, signals: list[ClassificationSignal]) -> PageClassificationV5:
        by_family: dict[str, list[ClassificationSignal]] = defaultdict(list)
        for signal in signals:
            by_family[signal.label].append(signal)

        ranked: list[tuple[str, float]] = []
        for label, items in by_family.items():
            # De-duplicate same-lineage votes so two correlated detectors do not
            # create artificial certainty.
            lineage_best: dict[str, float] = {}
            for item in items:
                lineage_best[item.lineage_id] = max(lineage_best.get(item.lineage_id, 0.0), item.confidence)
            score = sum(lineage_best.values()) / max(1, len(lineage_best))
            ranked.append((label, score))
        ranked.sort(key=lambda item: item[1], reverse=True)

        if not ranked:
            return self._unknown(page_id, signals, ["NO_CLASSIFICATION_SIGNALS"])

        top_label, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, top_score - runner_up)
        try:
            family = DocumentFamily(top_label)
        except ValueError:
            family = DocumentFamily.UNKNOWN

        if family is DocumentFamily.UNKNOWN or top_score < self.min_confidence or margin < self.min_margin:
            reasons = ["CLASSIFICATION_UNCERTAIN"]
            if top_score < self.min_confidence:
                reasons.append("LOW_CLASSIFICATION_CONFIDENCE")
            if margin < self.min_margin:
                reasons.append("LOW_CLASSIFICATION_MARGIN")
            return self._unknown(page_id, signals, reasons, confidence=top_score, margin=margin)

        page_kind = PageKind.NON_CLAIM if family in {DocumentFamily.NON_CLAIM, DocumentFamily.COVER_SHEET, DocumentFamily.CORRESPONDENCE} else PageKind.CLAIM
        structure = (
            StructureKind.STRUCTURED
            if family in {DocumentFamily.CMS_1500, DocumentFamily.UB_04, DocumentFamily.EOB, DocumentFamily.MEDICAL_BILL}
            else StructureKind.UNSTRUCTURED
        )
        route = {
            DocumentFamily.CMS_1500: "CMS_SPECIALIST",
            DocumentFamily.UB_04: "UB_SPECIALIST",
            DocumentFamily.EOB: "EOB_SPECIALIST",
            DocumentFamily.MEDICAL_BILL: "GENERIC_STRUCTURED",
        }.get(family, "GENERIC_UNSTRUCTURED")
        return PageClassificationV5(
            page_id=page_id,
            page_kind=page_kind,
            structure_kind=structure,
            family=family,
            confidence=top_score,
            margin=margin,
            signals=signals,
            route=route,
            requires_review=False,
        )

    def _unknown(
        self,
        page_id: str,
        signals: list[ClassificationSignal],
        reasons: list[str],
        confidence: float = 0.0,
        margin: float = 0.0,
    ) -> PageClassificationV5:
        return PageClassificationV5(
            page_id=page_id,
            page_kind=PageKind.UNKNOWN,
            structure_kind=StructureKind.UNKNOWN,
            family=DocumentFamily.UNKNOWN,
            confidence=confidence,
            margin=margin,
            signals=signals,
            route="SAFE_GENERIC",
            requires_review=True,
            reason_codes=reasons,
        )
