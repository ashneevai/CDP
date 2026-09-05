"""Page routing: Bundle A/B/C/D classification and CMS/UB claim-page
selection, per the escalation order in docs/ARCHITECTURE.md §9 --

    anchor phrases -> grid/layout signature -> template similarity
    -> MobileNetV3 fallback -> human review

applied at two different granularities:

- Bundle A/C (single page): verify anchors for the expected form; if
  confident, skip every later (more expensive) step entirely ("trusted
  anchor fast-path" in the spec).
- Bundle B (multipage CMS-1500 + attachments): the same escalation ladder
  is applied *per page* to find the one CMS-1500 page among N. Every page
  is classified (and every non-selected page is explicitly marked
  ATTACHMENT, never extracted, never discarded) so the "preserve
  attachments" acceptance criterion is auditable from the classification
  records alone, not just from what the extraction worker chose to touch.
- Bundle D: no single-page/no confident multipage CMS-1500 match ->
  unstructured, routed to `workers.unstructured_extraction` (Phase 2/4).
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from packages.document_routing import MultiSignalRoute, MultiSignalRouter, RoutingEvidence
from packages.domain.enums import BundleType, ClassificationMethod, PageRole
from packages.domain.registration import RegistrationEvidence
from packages.templates.models import Template
from workers.page_detection.anchor_matching import AnchorMatchResult, verify_anchors
from workers.page_detection.grid_signature import (
    GridSignature,
    compute_grid_signature,
    signature_similarity,
)
from workers.page_detection.template_alignment import align_to_reference
from workers.page_detection.text_extraction import ModelNotAvailableError, TextExtractor, TextLine

# Escalation thresholds -- tuned against the real dataset in
# tests/unit/test_page_routing.py; not claimed to be production-final.
ANCHOR_CONFIDENT_THRESHOLD = 0.75
# 0.75, not the un-tuned 0.90: across every real Group A/C scan in
# dataset_raw, true-positive own-form grid-signature scores ranged 0.80-0.98
# while cross-form contamination topped out at 0.79 -- a real but narrow
# margin at the absolute-threshold level alone (see
# test_single_page_grid_signature_thresholds_hold_against_real_dataset in
# test_page_routing.py, which re-measures this and fails loudly if a future
# template/dataset change erodes it). GRID_AMBIGUITY_MARGIN below is what
# actually carries the safety margin; the absolute floor here just rejects a
# page that doesn't convincingly match *either* template.
GRID_CONFIDENT_THRESHOLD = 0.75
ALIGNMENT_CONFIDENT_THRESHOLD = 0.35
# Minimum separation between the best and second-best page candidate for a
# multipage bundle to be resolved automatically rather than sent to review.
AMBIGUITY_MARGIN = 0.08
# Same idea, single-page grid-signature scoring (route_single_page): the
# real dataset's minimum observed separation between a page's own-form and
# cross-form grid-signature score was 0.133 -- 0.10 keeps a real cushion
# under that (see the same real-dataset test referenced above).
GRID_AMBIGUITY_MARGIN = 0.10


@dataclass(frozen=True)
class PageCandidateScore:
    page_number: int
    method: ClassificationMethod
    confidence: float
    reason_codes: list[str]
    registration_evidence: RegistrationEvidence | None = None


@dataclass(frozen=True)
class PageRoutingResult:
    bundle_type: BundleType
    selected_page_number: int | None
    template: Template | None
    page_roles: dict[int, PageRole]
    page_scores: dict[int, PageCandidateScore]
    needs_review: bool
    reason_codes: list[str]
    canonical_route: MultiSignalRoute | None = None
    route_decision: RoutingEvidence | None = None


class PageRoutingService:
    def __init__(
        self,
        cms_template: Template,
        ub_template: Template,
        text_extractor: TextExtractor | None = None,
        cms_reference_image: Image.Image | None = None,
        ub_reference_image: Image.Image | None = None,
        multi_signal_router: MultiSignalRouter | None = None,
        enable_router_v3: bool = False,
    ) -> None:
        self._cms_template = cms_template
        self._ub_template = ub_template
        self._text_extractor = text_extractor
        self._cms_reference_image = cms_reference_image
        self._ub_reference_image = ub_reference_image
        self._multi_signal_router = multi_signal_router or MultiSignalRouter.load()
        self._enable_router_v3 = enable_router_v3
        self._cms_reference_signature: GridSignature | None = (
            compute_grid_signature(cms_reference_image) if cms_reference_image else None
        )
        self._ub_reference_signature: GridSignature | None = (
            compute_grid_signature(ub_reference_image) if ub_reference_image else None
        )

    # -- single-page fast path (Bundle A / C) ----------------------------

    def _extract_anchor_lines(self, image: Image.Image) -> list[TextLine] | None:
        if self._text_extractor is None:
            return None
        try:
            return self._text_extractor.extract(image)
        except ModelNotAvailableError:
            return None

    @staticmethod
    def _anchor_score(lines: list[TextLine] | None, template: Template) -> AnchorMatchResult | None:
        if lines is None:
            return None
        return verify_anchors(lines, template.anchor_definitions)

    def route_single_page(self, image: Image.Image) -> PageRoutingResult:
        # OCR the page once and reuse the same evidence for both form
        # families. Previously a non-CMS page was passed through PaddleOCR
        # twice (CMS check, then UB check), doubling peak work and memory.
        anchor_lines = self._extract_anchor_lines(image)
        if self._enable_router_v3 and anchor_lines is not None:
            return self._canonical_single_page(image, anchor_lines)
        cms_anchors = self._anchor_score(anchor_lines, self._cms_template)
        if cms_anchors is not None and cms_anchors.all_required_matched:
            score = PageCandidateScore(
                page_number=1,
                method=ClassificationMethod.TRUSTED_ANCHOR_SKIP,
                confidence=cms_anchors.confidence,
                reason_codes=["cms1500_anchors_matched"],
            )
            return PageRoutingResult(
                bundle_type=BundleType.A_CMS1500_SINGLE,
                selected_page_number=1,
                template=self._cms_template,
                page_roles={1: PageRole.CMS1500_CLAIM_PAGE},
                page_scores={1: score},
                needs_review=False,
                reason_codes=["bundle_a_trusted_anchor_fast_path"],
            )

        ub_anchors = self._anchor_score(anchor_lines, self._ub_template)
        if ub_anchors is not None and ub_anchors.all_required_matched:
            score = PageCandidateScore(
                page_number=1,
                method=ClassificationMethod.TRUSTED_ANCHOR_SKIP,
                confidence=ub_anchors.confidence,
                reason_codes=["ub_anchors_matched"],
            )
            return PageRoutingResult(
                bundle_type=BundleType.C_UB_SINGLE,
                selected_page_number=1,
                template=self._ub_template,
                page_roles={1: PageRole.UB_CLAIM_PAGE},
                page_scores={1: score},
                needs_review=False,
                reason_codes=["bundle_c_trusted_anchor_fast_path"],
            )

        # Anchors inconclusive (or no text extractor available) -- fall
        # back to grid signature against both references. Require both an
        # absolute floor AND separation from the runner-up template: the
        # real dataset shows narrow absolute margins between own-form and
        # cross-form scores (see GRID_CONFIDENT_THRESHOLD's comment), so the
        # margin check is what actually protects against misclassifying one
        # form as the other, not the absolute threshold alone.
        scored = self._score_page_against_templates(image)
        if scored:
            template_id, best_score, method = scored[0]
            runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
            if (
                best_score >= self._confidence_threshold(method)
                and (best_score - runner_up_score) >= GRID_AMBIGUITY_MARGIN
            ):
                is_cms = template_id == self._cms_template.template_id
                role = PageRole.CMS1500_CLAIM_PAGE if is_cms else PageRole.UB_CLAIM_PAGE
                bundle = BundleType.A_CMS1500_SINGLE if is_cms else BundleType.C_UB_SINGLE
                template = self._cms_template if is_cms else self._ub_template
                score = PageCandidateScore(
                    page_number=1,
                    method=method,
                    confidence=best_score,
                    reason_codes=[f"{method.value}_match"],
                )
                return PageRoutingResult(
                    bundle_type=bundle,
                    selected_page_number=1,
                    template=template,
                    page_roles={1: role},
                    page_scores={1: score},
                    needs_review=False,
                    reason_codes=[f"single_page_{method.value}_match"],
                )

        # Independent standard fingerprint recovery. Generic fallback never
        # forces a standard unless its absolute score and inter-standard
        # margin both pass the versioned routing policy.
        if anchor_lines is not None:
            generic = self._multi_signal_router.route(image, anchor_lines)
            if generic.route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
                is_cms = generic.route is MultiSignalRoute.CMS1500
                template = self._cms_template if is_cms else self._ub_template
                role = PageRole.CMS1500_CLAIM_PAGE if is_cms else PageRole.UB_CLAIM_PAGE
                bundle = BundleType.A_CMS1500_SINGLE if is_cms else BundleType.C_UB_SINGLE
                score = PageCandidateScore(
                    page_number=1,
                    method=ClassificationMethod.ANCHOR_PHRASE,
                    confidence=generic.confidence,
                    reason_codes=generic.reason_codes,
                )
                return PageRoutingResult(
                    bundle_type=bundle,
                    selected_page_number=1,
                    template=template,
                    page_roles={1: role},
                    page_scores={1: score},
                    needs_review=False,
                    reason_codes=[
                        f"multi_signal_{generic.route.value.lower()}",
                        *generic.reason_codes,
                    ],
                )
            generic_reasons = [f"generic_route:{generic.route.value}", *generic.reason_codes]
            if generic.route in {
                MultiSignalRoute.UNKNOWN_STRUCTURED,
                MultiSignalRoute.UNKNOWN_UNSTRUCTURED,
                MultiSignalRoute.NON_CLAIM,
            }:
                bundle = {
                    MultiSignalRoute.UNKNOWN_STRUCTURED: BundleType.UNKNOWN_STRUCTURED,
                    MultiSignalRoute.UNKNOWN_UNSTRUCTURED: BundleType.UNKNOWN_UNSTRUCTURED,
                    MultiSignalRoute.NON_CLAIM: BundleType.NON_CLAIM,
                }[generic.route]
                return PageRoutingResult(
                    bundle_type=bundle,
                    selected_page_number=None,
                    template=None,
                    page_roles={
                        1: (
                            PageRole.UNKNOWN
                            if generic.route is MultiSignalRoute.NON_CLAIM
                            else PageRole.UNSTRUCTURED_CLAIM_PAGE
                        )
                    },
                    page_scores={},
                    needs_review=False,
                    reason_codes=generic_reasons,
                    canonical_route=generic.route,
                    route_decision=generic,
                )
        else:
            generic_reasons = []
        return PageRoutingResult(
            bundle_type=BundleType.D_UNSTRUCTURED,
            selected_page_number=None,
            template=None,
            page_roles={1: PageRole.UNSTRUCTURED_CLAIM_PAGE},
            page_scores={},
            needs_review=False,
            reason_codes=["no_standard_template_match_routed_to_unstructured", *generic_reasons],
        )

    def _canonical_single_page(
        self, image: Image.Image, lines: list[TextLine]
    ) -> PageRoutingResult:
        """One V3 decision brain; legacy logic is rollback-only evidence production."""
        decision = self._multi_signal_router.route(image, lines)
        if decision.route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}:
            is_cms = decision.route is MultiSignalRoute.CMS1500
            template = self._cms_template if is_cms else self._ub_template
            return PageRoutingResult(
                bundle_type=BundleType.A_CMS1500_SINGLE if is_cms else BundleType.C_UB_SINGLE,
                selected_page_number=1,
                template=template,
                page_roles={1: PageRole.CMS1500_CLAIM_PAGE if is_cms else PageRole.UB_CLAIM_PAGE},
                page_scores={
                    1: PageCandidateScore(
                        1,
                        ClassificationMethod.ANCHOR_PHRASE,
                        decision.confidence,
                        decision.reason_codes,
                    )
                },
                needs_review=False,
                reason_codes=decision.reason_codes,
                canonical_route=decision.route,
                route_decision=decision,
            )
        bundle = {
            MultiSignalRoute.OTHER_CLAIM_FORM: BundleType.UNKNOWN_STRUCTURED,
            MultiSignalRoute.UNKNOWN_STRUCTURED: BundleType.UNKNOWN_STRUCTURED,
            MultiSignalRoute.UNKNOWN_UNSTRUCTURED: BundleType.UNKNOWN_UNSTRUCTURED,
            MultiSignalRoute.NON_CLAIM: BundleType.NON_CLAIM,
        }[decision.route]
        role = (
            PageRole.UNKNOWN
            if decision.route is MultiSignalRoute.NON_CLAIM
            else PageRole.UNSTRUCTURED_CLAIM_PAGE
        )
        return PageRoutingResult(
            bundle_type=bundle,
            selected_page_number=None,
            template=None,
            page_roles={1: role},
            page_scores={},
            needs_review=False,
            reason_codes=decision.reason_codes,
            canonical_route=decision.route,
            route_decision=decision,
        )

    def _confidence_threshold(self, method: ClassificationMethod) -> float:
        return {
            ClassificationMethod.ANCHOR_PHRASE: ANCHOR_CONFIDENT_THRESHOLD,
            ClassificationMethod.TRUSTED_ANCHOR_SKIP: ANCHOR_CONFIDENT_THRESHOLD,
            ClassificationMethod.GRID_LAYOUT_SIGNATURE: GRID_CONFIDENT_THRESHOLD,
            ClassificationMethod.TEMPLATE_SIMILARITY: ALIGNMENT_CONFIDENT_THRESHOLD,
        }.get(method, 1.0)

    def _score_page_against_templates(
        self, image: Image.Image
    ) -> list[tuple[str, float, ClassificationMethod]]:
        """Grid-signature score against every configured reference template,
        best first -- callers need both the best score (absolute floor) and
        the runner-up (ambiguity margin), see route_single_page."""
        results: list[tuple[str, float, ClassificationMethod]] = []
        sig = (
            compute_grid_signature(image)
            if self._cms_reference_signature is not None or self._ub_reference_signature is not None
            else None
        )
        if self._cms_reference_signature is not None:
            results.append(
                (
                    self._cms_template.template_id,
                    signature_similarity(sig, self._cms_reference_signature),
                    ClassificationMethod.GRID_LAYOUT_SIGNATURE,
                )
            )
        if self._ub_reference_signature is not None:
            results.append(
                (
                    self._ub_template.template_id,
                    signature_similarity(sig, self._ub_reference_signature),
                    ClassificationMethod.GRID_LAYOUT_SIGNATURE,
                )
            )
        return sorted(results, key=lambda r: r[1], reverse=True)

    # -- multipage bundle path (Bundle B) --------------------------------

    def route_multipage_bundle(self, images: list[Image.Image]) -> PageRoutingResult:
        """`images` is 1-indexed conceptually; `images[0]` is page 1."""
        scores: dict[int, PageCandidateScore] = {}

        for index, image in enumerate(images, start=1):
            anchors = self._anchor_score(self._extract_anchor_lines(image), self._cms_template)
            if anchors is not None and anchors.confidence > 0:
                scores[index] = PageCandidateScore(
                    page_number=index,
                    method=ClassificationMethod.ANCHOR_PHRASE,
                    confidence=anchors.confidence,
                    reason_codes=list(anchors.matched_phrases),
                )
                continue

            if self._cms_reference_image is not None:
                alignment = align_to_reference(image, self._cms_reference_image)
                if alignment.success:
                    scores[index] = PageCandidateScore(
                        page_number=index,
                        method=ClassificationMethod.TEMPLATE_SIMILARITY,
                        confidence=alignment.alignment_score,
                        reason_codes=[f"{alignment.good_match_count}_sift_matches"],
                        registration_evidence=alignment.evidence,
                    )
                    continue

            scores[index] = PageCandidateScore(
                page_number=index,
                method=ClassificationMethod.GRID_LAYOUT_SIGNATURE,
                confidence=0.0,
                reason_codes=["no_signal"],
            )

        ranked = sorted(scores.values(), key=lambda s: s.confidence, reverse=True)
        best = ranked[0]
        runner_up_confidence = ranked[1].confidence if len(ranked) > 1 else 0.0
        threshold = self._confidence_threshold(best.method) if best.confidence > 0 else 1.0
        confident_and_unique = (
            best.confidence >= threshold
            and (best.confidence - runner_up_confidence) >= AMBIGUITY_MARGIN
        )

        page_roles = {
            i: (PageRole.CMS1500_CLAIM_PAGE if i == best.page_number else PageRole.ATTACHMENT)
            for i in range(1, len(images) + 1)
        }

        if confident_and_unique:
            return PageRoutingResult(
                bundle_type=BundleType.B_CMS1500_BUNDLE,
                selected_page_number=best.page_number,
                template=self._cms_template,
                page_roles=page_roles,
                page_scores=scores,
                needs_review=False,
                reason_codes=[f"selected_page_{best.page_number}_via_{best.method.value}"],
            )

        if best.confidence > 0:
            # some signal, but not confident/unique -- do not guess
            return PageRoutingResult(
                bundle_type=BundleType.B_CMS1500_BUNDLE,
                selected_page_number=None,
                template=self._cms_template,
                page_roles={i: PageRole.UNKNOWN for i in range(1, len(images) + 1)},
                page_scores=scores,
                needs_review=True,
                reason_codes=["ambiguous_cms1500_page_selection"],
            )

        # no CMS-1500 signal anywhere in the bundle
        return PageRoutingResult(
            bundle_type=BundleType.D_UNSTRUCTURED,
            selected_page_number=None,
            template=None,
            page_roles={i: PageRole.UNSTRUCTURED_CLAIM_PAGE for i in range(1, len(images) + 1)},
            page_scores=scores,
            needs_review=False,
            reason_codes=["no_cms1500_signal_routed_to_unstructured"],
        )

    # -- entry point -------------------------------------------------------

    def route(self, images: list[Image.Image]) -> PageRoutingResult:
        if len(images) == 1:
            return self.route_single_page(images[0])
        if self._enable_router_v3:
            decisions = []
            for page_number, image in enumerate(images, start=1):
                lines = self._extract_anchor_lines(image) or []
                decisions.append((page_number, self._multi_signal_router.route(image, lines)))
            standards = [
                item
                for item in decisions
                if item[1].route in {MultiSignalRoute.CMS1500, MultiSignalRoute.UB04}
            ]
            if len(standards) == 1:
                page_number, decision = standards[0]
                is_cms = decision.route is MultiSignalRoute.CMS1500
                template = self._cms_template if is_cms else self._ub_template
                roles = {
                    index: (PageRole.CMS1500_CLAIM_PAGE if is_cms else PageRole.UB_CLAIM_PAGE)
                    if index == page_number
                    else PageRole.ATTACHMENT
                    for index in range(1, len(images) + 1)
                }
                return PageRoutingResult(
                    bundle_type=BundleType.B_CMS1500_BUNDLE if is_cms else BundleType.C_UB_SINGLE,
                    selected_page_number=page_number,
                    template=template,
                    page_roles=roles,
                    page_scores={
                        page_number: PageCandidateScore(
                            page_number,
                            ClassificationMethod.ANCHOR_PHRASE,
                            decision.confidence,
                            decision.reason_codes,
                        )
                    },
                    needs_review=False,
                    reason_codes=decision.reason_codes,
                    canonical_route=decision.route,
                    route_decision=decision,
                )
            if len(standards) > 1:
                return PageRoutingResult(
                    bundle_type=BundleType.UNKNOWN_STRUCTURED,
                    selected_page_number=None,
                    template=None,
                    page_roles={i: PageRole.UNKNOWN for i in range(1, len(images) + 1)},
                    page_scores={},
                    needs_review=True,
                    reason_codes=["STANDARD_MARGIN_INSUFFICIENT"],
                    canonical_route=MultiSignalRoute.UNKNOWN_STRUCTURED,
                )
            aggregate = (
                MultiSignalRoute.OTHER_CLAIM_FORM
                if any(x.route is MultiSignalRoute.OTHER_CLAIM_FORM for _, x in decisions)
                else MultiSignalRoute.UNKNOWN_STRUCTURED
                if any(x.route is MultiSignalRoute.UNKNOWN_STRUCTURED for _, x in decisions)
                else MultiSignalRoute.NON_CLAIM
                if all(x.route is MultiSignalRoute.NON_CLAIM for _, x in decisions)
                else MultiSignalRoute.UNKNOWN_UNSTRUCTURED
            )
            bundle = {
                MultiSignalRoute.OTHER_CLAIM_FORM: BundleType.UNKNOWN_STRUCTURED,
                MultiSignalRoute.UNKNOWN_STRUCTURED: BundleType.UNKNOWN_STRUCTURED,
                MultiSignalRoute.UNKNOWN_UNSTRUCTURED: BundleType.UNKNOWN_UNSTRUCTURED,
                MultiSignalRoute.NON_CLAIM: BundleType.NON_CLAIM,
            }[aggregate]
            return PageRoutingResult(
                bundle_type=bundle,
                selected_page_number=None,
                template=None,
                page_roles={
                    i: (
                        PageRole.UNKNOWN
                        if aggregate is MultiSignalRoute.NON_CLAIM
                        else PageRole.UNSTRUCTURED_CLAIM_PAGE
                    )
                    for i in range(1, len(images) + 1)
                },
                page_scores={},
                needs_review=False,
                reason_codes=[f"{aggregate.value}_CONFIRMED"],
                canonical_route=aggregate,
            )
        return self.route_multipage_bundle(images)
