from types import SimpleNamespace

from PIL import Image

from evaluation import run_production_holdout_v2 as runner
from packages.document_taxonomy.taxonomy import DocumentClass
from packages.extraction_geometry import (
    ExtractionGeometryDecision,
    ExtractionGeometryMode,
    FormIdentityDecision,
    FormIdentityStatus,
)
from packages.processing_routes.contracts import ProcessingRoute
from packages.standard_form_verification.contracts import (
    FormIdentityVerification,
    StandardFormStatus,
    StandardFormVerification,
)


def test_unknown_or_unauthorized_routes_cannot_receive_claim_stp():
    assert not runner._eligible_for_claim_decision(
        standard_processing_used=False, route="UNKNOWN_STRUCTURED"
    )
    assert not runner._eligible_for_claim_decision(
        standard_processing_used=False, route="CMS1500"
    )
    assert runner._eligible_for_claim_decision(
        standard_processing_used=True, route="CMS1500"
    )


def test_holdout_uses_strict_router_and_current_processing_service(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    metadata = dataset / "metadata"
    pages = dataset / "pages"
    metadata.mkdir(parents=True)
    pages.mkdir()
    Image.new("L", (32, 32), 255).save(pages / "page.tif")
    (metadata / "document_metadata.jsonl").write_text(
        '{"document_id":"page-1","path":"pages/page.tif"}\n', encoding="utf-8"
    )

    template = SimpleNamespace(
        template_id="cms1500",
        version="02-12",
        reference_dimensions=SimpleNamespace(width_px=32, height_px=32),
    )

    class Registry:
        def get(self, template_id, _version):
            return template if template_id == "cms1500" else SimpleNamespace(
                template_id="ub04", version="2014"
            )

        def load_reference_image(self, _template):
            return Image.new("L", (32, 32), 255)

    monkeypatch.setattr(
        runner.TemplateRegistry, "load_from_directory", lambda _path: Registry()
    )
    monkeypatch.setattr(runner, "_prepare", lambda _path: Image.new("L", (32, 32), 255))
    router_options = {}

    class Router:
        def __init__(self, *_args, **kwargs):
            router_options.update(kwargs)

        def route_single_page(self, _image):
            return SimpleNamespace(
                template=template,
                canonical_route=SimpleNamespace(value="CMS1500"),
                route_decision=SimpleNamespace(confidence=0.99),
                reason_codes=["STRICT_IDENTITY_CONFIRMED"],
            )

    monkeypatch.setattr(runner, "PageRoutingService", Router)
    monkeypatch.setattr(runner, "evidence_from_router_features", lambda *_a, **_k: object())

    verification = StandardFormVerification(
        candidate_family=DocumentClass.CMS1500,
        status=StandardFormStatus.VERIFIED,
        verification_score=0.99,
        template_version="02-12",
        eligible_for_fixed_extractor=True,
        form_identity=FormIdentityVerification(
            status=StandardFormStatus.VERIFIED,
            family=DocumentClass.CMS1500,
            authorization_path="STRICT_IDENTITY_CHAIN",
        ),
    )
    routing_decision = SimpleNamespace(
        processing_route=ProcessingRoute.CMS_STANDARD_EXTRACTOR,
        standard_verification=verification,
        classification=SimpleNamespace(model_dump=lambda **_kwargs: {"family": "CMS1500"}),
    )
    monkeypatch.setattr(
        runner,
        "DocumentRoutingDecisionService",
        lambda: SimpleNamespace(decide_nomination=lambda **_kwargs: routing_decision),
    )

    identity = FormIdentityDecision(
        family=DocumentClass.CMS1500,
        status=FormIdentityStatus.VERIFIED,
        score=0.99,
        template_version="02-12",
    )
    geometry = ExtractionGeometryDecision(
        mode=ExtractionGeometryMode.ANCHOR_RELATIVE,
        form_identity=identity,
        template_id="cms1500",
        template_version="02-12",
        structural_confidence=0.9,
    )
    processed = SimpleNamespace(
        fields=[],
        geometry=geometry,
        diagnostics=SimpleNamespace(full_page_ocr_calls=1, regional_ocr_calls=0),
    )
    monkeypatch.setattr(
        runner,
        "StandardFormProcessingService",
        lambda *_args, **_kwargs: SimpleNamespace(process=lambda *_a, **_k: processed),
    )
    monkeypatch.setattr(
        runner.ClaimDecisionService,
        "load",
        lambda: SimpleNamespace(
            decide=lambda _context: SimpleNamespace(
                disposition=SimpleNamespace(value="REVIEW_REQUIRED"),
                model_dump=lambda **_kwargs: {"disposition": "REVIEW_REQUIRED"},
            )
        ),
    )

    predictions = runner.infer(dataset, tmp_path / "output")

    assert router_options["enable_router_v3"] is True
    assert predictions[0]["processing_route"] == "CMS_STANDARD_EXTRACTOR"
    assert predictions[0]["standard_form_verification"]["status"] == "VERIFIED"
    assert predictions[0]["counters"] == {
        "rapidocr_full_page_calls": 1,
        "rapidocr_regional_calls": 0,
    }
