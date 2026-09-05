from PIL import Image, ImageDraw

from packages.document_routing import MultiSignalRouter
from packages.domain.enums import BundleType
from packages.templates import TemplateRegistry
from packages.templates.registry import DEFAULT_TEMPLATE_DIR
from workers.page_detection.router import PageRoutingService
from workers.page_detection.text_extraction import TextLine


class FixedOCR:
    def __init__(self, lines):
        self.lines = lines

    def extract(self, image):
        return self.lines

    def extract_region(self, image, x0, y0, x1, y1):
        return self.lines


def _page():
    image = Image.new("L", (1000, 1300), 255)
    draw = ImageDraw.Draw(image)
    for y in range(150, 1100, 80):
        draw.line((30, y, 970, y), fill=0, width=2)
    for x in (30, 220, 480, 720, 970):
        draw.line((x, 150, x, 1100), fill=0, width=2)
    return image


def test_runtime_and_evaluation_use_identical_route_decision():
    lines = [
        TextLine("UB-04", 700, 20, 850, 50, 0.95),
        TextLine("TYPE OF BILL", 760, 70, 960, 100, 0.95),
        TextLine("STATEMENT COVERS", 650, 150, 900, 180, 0.95),
        TextLine("REVENUE CODE", 40, 420, 220, 450, 0.95),
    ]
    page = _page()
    canonical = MultiSignalRouter.load()
    evaluation = canonical.route(page, lines)
    registry = TemplateRegistry.load_from_directory(DEFAULT_TEMPLATE_DIR)
    runtime = PageRoutingService(
        registry.get("cms1500", "02-12"),
        registry.get("ub04", "2014"),
        FixedOCR(lines),
        multi_signal_router=canonical,
        enable_router_v3=True,
    ).route([page])
    assert runtime.bundle_type is BundleType.C_UB_SINGLE
    assert runtime.route_decision == evaluation


def test_other_claim_form_is_preserved_end_to_end():
    lines = [
        TextLine("PATIENT MEMBER PROVIDER", 20, 100, 500, 130, 0.95),
        TextLine("DIAGNOSIS PROCEDURE SERVICE DATE NPI CHARGE CLAIM", 20, 200, 800, 230, 0.95),
    ]
    page = _page()
    registry = TemplateRegistry.load_from_directory(DEFAULT_TEMPLATE_DIR)
    runtime = PageRoutingService(
        registry.get("cms1500", "02-12"),
        registry.get("ub04", "2014"),
        FixedOCR(lines),
        enable_router_v3=True,
    ).route([page])
    assert runtime.bundle_type is BundleType.UNKNOWN_STRUCTURED
    assert runtime.canonical_route.value == "OTHER_CLAIM_FORM"
    assert not runtime.route_decision.localization_allowed
