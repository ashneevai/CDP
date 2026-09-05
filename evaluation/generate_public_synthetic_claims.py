"""Generate non-PHI claim-like OCR fixtures from public CMS field specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SOURCES = [
    "https://www.cms.gov/medicare/billing/electronicbillingeditrans/1500",
    "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c26pdf.pdf",
]
CONDITIONS = ["clean_scan", "fax", "low_contrast", "rotation", "skew", "cropped_edges", "poor_dpi", "handwriting"]
NAMES = ["TEST ALPHA", "SAMPLE BRAVO", "DEMO CHARLIE", "FIXTURE DELTA", "MOCK ECHO"]


def _font(size: int, italic: bool = False):
    names = [
        "C:/Windows/Fonts/ariali.ttf" if italic else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans-Oblique.ttf" if italic else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
        if italic
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf"
        if italic
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _valid_npi(seed: int) -> str:
    first = f"19999{seed % 10000:04d}"
    for check in range(10):
        candidate = first + str(check)
        digits = [int(char) for char in "80840" + candidate]
        total = 0
        for position, digit in enumerate(reversed(digits)):
            if position % 2:
                digit = digit * 2 - (9 if digit * 2 > 9 else 0)
            total += digit
        if total % 10 == 0:
            return candidate
    raise AssertionError("unable to generate NPI checksum")


def _fields(family: str, index: int) -> dict[str, str]:
    amount = f"{100 + index * 7}.{index % 100:02d}"
    common = {
        "patient_name": NAMES[index % len(NAMES)], "patient_dob": f"01{index % 28 + 1:02d}19{70 + index % 25:02d}",
        "provider_npi": _valid_npi(index), "total_charge": amount,
    }
    if family == "CMS1500":
        return {**common, "insured_id_number": f"SYN{index:07d}", "diagnosis_code_1": "Z0000"}
    return {**common, "type_of_bill": "0117", "principal_diagnosis": "Z0000",
            "federal_tax_no": f"99{index:07d}", "total_charges": amount}


def _template(family: str) -> dict:
    name = "cms1500_v02_12.yaml" if family == "CMS1500" else "ub04_v2014.yaml"
    return yaml.safe_load((Path("config/templates") / name).read_text("utf-8"))


def _intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _value_layout(value: str, region: dict, italic: bool) -> tuple[ImageFont.FreeTypeFont, tuple[int, int]]:
    """Retain the frozen layout when it fits; shrink only clipped fixtures."""
    size = 24
    font = _font(size, italic=italic)
    x, y = region["x0"] + 7, region["y0"] + 7
    left, top, right, bottom = font.getbbox(value)
    if x + right <= region["x1"] - 2 and y + bottom <= region["y1"] - 2:
        return font, (x, y)
    max_width, max_height = region["x1"] - region["x0"] - 10, region["y1"] - region["y0"] - 4
    while size > 10:
        font = _font(size, italic=italic)
        left, top, right, bottom = font.getbbox(value)
        if right - left <= max_width and bottom - top <= max_height:
            break
        size -= 1
    glyph_height = bottom - top
    return font, (
        region["x0"] + 5 - left,
        region["y0"] + (region["y1"] - region["y0"] - glyph_height) // 2 - top,
    )


def _blank_form(family: str, spec: dict, populated_fields: set[str] | None = None) -> Image.Image:
    dims = spec["reference_dimensions"]
    image = Image.new("RGB", (dims["width_px"], dims["height_px"]), "white")
    draw = ImageDraw.Draw(image)
    color = (180, 75, 75) if family == "CMS1500" else (80, 80, 80)
    draw.rectangle((18, 18, image.width - 18, image.height - 18), outline=color, width=2)
    title = "SYNTHETIC CMS-1500 TEST FIXTURE" if family == "CMS1500" else "SYNTHETIC UB-04 TEST FIXTURE"
    draw.text((40, 35), title, fill=color, font=_font(28))
    draw.text((40, 75), "NOT A REAL CLAIM - NO PHI", fill=(180, 0, 0), font=_font(20))
    populated = {
        item["field_name"]: (item["x0"], item["y0"], item["x1"], item["y1"])
        for item in spec["field_regions"] if item["field_name"] in (populated_fields or set())
    }
    label_font = _font(10)
    for field in spec["field_regions"]:
        box = (field["x0"], field["y0"], field["x1"], field["y1"])
        overlaps_other_data = field["field_name"] not in populated and any(
            _intersects(box, data_box) for data_box in populated.values()
        )
        if not overlaps_other_data:
            draw.rectangle(box, outline=color, width=1)
        position = (field["x0"] + 2, max(100, field["y0"] - 13))
        label_box = draw.textbbox(position, field["field_name"], font=label_font)
        # A form label may sit above its box, but it must never overwrite a
        # populated data ROI. The prior generator violated this invariant and
        # created 201 benchmark errors that were not production OCR failures.
        if not overlaps_other_data and not any(_intersects(label_box, data_box) for data_box in populated.values()):
            draw.text(position, field["field_name"], fill=color, font=label_font)
    service = spec.get("service_line_region")
    if service:
        draw.rectangle((service["table_x0"], service["table_y0"], service["table_x1"], service["table_y1"]), outline=color)
    return image


def _render(family: str, index: int, condition: str) -> tuple[Image.Image, dict, dict]:
    spec = _template(family)
    values = _fields(family, index)
    image = _blank_form(family, spec, set(values))
    draw = ImageDraw.Draw(image)
    regions = {field["field_name"]: field for field in spec["field_regions"]}
    crops = {}
    for name, value in values.items():
        region = regions.get(name)
        if not region:
            continue
        font, position = _value_layout(value, region, condition == "handwriting")
        draw.text(position, value, fill="black", font=font)
        crops[name] = [region["x0"], region["y0"], region["x1"], region["y1"]]
    if condition == "fax":
        image = image.convert("L").point(lambda p: 255 if p > 165 else 0).convert("RGB").filter(ImageFilter.GaussianBlur(.35))
    elif condition == "low_contrast":
        image = ImageEnhance.Contrast(image).enhance(.38)
    elif condition == "rotation":
        image = image.rotate(1.2, resample=Image.Resampling.BICUBIC, fillcolor="white")
    elif condition == "skew":
        image = image.transform(image.size, Image.Transform.AFFINE, (1, .015, -16, 0, 1, 0), fillcolor="white")
    elif condition == "cropped_edges":
        image = image.crop((12, 8, image.width - 10, image.height - 8)).resize(image.size)
    elif condition == "poor_dpi":
        image = image.resize((image.width // 2, image.height // 2)).resize(image.size)
    return image, values, crops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation_data/synthetic_public_v1"))
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()
    rng = random.Random(20260822)
    images = args.output / "images"; images.mkdir(parents=True, exist_ok=True)
    crop_root = args.output / "crops"; crop_root.mkdir(parents=True, exist_ok=True)
    truth_docs, manifest, assets = [], {}, []
    for index in range(args.count):
        family = "CMS1500" if index % 2 == 0 else "UB04"
        condition = CONDITIONS[index % len(CONDITIONS)]
        document_id = f"SYN-{index + 1:04d}"
        image, values, crop_boxes = _render(family, index + 1, condition)
        if index % 19 == 0:
            image = ImageEnhance.Brightness(image).enhance(.92 + rng.random() * .08)
        filename = f"images/{document_id}.png"
        image.save(args.output / filename, optimize=True)
        doc_crop_dir = crop_root / document_id; doc_crop_dir.mkdir(exist_ok=True)
        for name, box in crop_boxes.items():
            image.crop(box).save(doc_crop_dir / f"{name}.png", optimize=True)
        fields = [{"field_name": name, "expected_raw": value, "expected_normalized": None,
                   "required": True, "critical": name in {"patient_name", "provider_npi", "total_charge", "total_charges"}}
                  for name, value in values.items() if name in crop_boxes]
        truth_doc = {"document_id": document_id, "file_name": filename, "form_type": family,
                     "image_quality_bucket": condition, "split": "holdout", "fields": fields}
        truth_docs.append(truth_doc)
        manifest[document_id] = {"file_name": filename, "form_type": family, "page_number": 1,
                                 "condition": condition, "crop_boxes": crop_boxes}
        raw = (args.output / filename).read_bytes()
        truth_hash = hashlib.sha256(json.dumps(truth_doc, sort_keys=True).encode()).hexdigest()
        assets.append({"asset_id": document_id, "source_id": "public-spec-synthetic-v1",
                       "document_sha256": hashlib.sha256(raw).hexdigest(), "truth_sha256": truth_hash,
                       "document_family": family, "conditions": [condition], "synthetic": True})
    (args.output / "ground_truth.json").write_text(json.dumps({"schema_version": "1.0", "documents": truth_docs}, indent=2), "utf-8")
    (args.output / "document_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    (args.output / "asset_inventory.json").write_text(json.dumps(assets, indent=2), "utf-8")
    (args.output / "provenance.json").write_text(json.dumps({"synthetic": True, "contains_phi": False,
        "seed": 20260822, "sources": SOURCES, "license_note": "Schema guidance only; artwork generated locally.",
        "holdout_qualification": False,
        "generator_contract": "label-safe-populated-roi-v2"}, indent=2), "utf-8")
    print(json.dumps({"documents": len(truth_docs), "CMS1500": args.count // 2,
                      "UB04": args.count // 2, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
