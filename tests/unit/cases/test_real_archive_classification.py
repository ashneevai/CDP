import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, TiffImagePlugin

from evaluation.real_archive_classification import (
    Observation,
    ResumeMismatchError,
    build_document_boundaries,
    run,
)
from workers.page_detection.text_extraction import TextLine


def _archive(source: Path, pages: int = 2) -> None:
    source.mkdir()
    images = []
    for _ in range(pages):
        image = Image.new("L", (1000, 1300), 255)
        draw = ImageDraw.Draw(image)
        for y in range(150, 1100, 100):
            draw.line((40, y, 960, y), fill=0, width=2)
        for x in (40, 250, 500, 750, 960):
            draw.line((x, 150, x, 1100), fill=0, width=2)
        images.append(image)
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[45000] = "source-asset"
    tags[45016] = "source-package"
    images[0].save(
        source / "claim.001", format="TIFF", save_all=True, append_images=images[1:], tiffinfo=tags
    )


def _line(text: str, index: int) -> TextLine:
    return TextLine(text, 20, 50 + index * 40, 900, 80 + index * 40, 0.95)


class FakeObserver:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, image, page):
        self.calls += 1
        values = (
            "UB-04",
            "TYPE OF BILL",
            "PATIENT CONTROL",
            "STATEMENT COVERS",
            "PRINCIPAL DIAGNOSIS",
            "REVENUE CODE HCPCS SERVICE DATE UNITS TOTAL CHARGES",
        )
        return Observation(
            tuple(_line(value, i) for i, value in enumerate(values)), 12.0, "fake", "1"
        )


@pytest.mark.asyncio
async def test_full_archive_candidate_run_is_phi_safe_and_candidate_only(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _archive(source)
    observer = FakeObserver()
    result = await run(source, output, "a" * 64, observer)

    assert result["classification"]["processed_pages"] == 2
    assert result["classification"]["counts"] == {"OTHER_CLAIM_FORM": 2}
    assert result["boundaries"]["confirmed_document_count"] == 0
    serialized = (output / "page_classification_candidates.json").read_text("utf-8")
    assert "PATIENT CONTROL" not in serialized
    assert "source-package" not in serialized
    assert '"trusted_ground_truth": false' in serialized
    assert '"recognized_text_persisted": false' in serialized


@pytest.mark.asyncio
async def test_resume_skips_completed_pages_and_rejects_different_source(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _archive(source, 3)
    first = FakeObserver()
    await run(source, output, "b" * 64, first, limit=1)
    second = FakeObserver()
    result = await run(source, output, "b" * 64, second)
    assert first.calls == 1
    assert second.calls == 2
    assert result["classification"]["processed_pages"] == 3
    with pytest.raises(ResumeMismatchError):
        await run(source, output, "c" * 64, FakeObserver())


@pytest.mark.asyncio
async def test_observation_failure_fails_closed_without_error_message(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _archive(source, 1)

    async def fail(image, page):
        raise RuntimeError("patient secret")

    result = await run(source, output, "d" * 64, fail)
    record = result["classification"]["records"][0]
    assert record["candidate_class"] == "UNKNOWN"
    assert record["reason_codes"] == ["OBSERVATION_FAILED"]
    assert "patient secret" not in json.dumps(result)


def test_boundaries_are_candidates_never_confirmed() -> None:
    rows = [
        {
            "package_id": "p",
            "source_asset_id": "a",
            "source_page_id": "p1",
            "source_page_number": 1,
            "source_asset_sequence": 1,
            "candidate_class": "CMS1500",
        },
        {
            "package_id": "p",
            "source_asset_id": "a",
            "source_page_id": "p2",
            "source_page_number": 2,
            "source_asset_sequence": 1,
            "candidate_class": "NON_CLAIM",
        },
    ]
    candidates = build_document_boundaries(rows)
    assert all(row["boundary_state"] == "CANDIDATE" and not row["confirmed"] for row in candidates)
    assert "SEPARATOR_PAGE_CANDIDATE" in candidates[-1]["reason_codes"]


@pytest.mark.asyncio
async def test_visual_mode_processes_full_archive_without_ocr_and_uses_existing_gates(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _archive(source, 3)
    cms = Image.new("L", (1000, 1300), 255)
    draw = ImageDraw.Draw(cms)
    for y in range(150, 1100, 100):
        draw.line((40, y, 960, y), fill=0, width=2)
    for x in (40, 250, 500, 750, 960):
        draw.line((x, 150, x, 1100), fill=0, width=2)
    blank_ub = Image.new("L", (1000, 1300), 255)

    class ForbiddenOCR:
        async def __call__(self, image, page):
            raise AssertionError("visual mode must not execute OCR")

    result = await run(
        source,
        output,
        "e" * 64,
        ForbiddenOCR(),
        mode="visual",
        visual_references={"CMS1500": ("cms@test", cms), "UB04": ("ub@test", blank_ub)},
    )
    assert result["classification"]["processed_pages"] == 3
    assert result["classification"]["counts"] == {"CMS1500": 3}
    assert all(
        row["text_evidence"]["state"] == "NOT_EXECUTED"
        for row in result["classification"]["records"]
    )
    checkpoint = output / ".page_classification_visual_checkpoint.jsonl"
    header = json.loads(checkpoint.read_text("utf-8").splitlines()[0])
    assert header["mode"] == "visual"
