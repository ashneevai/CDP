import json
import zipfile
from pathlib import Path

from PIL import Image, TiffImagePlugin

from evaluation.closure1000_source_audit import inspect_tiffs, inventory_archive, media_type


def _tiff(path: Path, *, bundle: str = "bundle-1", asset: str = "asset-1") -> None:
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[45000] = asset
    tags[45016] = bundle
    Image.new("1", (8, 8), 1).save(path, format="TIFF", tiffinfo=tags)


def test_archive_inventory_uses_magic_and_detects_duplicates(tmp_path: Path) -> None:
    image_path = tmp_path / "source.001"
    _tiff(image_path)
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(image_path, "Group A/A.001")
        archive.write(image_path, "Group A/A.002")

    inventory = inventory_archive(archive_path)

    assert inventory["files_by_type"] == {"TIFF": 2}
    assert inventory["corrupt_files"] == []
    assert inventory["unsafe_paths"] == []
    assert inventory["duplicates_by_sha256"][0]["count"] == 2


def test_archive_inventory_flags_nested_archives_and_unsafe_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested.zip", b"PK\x03\x04")
        archive.writestr("../escape.txt", b"unsafe")

    inventory = inventory_archive(archive_path)

    assert inventory["nested_archives"] == ["nested.zip"]
    assert inventory["unsafe_paths"] == ["../escape.txt"]


def test_tiff_lineage_uses_embedded_ids_without_persisting_content(tmp_path: Path) -> None:
    group = tmp_path / "Group A"
    group.mkdir()
    _tiff(group / "bundle.001", bundle="package-7", asset="asset-9")

    assets, pages, corrupt = inspect_tiffs(tmp_path)

    assert corrupt == []
    assert assets[0]["bundle_id"] == "package-7"
    assert assets[0]["asset_id"] == "asset-9"
    assert assets[0]["sequence"] == 1
    assert pages[0]["parent_document"] == "asset-9"
    assert len(pages[0]["page_content_sha256"]) == 64
    assert "text" not in json.dumps({"assets": assets, "pages": pages}).lower()


def test_invalid_image_is_reported_without_creating_lineage(tmp_path: Path) -> None:
    invalid = tmp_path / "Group B" / "bad.001"
    invalid.parent.mkdir()
    invalid.write_bytes(b"not a tiff")

    assets, pages, corrupt = inspect_tiffs(tmp_path)

    assert assets == []
    assert pages == []
    assert corrupt[0]["relative_path"] == "Group B/bad.001"


def test_media_type_does_not_trust_disguised_suffix() -> None:
    assert media_type("claim.050", b"II*\x00payload") == ("TIFF", "image/tiff")
