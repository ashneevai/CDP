import json
import zipfile

from PIL import Image

from evaluation.build_external_corpus_manifest import build_manifest
from evaluation.external_corpus_runner import build_truth_blind_dataset


def _tiff_bytes(tmp_path, name="source.tif"):
    path = tmp_path / name
    Image.new("L", (8, 8), 255).save(path, format="TIFF")
    return path.read_bytes()


def test_manifest_accepts_tiff_with_numeric_page_suffix(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "claim.001").write_bytes(_tiff_bytes(tmp_path))

    summary = build_manifest(corpus_root=corpus, output_jsonl=tmp_path / "manifest.jsonl")

    assert summary["pages"] == 1


def test_truth_blind_dataset_orders_zip_members_by_filename(tmp_path):
    archive_path = tmp_path / "claims.zip"
    payload = _tiff_bytes(tmp_path)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Group A/z.001", payload)
        archive.writestr("Group A/a.001", payload)

    dataset = tmp_path / "dataset"
    build_truth_blind_dataset(archive_path, dataset)
    rows = [json.loads(line) for line in
            (dataset / "metadata" / "document_metadata.jsonl").read_text().splitlines()]

    assert [row["path"] for row in rows] == ["pages/0001.tif", "pages/0002.tif"]


def test_production_holdout_runner_imports_with_current_geometry_api():
    from evaluation import run_production_holdout_v2

    assert callable(run_production_holdout_v2.infer)
