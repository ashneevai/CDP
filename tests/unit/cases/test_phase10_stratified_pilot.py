import json
from pathlib import Path

from evaluation.phase10_stratified_pilot import DEFAULT_ALLOCATION, select


def _write_manifest(path: Path) -> None:
    rows = []
    counts = {"Group A": 30, "Group B": 15, "Group C": 12, "Group D": 10}
    for group, count in counts.items():
        for index in range(count):
            rows.append({
                "document_id": f"{group}-{index}",
                "path": f"/tmp/{group}-{index}.tif",
                "sha256": "a" * 64,
                "group": group,
                "package_id": f"{group}-pkg-{index // 2}",
            })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_pilot_is_deterministic_and_group_balanced(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    first = select(manifest, seed=20260824)
    second = select(manifest, seed=20260824)
    assert [row["document_id"] for row in first] == [row["document_id"] for row in second]
    assert len(first) == 50
    for group, expected in DEFAULT_ALLOCATION.items():
        assert sum(row["group"] == group for row in first) == expected
