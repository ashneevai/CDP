"""Build a deterministic, explicitly untrusted review queue for Azure shadow truth."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation_results/real_eval/page_classification_candidates.json"
OUTPUT = ROOT / "evaluation_results/azure_live_shadow/review_cohort_candidates.json"
REVIEW_QUEUE = ROOT / "evaluation_results/azure_live_shadow/review_queue.json"


def _round_robin(rows: list[dict], limit: int) -> list[dict]:
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["package_id"], r["source_page_id"])):
        groups[row["package_id"]].append(row)
    queues = {key: deque(value) for key, value in groups.items()}
    selected = []
    while len(selected) < limit and queues:
        for key in sorted(queues):
            if len(selected) >= limit:
                break
            selected.append(queues[key].popleft())
            if not queues[key]:
                del queues[key]
    return selected


def build(source: Path = SOURCE, output: Path = OUTPUT, target: int = 200) -> dict:
    records = json.loads(source.read_text("utf-8"))["records"]
    half = target // 2
    selected = _round_robin([r for r in records if r["candidate_class"] == "UB04"], half)
    selected += _round_robin(
        [r for r in records if r["candidate_class"] == "UNKNOWN"], target - len(selected)
    )
    artifact = {
        "candidate_only": True,
        "trusted_ground_truth": False,
        "scoring_eligible": False,
        "selection_method": "PACKAGE_ROUND_ROBIN_BY_CANDIDATE_CLASS",
        "target_pages": target,
        "selected_pages": len(selected),
        "candidate_class_counts": {
            name: sum(r["candidate_class"] == name for r in selected)
            for name in ("CMS1500", "UB04", "UNKNOWN")
        },
        "limitations": ["NO_CONFIRMED_CMS1500_PAGES", "SOURCE_QUALITY_BAND_REQUIRES_HUMAN_REVIEW"],
        "required_review_fields": [
            "reviewed_page_class",
            "source_quality_band",
            "page_complexity",
            "reviewer",
            "review_timestamp",
            "classification_evidence",
        ],
        "records": [
            {
                k: r[k]
                for k in (
                    "package_id",
                    "source_asset_id",
                    "source_page_id",
                    "candidate_class",
                    "classification_confidence",
                    "candidate_record_sha256",
                )
            }
            | {
                "review_state": "REVIEW_REQUIRED",
                "source_quality_band": "UNKNOWN_REQUIRES_REVIEW",
                "page_complexity": "UNKNOWN_REQUIRES_REVIEW",
            }
            for r in selected
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, "utf-8")
    if output.resolve() == OUTPUT.resolve():
        REVIEW_QUEUE.write_text(serialized, "utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
