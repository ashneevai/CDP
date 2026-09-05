"""Phase 8.27 source-bundle capture readiness and immutable lineage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from packages.claim_evidence.source_bundle_manifest import (
    AvailabilityStatus,
    CapturedAsset,
    NormalizedPageLineage,
    SourceBundleCaptureRecord,
)

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "evaluation_results/phase8_8c/source_b/observations"
PHASE826 = ROOT / "evaluation_results/phase8_26"
LINEAGE = ROOT / "evaluation/phase8_27_lineage_hashes.json"
OUTPUT = ROOT / "evaluation_results/phase8_27"
VERSION = "phase8.27-source-capture-readiness-v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")


def run(output: Path = OUTPUT) -> dict[str, Any]:
    expected = _json(LINEAGE)
    actual = {name: _sha(PHASE826 / name) for name in expected}
    records = []
    for path in sorted(OBSERVATIONS.glob("SB-*.json")):
        observation = _json(path)
        claim_id = observation["page_id"]
        record = SourceBundleCaptureRecord(
            claim_id=claim_id,
            bundle_id=claim_id,
            raw_bundle=CapturedAsset(status=AvailabilityStatus.UNAVAILABLE),
            raw_documents=(),
            attachment_inventory_status=AvailabilityStatus.UNAVAILABLE,
            normalized_pages=(NormalizedPageLineage(
                page_id=observation["page_id"],
                page_sha256=observation["page_sha256"],
                observation_uri=str(path.relative_to(ROOT)).replace("\\", "/"),
                observation_version=observation["observation_version"],
            ),),
            acquisition_reason_codes=(
                "RAW_SOURCE_BUNDLE_NOT_PROVIDED",
                "SOURCE_DOCUMENT_PAGE_MAPPING_UNAVAILABLE",
                "ATTACHMENT_INVENTORY_NOT_PROVIDED",
            ),
        )
        records.append(record.model_dump(mode="json"))
    raw_available = sum(r["raw_bundle"]["status"] == "AVAILABLE" for r in records)
    attachment_available = sum(r["attachment_inventory_status"] == "AVAILABLE" for r in records)
    linked_pages = sum(len(r["normalized_pages"]) for r in records)
    source_linked_pages = sum(page["source_document_id"] is not None
                              for r in records for page in r["normalized_pages"])
    metrics = {
        "claims_manifested": len(records), "normalized_pages_manifested": linked_pages,
        "raw_bundles_available": raw_available, "raw_documents_available": 0,
        "attachment_inventories_available": attachment_available,
        "pages_with_source_document_lineage": source_linked_pages,
        "claims_blocked_on_acquisition": len(records) - raw_available,
        "phase8_26_lineage_unchanged": actual == expected,
    }
    acquisition_requests = [{
        "claim_id": r["claim_id"], "bundle_id": r["bundle_id"],
        "required_assets": ["original raw bundle", "document inventory", "page-to-document map",
                            "attachment inventory with semantic document roles"],
        "required_metadata": ["sha256", "mime_type", "page_count", "source_document_id",
                              "source_page_index", "acquisition authorization"],
        "status": "AWAITING_AUTHORIZED_SOURCE_CAPTURE",
    } for r in records]
    gates = {
        "all_source_b_claims_manifested": len(records) == 20,
        "normalized_observations_linked_without_rehashing": linked_pages == 20,
        "no_raw_asset_identity_fabricated": all(
            r["raw_bundle"]["asset_uri"] is None and r["raw_bundle"]["sha256"] is None
            for r in records
        ),
        "phase8_26_frozen": actual == expected,
        "ready_for_definitive_phase8_26_rerun": raw_available == len(records)
                                                   and attachment_available == len(records),
    }
    verdict = "PASS" if all(gates.values()) else "NEEDS_MORE_DATA"
    _write(output / "source_bundle_capture_manifest.json", {"version": VERSION, "records": records})
    _write(output / "acquisition_request.json", {"version": VERSION, "requests": acquisition_requests})
    _write(output / "capture_gap_metrics.json", metrics)
    report = {"phase": "8.27", "version": VERSION, "metrics": metrics,
              "acceptance_gates": gates, "verdict": verdict,
              "next_action": "Provide authorized raw Source-B bundles and attachment manifests; then rerun Phase 8.26."}
    _write(output / "comparative_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
