"""Build a hash-addressed weekly shadow and production-readiness artifact."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from packages.production_readiness_gate import ProductionReadinessGate, ReadinessEvidence
from packages.shadow_evaluation import (
    AppendOnlyShadowClaimSink,
    fingerprinted_source_groups,
    qualify_shadow_claims,
)


@dataclass(frozen=True)
class WeeklyGovernanceResult:
    artifact: dict
    exit_code: int


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def generate_weekly_governance(
    ledger: Path,
    *,
    as_of_week: str,
    base_evidence: ReadinessEvidence | None = None,
    prohibited_source_groups: set[str] | None = None,
) -> WeeklyGovernanceResult:
    sink = AppendOnlyShadowClaimSink(ledger, identity_key=b"read-only-ledger-verification")
    observations = sink.observations()
    shadow = qualify_shadow_claims(
        observations, prohibited_source_groups=prohibited_source_groups
    )
    evidence = base_evidence or ReadinessEvidence()
    evidence = evidence.model_copy(update={
        "holdout_frozen": shadow.gates["locked_holdout"],
        "holdout_independent": shadow.gates["source_disjoint"],
        "holdout_documents": shadow.claim_count,
        "holdout_fields": shadow.evaluated_field_decisions,
        "overall_raw_accuracy": shadow.overall_raw_accuracy,
        "critical_accuracy": shadow.critical_accuracy,
        "total_false_accept_rate": shadow.false_accept_rate,
        "critical_false_accept_count": shadow.critical_false_accepts,
        "safe_field_coverage": shadow.safe_field_coverage,
        "accepted_precision": shadow.accepted_precision,
        "claim_stp": shadow.claim_stp,
        "claim_hitl": shadow.claim_hitl,
        "claim_hitl_count": sum(row.shadow_requires_review for row in observations),
        "ocr_only_processing_rate": shadow.ocr_only_processing_rate,
        "llm_escalation_rate": shadow.llm_escalation_rate,
        "accepted_critical_field_decisions": shadow.accepted_critical_field_decisions,
        "critical_accepted_precision": shadow.critical_accepted_precision,
        "wrong_crop_recall": shadow.wrong_crop_recall,
        "maximum_segment_claim_hitl": shadow.maximum_segment_claim_hitl,
        "p95_latency_ms": shadow.p95_latency_ms,
        "cost_per_document_usd": shadow.cost_per_document_usd,
        "runtime_parity_passed": shadow.gates["runtime_parity"],
        "route_governance_passed": shadow.gates["route_governance"],
        "shadow_validation_passed": shadow.status == "QUALIFIED",
    })
    readiness = ProductionReadinessGate.load().evaluate(evidence)
    payload = {
        "schema_version": "weekly-shadow-governance-v1",
        "as_of_week": as_of_week,
        "promotion_authority": False,
        "ledger_sha256": _sha256(ledger),
        "shadow_qualification": shadow.model_dump(mode="json"),
        "readiness_evidence": evidence.model_dump(mode="json"),
        "production_readiness": readiness.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["artifact_sha256"] = sha256(canonical.encode()).hexdigest()
    return WeeklyGovernanceResult(
        artifact=payload,
        exit_code=0 if shadow.status == "QUALIFIED" else 2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--as-of-week", required=True)
    parser.add_argument("--base-evidence", type=Path)
    parser.add_argument(
        "--correction-dataset",
        type=Path,
        help="Directory containing train.jsonl/calibration.jsonl source groups",
    )
    parser.add_argument("--identity-key-env", default="SHADOW_IDENTITY_KEY")
    args = parser.parse_args()
    if not args.ledger.is_file():
        parser.error(
            f"shadow ledger not found: {args.ledger}. "
            "Capture adjudicated claims with scripts/capture_shadow_claim.py first."
        )
    if args.base_evidence and not args.base_evidence.is_file():
        parser.error(
            f"base evidence not found: {args.base_evidence}. "
            "Omit --base-evidence or copy config/shadow_operational_evidence.template.json."
        )
    if args.correction_dataset and not args.correction_dataset.is_dir():
        parser.error(f"correction dataset directory not found: {args.correction_dataset}")
    identity_key = os.environ.get(args.identity_key_env, "").encode()
    if args.correction_dataset and not identity_key:
        parser.error(
            f"{args.identity_key_env} must contain the same non-empty secret used "
            "to capture the shadow ledger when --correction-dataset is supplied"
        )
    base = (
        ReadinessEvidence.model_validate_json(args.base_evidence.read_text(encoding="utf-8"))
        if args.base_evidence else None
    )
    result = generate_weekly_governance(
        args.ledger,
        as_of_week=args.as_of_week,
        base_evidence=base,
        prohibited_source_groups=(
            fingerprinted_source_groups(
                args.correction_dataset,
                {"train", "calibration"},
                identity_key=identity_key,
            )
            if args.correction_dataset else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.artifact, indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
