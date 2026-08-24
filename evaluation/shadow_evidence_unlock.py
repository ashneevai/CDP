"""Estimate evidence-coverage opportunities without changing production decisions.

Input raw predictions may contain PHI and must remain local. Output contains only
aggregate counts by field/evidence reason and is safe to review in Git when field
names themselves are non-sensitive schema identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from packages.independent_evidence import EvidenceRequest, IndependentEvidenceService
from packages.independent_evidence.contracts import EvidenceOutcome


ACCEPTED = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED", "HUMAN_CONFIRMED"}


def analyze(predictions_jsonl: Path) -> dict:
    service = IndependentEvidenceService()
    reviewed = Counter()
    support = Counter()
    contradictions = Counter()
    support_reasons: dict[str, Counter] = defaultdict(Counter)
    contradiction_reasons: dict[str, Counter] = defaultdict(Counter)

    with predictions_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prediction = json.loads(line)
            document_family = str(prediction.get("schema") or prediction.get("route") or "UNKNOWN")
            fields = prediction.get("fields") or {}
            claim_context = {
                name: payload.get("value")
                for name, payload in fields.items()
                if isinstance(payload, dict) and payload.get("value") is not None
            }
            service_line_amounts = []
            table = prediction.get("table") or {}
            for row in table.get("rows", []) if isinstance(table, dict) else []:
                for key in ("charges", "charge_amount", "total_charges"):
                    if isinstance(row, dict) and row.get(key) is not None:
                        service_line_amounts.append(row[key])
            if service_line_amounts:
                claim_context["service_line_amounts"] = service_line_amounts

            for field_name, payload in fields.items():
                if not isinstance(payload, dict):
                    continue
                decision = payload.get("decision") or {}
                disposition = decision.get("disposition")
                if disposition in ACCEPTED:
                    continue
                reviewed[field_name] += 1
                enrichment = service.collect(EvidenceRequest(
                    field_name=field_name,
                    candidate_value=payload.get("value"),
                    document_family=document_family,
                    claim_context=claim_context,
                ))
                if any(item.outcome is EvidenceOutcome.SUPPORT for item in enrichment.observations):
                    support[field_name] += 1
                if enrichment.contradictions:
                    contradictions[field_name] += 1
                for item in enrichment.observations:
                    if item.outcome is EvidenceOutcome.SUPPORT:
                        support_reasons[field_name][item.reason_code] += 1
                    elif item.outcome is EvidenceOutcome.CONTRADICT:
                        contradiction_reasons[field_name][item.reason_code] += 1

    potential = []
    for field_name, count in reviewed.most_common():
        supported = support[field_name]
        potential.append({
            "field_name": field_name,
            "reviewed": count,
            "independent_support_available": supported,
            "support_rate": supported / count if count else 0,
            "contradictions": contradictions[field_name],
            "support_reasons": dict(support_reasons[field_name]),
            "contradiction_reasons": dict(contradiction_reasons[field_name]),
        })

    return {
        "qualification": {
            "shadow_only": True,
            "changes_production_disposition": False,
            "accuracy_claimed": False,
            "stp_unlock_claimed": False,
        },
        "reviewed_fields": sum(reviewed.values()),
        "fields_with_independent_support_signal": sum(support.values()),
        "fields_with_contradiction_signal": sum(contradictions.values()),
        "by_field": potential,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
