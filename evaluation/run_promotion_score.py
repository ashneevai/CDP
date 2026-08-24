"""CLI for scoring a frozen external benchmark after independent truth is revealed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.promotion_scorer import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-manifest-id")
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--fully-loaded-cost-usd", type=float)
    args = parser.parse_args()

    report = score(
        predictions_jsonl=args.predictions,
        prediction_freeze_json=args.freeze,
        truth_jsonl=args.truth,
        output_json=args.output,
        runtime_manifest_id=args.runtime_manifest_id,
        corpus_sha256=args.corpus_sha256,
        fully_loaded_cost_usd=args.fully_loaded_cost_usd,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
