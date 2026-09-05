from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from packages.hitl_reduction import (
    BlindReviewSubmission,
    GovernedFieldLabel,
    HITLReductionInput,
    HITLReductionService,
    build_review_assignments,
    compile_review_submissions,
    verify_review_assignment,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_outputs(output: Path, results: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in results.items():
        if name == "governed_labels":
            (output / "governed_labels.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload),
                encoding="utf-8",
            )
            continue
        (output / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and score leakage-resistant HITL reduction evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    assign = subparsers.add_parser("assign")
    assign.add_argument("--queue", type=Path, required=True)
    assign.add_argument("--reviewer", action="append", dest="reviewers", required=True)
    assign.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-assignments")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    compile_reviews = subparsers.add_parser("compile-reviews")
    compile_reviews.add_argument("--queue", type=Path, required=True)
    compile_reviews.add_argument("--manifest", type=Path, required=True)
    compile_reviews.add_argument("--reviews", type=Path, action="append", required=True)
    compile_reviews.add_argument("--adjudications", type=Path, action="append", default=[])
    compile_reviews.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--sealed", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    service = HITLReductionService()
    if args.command == "prepare":
        result = service.prepare(HITLReductionInput.model_validate(_read_json(args.input)))
    elif args.command == "assign":
        result = build_review_assignments(_read_json(args.queue), args.reviewers)
    elif args.command == "verify-assignments":
        result = {"review_assignment_verification": verify_review_assignment(
            _read_json(args.manifest)
        )}
    elif args.command == "compile-reviews":
        reviews = [
            BlindReviewSubmission.model_validate(row)
            for path in args.reviews
            for row in _read_jsonl(path)
        ]
        adjudications = [
            BlindReviewSubmission.model_validate(row)
            for path in args.adjudications
            for row in _read_jsonl(path)
        ]
        result = compile_review_submissions(
            _read_json(args.queue),
            _read_json(args.manifest),
            reviews,
            adjudications,
        )
    else:
        labels = [GovernedFieldLabel.model_validate(item) for item in _read_jsonl(args.labels)]
        result = service.score(_read_json(args.sealed), labels)
    _write_outputs(args.output, result)


if __name__ == "__main__":
    main()
