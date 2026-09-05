from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.production_evidence import load_and_inspect


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify hashes for external artifacts required by production gates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/production_evidence_requirements.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = load_and_inspect(args.config, args.repository_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    if report["status"] != "VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
