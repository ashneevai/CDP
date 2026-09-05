"""Validate repository structure and dependency direction.

This check intentionally uses only the standard library so it can run before
the project dependencies are installed. It protects the runtime layers from
evaluation-only code and prevents deployable applications from reaching into
worker implementations.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("apps", "packages", "workers")
FORBIDDEN_IMPORT_PREFIXES = {
    "packages": ("apps", "workers", "evaluation"),
    "apps": ("workers", "evaluation"),
    "workers": ("evaluation",),
}
FORBIDDEN_TRACKED_PARTS = {
    "dataset_raw",
    "evaluation_data",
    "evaluation_results",
    "node_modules",
    "__pycache__",
}
ALLOWED_PRIVATE_DOCUMENTATION = {"evaluation_data/README.md"}
ALLOWED_GOVERNED_RESULT_PREFIXES = {
    "evaluation_results/phase7a13/",
    "evaluation_results/phase7a14/",
    "evaluation_results/phase7a14b/",
    # These phases contain only governed measurements from the synthetic
    # engineering golden packs; operator/runtime outputs remain forbidden.
    "evaluation_results/phase8_5/",
    "evaluation_results/phase8_6/",
    "evaluation_results/phase8_7/",
    "evaluation_results/phase8_8/",
    "evaluation_results/phase8_21a/",
    "evaluation_results/phase8_22/",
    "evaluation_results/phase8_23/",
    "evaluation_results/phase8_24/",
    "evaluation_results/phase8_25/",
    "evaluation_results/phase8_26/",
    "evaluation_results/phase8_27/",
    "evaluation_results/phase9a/",
    "evaluation_results/phase9b/",
    "evaluation_results/phase9c/",
    "evaluation_results/phase9d/",
    "evaluation_results/phase9e/",
    "evaluation_results/closure1000/",
    "evaluation_results/closure/",
    "evaluation_results/real_eval/",
    "evaluation_results/azure_llm_shadow/",
    "evaluation_results/azure_live_shadow/",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def dependency_errors() -> list[str]:
    errors: list[str] = []
    for layer, forbidden in FORBIDDEN_IMPORT_PREFIXES.items():
        for path in (ROOT / layer).rglob("*.py"):
            for imported in _imports(path):
                if imported.split(".", 1)[0] in forbidden:
                    relative = path.relative_to(ROOT).as_posix()
                    errors.append(f"{relative}: {layer} must not import {imported}")
    return errors


def tracked_artifact_errors() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    errors: list[str] = []
    for tracked in result.stdout.splitlines():
        if tracked in ALLOWED_PRIVATE_DOCUMENTATION:
            continue
        if any(tracked.startswith(prefix) for prefix in ALLOWED_GOVERNED_RESULT_PREFIXES):
            continue
        parts = set(Path(tracked).parts)
        if parts & FORBIDDEN_TRACKED_PARTS:
            errors.append(f"tracked runtime/private artifact: {tracked}")
        if (
            any(part.startswith(".env") for part in Path(tracked).parts)
            and tracked != ".env.example"
        ):
            errors.append(f"tracked environment file: {tracked}")
    return errors


def main() -> int:
    errors = dependency_errors() + tracked_artifact_errors()
    if errors:
        print("Architecture validation failed:")
        for error in sorted(errors):
            print(f"  - {error}")
        return 1
    print("Architecture validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
