"""Canonical runtime identity for reproducible CDP decisions.

A RuntimeManifest is the immutable identity of the components and policies that
can affect extraction or decision outcomes. Production and evaluation must use
the same manifest before promotion claims can be made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    pipeline_version: str
    ocr_runtime: str
    preprocessing_version: str
    router_version: str
    taxonomy_version: str
    cms_template_version: str
    cms_field_graph_version: str
    ub_template_version: str
    ub_field_registry_version: str
    localization_version: str
    candidate_scoring_version: str
    normalization_version: str
    evidence_policy_version: str
    claim_evidence_version: str
    claim_decision_version: str
    route_registry_version: str
    vlm_version: str = "disabled"
    threshold_set_version: str = "default"

    def canonical_payload(self) -> dict[str, str]:
        return dict(sorted(asdict(self).items()))

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_compatible(self, other: "RuntimeManifest") -> None:
        if self.manifest_id == other.manifest_id:
            return
        expected = self.canonical_payload()
        actual = other.canonical_payload()
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        raise ValueError(f"RUNTIME_MANIFEST_MISMATCH:{json.dumps(differences, sort_keys=True)}")


def manifest_from_mapping(values: Mapping[str, object]) -> RuntimeManifest:
    """Construct from explicit config; missing required identity fails closed."""
    return RuntimeManifest(**{key: str(value) for key, value in values.items()})
