import pytest

from packages.runtime_manifest import RuntimeManifest, manifest_from_mapping


def _values(**overrides):
    values = {
        "pipeline_version": "test",
        "ocr_runtime": "rapidocr@1",
        "preprocessing_version": "prep@1",
        "router_version": "router@1",
        "taxonomy_version": "taxonomy@1",
        "cms_template_version": "cms1500@02-12",
        "cms_field_graph_version": "cms1500_v1",
        "ub_template_version": "ub04@2014",
        "ub_field_registry_version": "ub04_v1",
        "localization_version": "loc@1",
        "candidate_scoring_version": "candidate@1",
        "normalization_version": "normalization@1",
        "evidence_policy_version": "evidence-policy-v4-dependency-aware",
        "claim_evidence_version": "claim-evidence-v1",
        "claim_decision_version": "claim-decision-v1",
        "route_registry_version": "runtime",
    }
    values.update(overrides)
    return values


def test_manifest_id_is_deterministic():
    first = manifest_from_mapping(_values())
    second = manifest_from_mapping(dict(reversed(list(_values().items()))))
    assert first.manifest_id == second.manifest_id


def test_manifest_rejects_missing_required_identity():
    values = _values()
    values.pop("evidence_policy_version")
    with pytest.raises(ValueError, match="RUNTIME_MANIFEST_MISSING:evidence_policy_version"):
        manifest_from_mapping(values)


def test_manifest_rejects_unknown_identity_key():
    with pytest.raises(ValueError, match="RUNTIME_MANIFEST_UNKNOWN:surprise"):
        manifest_from_mapping(_values(surprise="x"))


def test_manifest_detects_runtime_evaluation_policy_drift():
    runtime = manifest_from_mapping(_values())
    evaluation = manifest_from_mapping(
        _values(evidence_policy_version="evidence-policy-v4-dependency-aware-balanced")
    )
    with pytest.raises(ValueError, match="RUNTIME_MANIFEST_MISMATCH"):
        runtime.assert_compatible(evaluation)


def test_manifest_accepts_identical_runtime_identity():
    runtime = RuntimeManifest(**_values())
    evaluation = RuntimeManifest(**_values())
    runtime.assert_compatible(evaluation)
