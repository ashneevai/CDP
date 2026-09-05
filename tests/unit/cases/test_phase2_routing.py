from packages.policy_engine.contracts import DecisionContext, PolicyAction
from packages.policy_engine.engine import AdaptivePolicyEngine


def context(field_name: str, *attempts: PolicyAction) -> DecisionContext:
    return DecisionContext(
        document_type="CMS1500", field_name=field_name, criticality="critical",
        previous_attempts={PolicyAction.RAPIDOCR, *attempts},
        remaining_budget=1, remaining_sla=10,
    )


def test_secondary_ocr_is_selected_by_field_family() -> None:
    policy = AdaptivePolicyEngine.load()
    for field in (
        "patient_dob", "service_charge", "procedure_code",
        "patient_name", "patient_address",
    ):
        assert policy.decide(context(field)).action is PolicyAction.RETRY_PREPROCESSING
        assert policy.decide(
            context(field, PolicyAction.RETRY_PREPROCESSING)
        ).action is PolicyAction.PADDLEOCR
        assert policy.decide(
            context(field, PolicyAction.RETRY_PREPROCESSING, PolicyAction.PADDLEOCR)
        ).action is PolicyAction.TESSERACT
