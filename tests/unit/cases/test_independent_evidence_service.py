from packages.independent_evidence import EvidenceRequest, IndependentEvidenceService
from packages.independent_evidence.contracts import EvidenceOutcome
from packages.independent_evidence.providers import IndependentEvidenceProvider


def test_valid_npi_emits_support_without_deciding_field():
    service = IndependentEvidenceService()
    result = service.collect(EvidenceRequest(
        field_name="billing_provider_npi",
        candidate_value="1234567893",
        document_family="CMS1500",
    ))
    assert any(item.reason_code == "NPI_CHECKSUM_VALID" for item in result.observations)
    assert "NPI_CHECKSUM_VALID" in result.deterministic_evidence
    assert not hasattr(service, "decide")


def test_invalid_npi_is_contradiction_not_support():
    service = IndependentEvidenceService()
    result = service.collect(EvidenceRequest(
        field_name="billing_provider_npi",
        candidate_value="1234567890",
        document_family="CMS1500",
    ))
    assert "NPI_CHECKSUM_INVALID" in result.contradictions
    assert "NPI_CHECKSUM_VALID" not in result.deterministic_evidence


def test_claim_total_reconciliation_is_cross_field_evidence():
    service = IndependentEvidenceService()
    result = service.collect(EvidenceRequest(
        field_name="claim_total",
        candidate_value="30.00",
        document_family="UB04",
        claim_context={"service_line_amounts": ["10.00", "20.00"]},
    ))
    assert "CLAIM_TOTAL_RECONCILED" in result.cross_field_evidence


def test_duplicate_lineage_cannot_count_twice():
    class ProviderA:
        provider_id = "a"
        def supports(self, request): return True
        def collect(self, request):
            from packages.independent_evidence.contracts import (
                EvidenceAuthority, EvidenceClass, EvidenceObservation, EvidenceOutcome,
            )
            return EvidenceObservation(
                provider_id=self.provider_id,
                evidence_class=EvidenceClass.DETERMINISTIC,
                authority=EvidenceAuthority.SUPPORTING,
                outcome=EvidenceOutcome.SUPPORT,
                reason_code="FORMAT_VALID",
                lineage_key="same:fact",
            )

    class ProviderB(ProviderA):
        provider_id = "b"

    result = IndependentEvidenceService((ProviderA(), ProviderB())).collect(EvidenceRequest(
        field_name="member_id", candidate_value="ABC", document_family="CMS1500"
    ))
    assert len(result.observations) == 1
    assert result.deterministic_evidence == frozenset({"FORMAT_VALID"})
