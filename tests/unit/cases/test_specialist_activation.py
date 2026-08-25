from packages.specialist_activation import SpecialistActivationGate, SpecialistMetrics


def _metrics(**overrides):
    base = dict(
        sample_size=50,
        baseline_accuracy=0.90,
        specialist_accuracy=0.97,
        accepted_precision=0.997,
        critical_accepted_precision=0.999,
        critical_false_accepts=0,
        field_hitl_rate=0.10,
        p95_seconds_per_page=3.0,
        cost_usd_per_page=0.02,
    )
    base.update(overrides)
    return SpecialistMetrics(**base)


def test_specialist_gate_passes_only_when_all_requirements_pass():
    decision = SpecialistActivationGate().evaluate(_metrics())
    assert decision.activate is True
    assert decision.reasons == ()


def test_specialist_gate_fails_closed_on_critical_false_accept():
    decision = SpecialistActivationGate().evaluate(_metrics(critical_false_accepts=1))
    assert decision.activate is False
    assert "CRITICAL_FALSE_ACCEPT_PRESENT" in decision.reasons


def test_specialist_gate_requires_measurable_accuracy_gain():
    decision = SpecialistActivationGate().evaluate(
        _metrics(baseline_accuracy=0.96, specialist_accuracy=0.965)
    )
    assert decision.activate is False
    assert "INSUFFICIENT_ACCURACY_GAIN" in decision.reasons
