from evaluation.cost_model import CostRates, calculate_cost


def test_cost_model_prices_measured_compute_and_cloud_calls():
    result = calculate_cost(
        pages=1000,
        cpu_seconds=3600.0,
        gpu_seconds=1800.0,
        cloud_api_cost_usd=2.0,
        rates=CostRates(
            cpu_usd_per_hour=1.0,
            gpu_usd_per_hour=4.0,
            fixed_run_usd=1.0,
        ),
    )

    assert result.cpu_cost_usd == 1.0
    assert result.gpu_cost_usd == 2.0
    assert result.total_cost_usd == 6.0
    assert result.cost_usd_per_page == 0.006
