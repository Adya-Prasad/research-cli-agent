from decimal import Decimal

from research_agent.evaluation.models import UsageMetrics
from research_agent.evaluation.pricing import (
    PriceCard,
    estimate_nominal_cost_microusd,
)


def test_cost_uses_uncached_and_cached_input_rates() -> None:
    card = PriceCard(
        model="gemini-test",
        effective_from="2026-01-01",
        input_usd_per_million=Decimal("1.00"),
        cached_input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("2.00"),
    )
    usage = UsageMetrics(
        model_calls=1,
        tool_calls=0,
        input_tokens=1_000,
        cached_input_tokens=200,
        output_tokens=100,
        nominal_cost_microusd=None,
    )

    # 800 * 1.00 + 200 * 0.10 + 100 * 2.00
    assert estimate_nominal_cost_microusd(
        usage,
        card,
    ) == 1_020