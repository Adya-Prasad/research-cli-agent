"""Versioned nominal-cost calculation."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from pydantic import BaseModel, Field

from research_agent.evaluation.models import UsageMetrics


class PriceCard(BaseModel):
    """List prices captured at a specific effective date."""

    model_config = {"frozen": True}

    model: str = Field(min_length=1)
    effective_from: str = Field(min_length=1)
    input_usd_per_million: Decimal = Field(ge=0)
    cached_input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)


def load_price_card(path: Path) -> PriceCard:
    return PriceCard.model_validate_json(path.read_text(encoding="utf-8"))


def estimate_nominal_cost_microusd(
    usage: UsageMetrics,
    price_card: PriceCard,
) -> int | None:
    """Estimate list-price cost; this is not an invoice."""

    if (
        usage.input_tokens is None
        or usage.output_tokens is None
        or usage.cached_input_tokens is None
    ):
        return None

    cached = min(
        usage.cached_input_tokens,
        usage.input_tokens,
    )
    uncached = usage.input_tokens - cached

    # USD per million tokens becomes micro-USD per token,
    # so no additional million-token division is required.
    microusd = (
        Decimal(uncached) * price_card.input_usd_per_million
        + Decimal(cached) * price_card.cached_input_usd_per_million
        + Decimal(usage.output_tokens) * price_card.output_usd_per_million
    )

    return int(
        microusd.quantize(
            Decimal(1),
            rounding=ROUND_HALF_UP,
        )
    )
