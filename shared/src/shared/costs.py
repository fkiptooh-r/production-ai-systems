"""Token cost estimation for LLM calls.

Providers price models per million tokens (MTok), with different prices
for input (prompt) and output (completion) tokens.
"""

TOKENS_PER_MILLION = 1_000_000


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> float:
    """Return the estimated cost in USD of one LLM call."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")

    input_cost = input_tokens / TOKENS_PER_MILLION * input_price_per_mtok
    output_cost = output_tokens / TOKENS_PER_MILLION * output_price_per_mtok
    return input_cost + output_cost
