import pytest

from shared.costs import estimate_cost


def test_one_million_tokens_each_costs_the_sum_of_both_prices():
    assert estimate_cost(1_000_000, 1_000_000, 3.0, 15.0) == pytest.approx(18.0)


def test_typical_small_call():
    # 1,200 prompt tokens + 300 completion tokens at $3 / $15 per MTok
    assert estimate_cost(1_200, 300, 3.0, 15.0) == pytest.approx(0.0081)


def test_zero_tokens_costs_nothing():
    assert estimate_cost(0, 0, 3.0, 15.0) == 0


def test_negative_tokens_are_rejected():
    with pytest.raises(ValueError):
        estimate_cost(-1, 0, 3.0, 15.0)
