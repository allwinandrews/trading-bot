import pandas as pd
import pytest

from strategies.trendiness import efficiency_ratio


def test_straight_line_trend_is_one():
    prices = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]})
    assert efficiency_ratio(prices) == pytest.approx(1.0)


def test_straight_line_downtrend_is_also_one():
    # Efficiency ratio cares about trend strength, not direction.
    prices = pd.DataFrame({"Close": [105, 104, 103, 102, 101, 100]})
    assert efficiency_ratio(prices) == pytest.approx(1.0)


def test_pure_back_and_forth_chop_is_low():
    # Net move = |101-100| = 1. Total path = 1+1+1+1+1 = 5. ER = 0.2.
    prices = pd.DataFrame({"Close": [100, 101, 100, 101, 100, 101]})
    assert efficiency_ratio(prices) == pytest.approx(0.2)


def test_flat_price_is_zero_not_a_divide_error():
    prices = pd.DataFrame({"Close": [100, 100, 100, 100]})
    assert efficiency_ratio(prices) == 0.0
