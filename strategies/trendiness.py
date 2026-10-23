"""Trendiness screen: does a ticker's price history actually suit a
trend-following strategy, or is it mostly noise?

Kaufman's Efficiency Ratio: the net directional move over a period, divided
by the total distance the price actually travelled (sum of every day's
absolute move). 1.0 = moved in a straight line (ideal for MA-crossover);
near 0 = pure back-and-forth chop that cancels out (gets whipsawed — see
the XOM vs MSFT finding in README "Does it actually have an edge?").

This is a screen, not a strategy — it doesn't generate buy/sell signals,
it answers "should this ticker even be considered" before running one.
"""
from __future__ import annotations

import pandas as pd


def efficiency_ratio(prices: pd.DataFrame) -> float:
    """Kaufman's Efficiency Ratio for `prices["Close"]` over its full range.

    Returns a value in [0, 1]. 0.0 if the price never moved at all (total
    path length of zero), rather than dividing by zero.
    """
    close = prices["Close"]
    net_move = abs(close.iloc[-1] - close.iloc[0])
    total_path = close.diff().abs().sum()
    if total_path == 0:
        return 0.0
    return float(net_move / total_path)
