"""Moving-average crossover strategy.

The simplest, most well-understood trend-following strategy — a deliberate
starting point (see README "Build order"). Long-only: go long when the short
moving average crosses above the long one, go flat when it crosses back below.

A strategy's only job is to look at price history and output a signal — it
never touches money, sizing, or stops. That's the backtester/risk layer's job,
kept separate so strategies stay simple and swappable.
"""
from __future__ import annotations

import pandas as pd


def generate_signals(
    prices: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.Series:
    """Return a signal series aligned to `prices.index`: 1 = long, 0 = flat.

    `prices` must have a "Close" column. Uses only data available up to and
    including each row — no lookahead — so the caller can safely execute on
    the *next* bar.
    """
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    close = prices["Close"]
    short_ma = close.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = close.rolling(window=long_window, min_periods=long_window).mean()

    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[short_ma > long_ma] = 1

    # No opinion until both averages have enough history to be meaningful.
    warmup = long_ma.isna()
    signal[warmup] = 0

    return signal
