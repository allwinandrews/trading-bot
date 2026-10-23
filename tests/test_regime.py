import pandas as pd
import pytest

from strategies.regime import apply_regime_filter, market_regime_signal

DATES = pd.date_range("2024-01-01", periods=6, freq="B")


def test_market_regime_signal_on_during_uptrend():
    # Price rising steadily and staying above a 3-day average -> regime on.
    prices = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]}, index=DATES)
    signal = market_regime_signal(prices, window=3)
    # First 2 bars have no 3-day average yet -> off by default.
    assert list(signal) == [0, 0, 1, 1, 1, 1]


def test_market_regime_signal_off_during_downtrend():
    prices = pd.DataFrame({"Close": [105, 104, 103, 102, 101, 100]}, index=DATES)
    signal = market_regime_signal(prices, window=3)
    assert list(signal) == [0, 0, 0, 0, 0, 0]


def test_apply_regime_filter_blocks_entries_when_regime_off():
    strategy_signal = pd.Series([1, 1, 1, 1, 1, 1], index=DATES)
    regime_signal = pd.Series([0, 0, 1, 1, 0, 0], index=DATES)

    result = apply_regime_filter(strategy_signal, regime_signal)
    assert list(result) == [0, 0, 1, 1, 0, 0]


def test_apply_regime_filter_treats_missing_regime_days_as_off():
    strategy_signal = pd.Series([1, 1, 1], index=DATES[:3])
    # Regime data only covers the first 2 days (e.g. SPY missing a day).
    regime_signal = pd.Series([1, 1], index=DATES[:2])

    result = apply_regime_filter(strategy_signal, regime_signal)
    assert list(result) == [1, 1, 0]


def test_apply_regime_filter_all_off_when_strategy_signal_flat():
    strategy_signal = pd.Series([0, 0, 0], index=DATES[:3])
    regime_signal = pd.Series([1, 1, 1], index=DATES[:3])

    result = apply_regime_filter(strategy_signal, regime_signal)
    assert list(result) == [0, 0, 0]
