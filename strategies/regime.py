"""Market regime filter: is the broad market itself in an uptrend?

Individual-stock trend-following struggles in choppy/bear conditions
regardless of which stock you pick — the XOM/JNJ underperformance in the
portfolio backtest wasn't really about those specific stocks so much as the
kind of price action they had (see README "Does it actually have an edge?").
This checks whether the broad market (by default SPY) is itself trending,
so a strategy's own signal can be gated to only act when the overall
backdrop favors trend-following — sitting out, and exiting existing
positions, when it doesn't.

Deliberately the simplest possible integration: this produces a signal in
the exact same shape (1/0 series indexed by date) as
strategies/moving_average.py's generate_signals, so it combines with any
strategy's signal via a plain AND — no changes needed to the backtest
engines themselves.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_REGIME_WINDOW = 200


def market_regime_signal(market_prices: pd.DataFrame, window: int = DEFAULT_REGIME_WINDOW) -> pd.Series:
    """1 when market_prices["Close"] is above its own `window`-day average, else 0.

    No opinion (0, the conservative default) until there's enough history
    for the average to be meaningful — same warmup handling as
    strategies/moving_average.py's generate_signals.
    """
    close = market_prices["Close"]
    regime_ma = close.rolling(window=window, min_periods=window).mean()

    signal = pd.Series(0, index=market_prices.index, dtype=int)
    signal[close > regime_ma] = 1
    signal[regime_ma.isna()] = 0
    return signal


def apply_regime_filter(strategy_signal: pd.Series, regime_signal: pd.Series) -> pd.Series:
    """Only allow the strategy's signal through when the regime signal also agrees.

    A plain AND: this both blocks new entries and forces an exit from an
    existing position if the regime turns off mid-trade — a deliberate
    choice, consistent with this being a risk-reduction strategy (see
    README) rather than a return-maximizing one. Aligns on
    strategy_signal's index since the regime data (e.g. SPY) may cover a
    different trading-day range than the traded ticker; missing regime
    days are treated as "risk off" (0), the conservative default.
    """
    aligned_regime = regime_signal.reindex(strategy_signal.index).fillna(0)
    return (strategy_signal.astype(int) & aligned_regime.astype(int)).astype(int)
