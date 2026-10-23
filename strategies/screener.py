"""Rank a universe of tickers to find the best current candidates to trade.

Combines the statistical tools already built and validated individually —
MA-crossover signal (strategies/moving_average.py), trendiness/efficiency
ratio (strategies/trendiness.py), and a market regime gate
(strategies/regime.py) — into one composite score, so the portfolio bot can
answer "which of these many tickers is actually worth trading right now"
instead of a human picking one ticker by hand.

Deliberately simple and explainable (a weighted product of two already-
understood numbers), not a black-box model — each factor was independently
backtested before combining them here.
"""
from __future__ import annotations

import logging

import pandas as pd

from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW, generate_signals
from strategies.regime import DEFAULT_REGIME_WINDOW, apply_regime_filter, market_regime_signal
from strategies.trendiness import efficiency_ratio

logger = logging.getLogger(__name__)

DEFAULT_MOMENTUM_WINDOW = 90  # trading days of lookback for the momentum factor


def _momentum_score(prices: pd.DataFrame, window: int = DEFAULT_MOMENTUM_WINDOW) -> float:
    """% price change over the trailing `window` days — simple relative-strength proxy.

    Negative momentum isn't clamped to zero here; rank_candidates already
    excludes anything whose MA signal is off, so a candidate reaching this
    far has a positive trend signal even if raw momentum is small/negative
    (e.g. right at a fresh crossover) — the composite score naturally ranks
    those lower without a separate cutoff needed.
    """
    close = prices["Close"]
    if len(close) <= window or close.iloc[-window] == 0:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-window] - 1) * 100)


def rank_candidates(
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    regime_window: int = DEFAULT_REGIME_WINDOW,
) -> pd.DataFrame:
    """Score and rank every ticker with enough data, best candidate first.

    Only tickers whose MA-crossover signal is currently "on" (uptrend) *and*
    pass the market regime gate are included at all — this is a screen for
    "worth considering today," not a ranking of the whole universe including
    tickers with no business being traded right now. Returns a DataFrame
    with columns [ticker, last_price, efficiency_ratio, momentum_pct,
    score], sorted by score descending; empty if nothing qualifies.
    """
    regime_signal = market_regime_signal(market_prices, window=regime_window)

    rows = []
    for ticker, prices in prices_by_ticker.items():
        if prices.empty or len(prices) < long_window:
            continue
        close = prices["Close"]
        if close.isna().any():
            logger.warning("%s: skipping from screen — NaN in price history", ticker)
            continue

        raw_signal = generate_signals(prices, short_window, long_window)
        gated_signal = apply_regime_filter(raw_signal, regime_signal)
        if int(gated_signal.iloc[-1]) != 1:
            continue

        trendiness = efficiency_ratio(prices)
        momentum = _momentum_score(prices, momentum_window)
        if trendiness <= 0 or momentum <= 0:
            continue  # trending sideways/noisy, or trending down despite a fresh MA crossover

        score = trendiness * momentum
        rows.append({
            "ticker": ticker,
            "last_price": float(close.iloc[-1]),
            "efficiency_ratio": round(trendiness, 4),
            "momentum_pct": round(momentum, 2),
            "score": round(score, 4),
        })

    if not rows:
        return pd.DataFrame(columns=["ticker", "last_price", "efficiency_ratio", "momentum_pct", "score"])

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
