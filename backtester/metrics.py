"""Performance metrics computed from an equity curve and trade log.

Kept separate from the simulation loop in engine.py so the same metrics can
score a backtest, a paper-trading run, or (later) a live run identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return_pct(equity_curve: pd.Series) -> float:
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def cagr_pct(equity_curve: pd.Series) -> float:
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if growth <= 0:
        return -100.0
    return (growth ** (1 / years) - 1) * 100


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    return drawdown.min() * 100


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    excess = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / daily_returns.std())


def summarize(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
    """One dict of headline metrics — what gets printed/compared for a run."""
    num_trades = len(trades)
    win_rate = (
        (trades["return_pct"] > 0).mean() * 100 if num_trades > 0 else 0.0
    )
    avg_trade_return = trades["return_pct"].mean() if num_trades > 0 else 0.0

    return {
        "total_return_pct": round(total_return_pct(equity_curve), 2),
        "cagr_pct": round(cagr_pct(equity_curve), 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity_curve), 2),
        "sharpe_ratio": round(sharpe_ratio(equity_curve), 2),
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_trade_return_pct": round(avg_trade_return, 2),
    }
