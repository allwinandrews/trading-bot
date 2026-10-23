"""Backtest simulation engine.

Long-only, single-position-at-a-time, daily-bar simulation. Deliberately
simple for a first build — no shorting, no pyramiding, no intraday granularity.

Execution model (avoids lookahead bias): a signal computed from bar t's Close
is acted on at bar t+1's Open — you can never trade on information you
wouldn't actually have had yet. Stop-losses are checked intrabar against the
day's Low and assumed to fill at the stop price (a simplification — real
fills can slip past the stop on a fast move; see README).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.costs import ibkr_commission
from risk.position_sizing import shares_for_fixed_risk
from risk.stops import fixed_stop_price


def _commission(shares: float, price: float, commission_pct: float | None) -> float:
    """Cost of one fill — IBKR's real schedule unless a flat rate is forced."""
    if commission_pct is None:
        return ibkr_commission(shares, price)
    return shares * price * commission_pct

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_STOP_LOSS_PCT = 0.02
DEFAULT_RISK_PER_TRADE_PCT = 0.01
# None = model IBKR's real fee schedule (backtester/costs.py), where the
# effective rate depends on trade size. A flat rate was the old default and
# understated costs ~10x at small trade sizes; pass a float to force one.
DEFAULT_COMMISSION_PCT = None


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame


def _trade_record(entry_date, entry_price, exit_date, exit_price, shares, exit_reason) -> dict:
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "return_pct": (exit_price - entry_price) / entry_price * 100,
        "exit_reason": exit_reason,
    }


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    commission_pct: float | None = DEFAULT_COMMISSION_PCT,
) -> BacktestResult:
    """Simulate `signals` (1=long, 0=flat) against `prices` (needs Open/High/Low/Close)."""
    cash = initial_capital
    shares = 0
    in_position = False
    entry_price = None
    entry_date = None
    stop_price = None

    equity_values = []
    trades: list[dict] = []

    for i, date in enumerate(prices.index):
        o, h, l, c = (
            prices["Open"].iloc[i],
            prices["High"].iloc[i],
            prices["Low"].iloc[i],
            prices["Close"].iloc[i],
        )
        signal_prev = signals.iloc[i - 1] if i > 0 else 0

        # 1. Act on yesterday's signal at today's open.
        if not in_position and signal_prev == 1:
            stop_price = fixed_stop_price(o, stop_loss_pct)
            shares = shares_for_fixed_risk(cash, o, stop_price, risk_per_trade_pct)
            if shares > 0:
                cost = shares * o
                cash -= cost + _commission(shares, o, commission_pct)
                in_position = True
                entry_price = o
                entry_date = date
        elif in_position and signal_prev == 0:
            cash += shares * o - _commission(shares, o, commission_pct)
            trades.append(_trade_record(entry_date, entry_price, date, o, shares, "signal_exit"))
            in_position = False
            shares = 0

        # 2. Stop-loss check, intrabar, only if still in a position.
        if in_position and l <= stop_price:
            cash += shares * stop_price - _commission(shares, stop_price, commission_pct)
            trades.append(
                _trade_record(entry_date, entry_price, date, stop_price, shares, "stop_loss")
            )
            in_position = False
            shares = 0

        # 3. Mark to market at today's close.
        equity_values.append(cash + shares * c)

    equity_curve = pd.Series(equity_values, index=prices.index, name="equity")
    trades_df = pd.DataFrame(
        trades,
        columns=["entry_date", "entry_price", "exit_date", "exit_price", "shares",
                 "return_pct", "exit_reason"],
    )
    return BacktestResult(equity_curve=equity_curve, trades=trades_df)


def run_buy_and_hold(prices: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> pd.Series:
    """Benchmark: buy at the first Open with all capital, hold to the end."""
    entry_price = prices["Open"].iloc[0]
    shares = initial_capital // entry_price
    leftover_cash = initial_capital - shares * entry_price
    equity_curve = leftover_cash + shares * prices["Close"]
    equity_curve.name = "buy_and_hold_equity"
    return equity_curve
