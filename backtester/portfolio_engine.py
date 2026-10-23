"""Multi-ticker portfolio backtest: the same strategy run across several
tickers at once, sharing ONE capital pool.

This is not the same thing as run_robustness_check.py, which runs N
independent single-ticker backtests each with their own full capital — that
tests "does this work on ticker X," repeated N times. This module tests
"if I split one account across N tickers at once, does spreading risk
actually help" — real diversification, not repetition.

Same execution model as engine.py (signal from day t's close acted on at
day t+1's open, stops checked intrabar against Low) — extended to track
several simultaneous positions against one shared cash balance.
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
# None = model IBKR's real fee schedule (backtester/costs.py), where the
# effective rate depends on trade size. A flat rate was the old default and
# understated costs ~10x at small trade sizes; pass a float to force one.
DEFAULT_COMMISSION_PCT = None
DEFAULT_MAX_POSITIONS = 5

# Deliberately smaller than the single-ticker default (0.01). The position
# sizing formula (risk_per_trade_pct / stop_loss_pct fraction of capital
# per trade — see risk/position_sizing.py) claims ~50% of capital per
# position at the single-ticker default, which would let 1-2 positions
# swallow the whole shared pool and defeat the point of diversifying.
# 0.0025 here targets roughly 12.5% of capital per position instead, so
# DEFAULT_MAX_POSITIONS positions can actually coexist.
DEFAULT_RISK_PER_TRADE_PCT = 0.0025


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    # Total commission paid across every fill. Reported because at small
    # account sizes it's often the difference between a profitable strategy
    # and a losing one, and a returns-only summary hides that entirely.
    total_commission: float = 0.0


def _common_index(prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Trading days every ticker has data for, sorted."""
    common = None
    for prices in prices_by_ticker.values():
        common = prices.index if common is None else common.intersection(prices.index)
    return common.sort_values()


def run_portfolio_backtest(
    prices_by_ticker: dict[str, pd.DataFrame],
    signals_by_ticker: dict[str, pd.Series],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> PortfolioResult:
    """Simulate all tickers' signals together against one shared cash pool.

    Each new entry is sized off *currently available cash* (not total
    equity), and `max_positions` caps how many names can be held
    simultaneously — both needed because a single position's sizing formula
    alone can claim a large share of capital; without a cap, the first
    signal or two to fire would starve out any real diversification.
    """
    index = _common_index(prices_by_ticker)
    signals_prev = {t: s.shift(1) for t, s in signals_by_ticker.items()}

    cash = initial_capital
    total_commission = 0.0
    positions: dict[str, dict] = {}  # ticker -> {shares, entry_price, entry_date, stop_price}
    trades: list[dict] = []
    equity_values = []

    for date in index:
        # 1. Exits first, to free up cash for the same day's entries.
        for ticker in list(positions):
            prices = prices_by_ticker[ticker]
            pos = positions[ticker]
            o = prices.loc[date, "Open"]
            l = prices.loc[date, "Low"]
            prev_signal = signals_prev[ticker].loc[date] if date in signals_prev[ticker].index else 0

            exit_reason = exit_price = None
            if l <= pos["stop_price"]:
                exit_reason, exit_price = "stop_loss", pos["stop_price"]
            elif prev_signal == 0:
                exit_reason, exit_price = "signal_exit", o

            if exit_reason:
                exit_fee = _commission(pos["shares"], exit_price, commission_pct)
                total_commission += exit_fee
                cash += pos["shares"] * exit_price - exit_fee
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                    "exit_date": date, "exit_price": exit_price,
                    "shares": pos["shares"],
                    "return_pct": (exit_price - pos["entry_price"]) / pos["entry_price"] * 100,
                    "exit_reason": exit_reason,
                })
                del positions[ticker]

        # 2. Entries, capped by max_positions, sized off current cash.
        for ticker in signals_by_ticker:
            if len(positions) >= max_positions:
                break
            if ticker in positions or date not in signals_prev[ticker].index:
                continue
            if signals_prev[ticker].loc[date] != 1:
                continue

            o = prices_by_ticker[ticker].loc[date, "Open"]
            stop_price = fixed_stop_price(o, stop_loss_pct)
            shares = shares_for_fixed_risk(cash, o, stop_price, risk_per_trade_pct)
            if shares <= 0:
                continue

            entry_fee = _commission(shares, o, commission_pct)
            total_commission += entry_fee
            cash -= shares * o + entry_fee
            positions[ticker] = {
                "shares": shares, "entry_price": o, "entry_date": date, "stop_price": stop_price,
            }

        # 3. Mark to market.
        position_value = sum(
            pos["shares"] * prices_by_ticker[t].loc[date, "Close"] for t, pos in positions.items()
        )
        equity_values.append(cash + position_value)

    equity_curve = pd.Series(equity_values, index=index, name="equity")
    trades_df = pd.DataFrame(
        trades,
        columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
                 "shares", "return_pct", "exit_reason"],
    )
    return PortfolioResult(
        equity_curve=equity_curve, trades=trades_df, total_commission=total_commission
    )


def run_portfolio_buy_and_hold(
    prices_by_ticker: dict[str, pd.DataFrame], initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> pd.Series:
    """Benchmark: split capital equally across all tickers at the start, hold to the end."""
    index = _common_index(prices_by_ticker)
    per_ticker_capital = initial_capital / len(prices_by_ticker)

    first_date = index[0]
    shares = {}
    leftover_cash = 0.0
    for ticker, prices in prices_by_ticker.items():
        entry_price = prices.loc[first_date, "Open"]
        n = per_ticker_capital // entry_price
        shares[ticker] = n
        leftover_cash += per_ticker_capital - n * entry_price

    equity_values = [
        leftover_cash + sum(shares[t] * prices_by_ticker[t].loc[date, "Close"] for t in shares)
        for date in index
    ]
    return pd.Series(equity_values, index=index, name="portfolio_buy_and_hold_equity")
