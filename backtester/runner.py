"""Shared orchestration: data -> signals -> backtest -> benchmark -> metrics.

The single source of truth for "run a backtest" — run_backtest.py (CLI) and
dashboard.py (Streamlit) both call `execute_backtest` rather than each
re-wiring data loading, signal generation, simulation, and scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.engine import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
    BacktestResult,
    run_backtest,
    run_buy_and_hold,
)
from backtester.metrics import summarize
from data.loader import load_price_data
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW, generate_signals


@dataclass
class BacktestRun:
    prices: pd.DataFrame
    signals: pd.Series
    strategy_result: BacktestResult
    benchmark_equity: pd.Series
    strategy_metrics: dict
    benchmark_metrics: dict
    strategy_desc: str


def execute_backtest(
    ticker: str,
    start: str,
    end: str,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
) -> BacktestRun:
    prices = load_price_data(ticker, start, end)
    signals = generate_signals(prices, short_window, long_window)

    strategy_result = run_backtest(
        prices,
        signals,
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss_pct,
        risk_per_trade_pct=risk_per_trade_pct,
        commission_pct=commission_pct,
    )
    benchmark_equity = run_buy_and_hold(prices, initial_capital=initial_capital)

    strategy_metrics = summarize(strategy_result.equity_curve, strategy_result.trades)
    # Buy-and-hold has no trade log of its own — score it with an empty one.
    benchmark_metrics = summarize(benchmark_equity, strategy_result.trades.iloc[0:0])

    strategy_desc = (
        f"Moving average crossover (short={short_window}, long={long_window}), "
        f"stop-loss={stop_loss_pct:.1%}, risk/trade={risk_per_trade_pct:.1%}"
    )

    return BacktestRun(
        prices=prices,
        signals=signals,
        strategy_result=strategy_result,
        benchmark_equity=benchmark_equity,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        strategy_desc=strategy_desc,
    )
