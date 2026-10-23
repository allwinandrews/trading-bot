"""CLI entry point: does the MA-crossover strategy actually hold up?

Usage:
    python run_robustness_check.py
    python run_robustness_check.py --tickers AAPL MSFT SPY --start 2015-01-01 --end 2024-01-01

A strategy that only looks good on one stock over one time window hasn't
proven anything. This runs the same backtest (via backtester.runner.execute_backtest
— no logic duplicated) across a diverse basket of tickers over one long
period spanning several market regimes (the 2018 correction, the 2020 COVID
crash, the 2022 bear market), and reports how often it actually beats
buy-and-hold rather than just looking at a single favorable run.
"""
from __future__ import annotations

import argparse

import pandas as pd

from backtester.runner import execute_backtest
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW

# Deliberately diverse: two tech names, healthcare, energy, and a broad
# index ETF — a strategy that only "works" on one sector hasn't proven much.
DEFAULT_TICKERS = ["AAPL", "MSFT", "JNJ", "XOM", "SPY"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-01-01"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--short-window", type=int, default=DEFAULT_SHORT_WINDOW)
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for ticker in args.tickers:
        try:
            run = execute_backtest(
                ticker=ticker, start=args.start, end=args.end,
                short_window=args.short_window, long_window=args.long_window,
            )
        except ValueError as e:
            print(f"Skipping {ticker}: {e}")
            continue

        rows.append({
            "ticker": ticker,
            "strategy_return_pct": run.strategy_metrics["total_return_pct"],
            "buy_hold_return_pct": run.benchmark_metrics["total_return_pct"],
            "beat_buy_hold": run.strategy_metrics["total_return_pct"] > run.benchmark_metrics["total_return_pct"],
            "num_trades": run.strategy_metrics["num_trades"],
            "win_rate_pct": run.strategy_metrics["win_rate_pct"],
            "sharpe": run.strategy_metrics["sharpe_ratio"],
            "strategy_drawdown_pct": run.strategy_metrics["max_drawdown_pct"],
            "buy_hold_drawdown_pct": run.benchmark_metrics["max_drawdown_pct"],
        })

    if not rows:
        print("No results — every ticker failed to load.")
        return

    results = pd.DataFrame(rows)
    print(f"\nMA crossover ({args.short_window}/{args.long_window}) vs buy & hold, {args.start} -> {args.end}\n")
    print(results.to_string(index=False))

    beat_count = int(results["beat_buy_hold"].sum())
    safer_count = int((results["strategy_drawdown_pct"] > results["buy_hold_drawdown_pct"]).sum())
    print(
        f"\nBeat buy & hold on {beat_count}/{len(results)} tickers "
        f"({results['beat_buy_hold'].mean() * 100:.0f}%)"
    )
    print(
        f"Smaller max drawdown than buy & hold on {safer_count}/{len(results)} tickers "
        f"({safer_count / len(results) * 100:.0f}%)"
    )
    print(
        f"Average strategy return: {results['strategy_return_pct'].mean():.2f}%  "
        f"vs buy & hold: {results['buy_hold_return_pct'].mean():.2f}%"
    )
    print(
        f"Average strategy drawdown: {results['strategy_drawdown_pct'].mean():.2f}%  "
        f"vs buy & hold: {results['buy_hold_drawdown_pct'].mean():.2f}%"
    )
    print(
        f"Average win rate: {results['win_rate_pct'].mean():.1f}%   "
        f"Average Sharpe: {results['sharpe'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
