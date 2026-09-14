"""CLI entry point: backtest a moving-average crossover strategy and compare
it against a buy-and-hold benchmark on the same data.

Usage:
    python run_backtest.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
    python run_backtest.py --ticker RELIANCE.NS --short-window 10 --long-window 30 --stop-loss-pct 0.03
    python run_backtest.py --ticker AAPL --ai-journal

Thin wrapper around backtester.runner.execute_backtest — the actual wiring
lives there so the Streamlit dashboard can reuse it unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from ai.trade_journal import generate_trade_journal
from backtester.engine import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
)
from backtester.runner import execute_backtest
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="AAPL", help="Ticker symbol (Yahoo Finance format)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--short-window", type=int, default=DEFAULT_SHORT_WINDOW)
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW)
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    p.add_argument("--risk-per-trade-pct", type=float, default=DEFAULT_RISK_PER_TRADE_PCT)
    p.add_argument("--commission-pct", type=float, default=DEFAULT_COMMISSION_PCT)
    p.add_argument("--no-plot", action="store_true", help="Skip saving the equity curve chart")
    p.add_argument(
        "--ai-journal", action="store_true",
        help="Ask Claude to write a plain-English review of the trade log (needs ANTHROPIC_API_KEY)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run = execute_backtest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        short_window=args.short_window,
        long_window=args.long_window,
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss_pct,
        risk_per_trade_pct=args.risk_per_trade_pct,
        commission_pct=args.commission_pct,
    )

    print(f"\n{args.ticker}  {args.start} -> {args.end}   {run.strategy_desc}\n")

    header = f"{'metric':<20}{'strategy':>15}{'buy_and_hold':>15}"
    print(header)
    print("-" * len(header))
    for key in run.strategy_metrics:
        print(f"{key:<20}{run.strategy_metrics[key]:>15}{run.benchmark_metrics.get(key, '-'):>15}")

    if not run.strategy_result.trades.empty:
        print(f"\nLast 5 trades:\n{run.strategy_result.trades.tail(5).to_string(index=False)}")

    if not args.no_plot:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        run.strategy_result.equity_curve.plot(ax=ax, label="Strategy")
        run.benchmark_equity.plot(ax=ax, label="Buy & Hold")
        ax.set_title(f"{args.ticker} - MA crossover vs. buy & hold")
        ax.set_ylabel("Equity")
        ax.legend()
        fig.tight_layout()
        out_path = RESULTS_DIR / f"{args.ticker}_equity_curve.png"
        fig.savefig(out_path)
        print(f"\nSaved equity curve chart to {out_path}")

    if args.ai_journal:
        try:
            journal = generate_trade_journal(
                args.ticker, run.strategy_desc, run.strategy_metrics, run.strategy_result.trades
            )
        except RuntimeError as e:
            print(f"\n[AI journal skipped] {e}")
        else:
            print(f"\n--- AI trade journal ---\n{journal}")
            RESULTS_DIR.mkdir(exist_ok=True)
            journal_path = RESULTS_DIR / f"{args.ticker}_trade_journal.md"
            journal_path.write_text(journal, encoding="utf-8")
            print(f"\nSaved AI trade journal to {journal_path}")


if __name__ == "__main__":
    main()
