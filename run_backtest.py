"""CLI entry point: backtest a moving-average crossover strategy and compare
it against a buy-and-hold benchmark on the same data.

Usage:
    python run_backtest.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
    python run_backtest.py --ticker RELIANCE.NS --short-window 10 --long-window 30 --stop-loss-pct 0.03

This is the single source of truth for "run a backtest" — the future
Streamlit dashboard and any notebooks should call into these same modules
rather than re-implementing this wiring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from ai.trade_journal import generate_trade_journal
from backtester.engine import run_backtest, run_buy_and_hold
from backtester.metrics import summarize
from data.loader import load_price_data
from strategies.moving_average import generate_signals

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="AAPL", help="Ticker symbol (Yahoo Finance format)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--short-window", type=int, default=20)
    p.add_argument("--long-window", type=int, default=50)
    p.add_argument("--initial-capital", type=float, default=100_000)
    p.add_argument("--stop-loss-pct", type=float, default=0.02)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    p.add_argument("--commission-pct", type=float, default=0.001)
    p.add_argument("--no-plot", action="store_true", help="Skip saving the equity curve chart")
    p.add_argument(
        "--ai-journal", action="store_true",
        help="Ask Claude to write a plain-English review of the trade log (needs ANTHROPIC_API_KEY)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_price_data(args.ticker, args.start, args.end)

    signals = generate_signals(prices, args.short_window, args.long_window)

    strategy_result = run_backtest(
        prices,
        signals,
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss_pct,
        risk_per_trade_pct=args.risk_per_trade_pct,
        commission_pct=args.commission_pct,
    )
    benchmark_equity = run_buy_and_hold(prices, initial_capital=args.initial_capital)

    strategy_metrics = summarize(strategy_result.equity_curve, strategy_result.trades)
    benchmark_metrics = summarize(benchmark_equity, strategy_result.trades.iloc[0:0])

    print(f"\n{args.ticker}  {args.start} -> {args.end}   "
          f"MA({args.short_window}/{args.long_window})  "
          f"stop-loss={args.stop_loss_pct:.1%}  risk/trade={args.risk_per_trade_pct:.1%}\n")

    header = f"{'metric':<20}{'strategy':>15}{'buy_and_hold':>15}"
    print(header)
    print("-" * len(header))
    for key in strategy_metrics:
        print(f"{key:<20}{strategy_metrics[key]:>15}{benchmark_metrics.get(key, '-'):>15}")

    if not strategy_result.trades.empty:
        print(f"\nLast 5 trades:\n{strategy_result.trades.tail(5).to_string(index=False)}")

    if not args.no_plot:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        strategy_result.equity_curve.plot(ax=ax, label="Strategy")
        benchmark_equity.plot(ax=ax, label="Buy & Hold")
        ax.set_title(f"{args.ticker} — MA crossover vs. buy & hold")
        ax.set_ylabel("Equity")
        ax.legend()
        fig.tight_layout()
        out_path = RESULTS_DIR / f"{args.ticker}_equity_curve.png"
        fig.savefig(out_path)
        print(f"\nSaved equity curve chart to {out_path}")

    if args.ai_journal:
        strategy_desc = (
            f"Moving average crossover (short={args.short_window}, long={args.long_window}), "
            f"stop-loss={args.stop_loss_pct:.1%}, risk/trade={args.risk_per_trade_pct:.1%}"
        )
        try:
            journal = generate_trade_journal(
                args.ticker, strategy_desc, strategy_metrics, strategy_result.trades
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
