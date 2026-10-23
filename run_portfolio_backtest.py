"""CLI entry point: does spreading capital across multiple tickers actually help?

Usage:
    python run_portfolio_backtest.py
    python run_portfolio_backtest.py --tickers AAPL MSFT JNJ XOM SPY --start 2015-01-01

Runs the strategy across all given tickers at once, sharing ONE capital pool
(backtester.portfolio_engine — real diversification, not N separate
single-ticker runs like run_robustness_check.py). Compares against an
equal-weight buy-and-hold of the same basket, using the same metrics
(backtester.metrics.summarize) as every other backtest in this project.
"""
from __future__ import annotations

import argparse

from backtester.metrics import summarize
from backtester.portfolio_engine import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
    run_portfolio_backtest,
    run_portfolio_buy_and_hold,
)
from data.loader import load_price_data
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW, generate_signals
from strategies.regime import DEFAULT_REGIME_WINDOW, apply_regime_filter, market_regime_signal
from strategies.trendiness import efficiency_ratio

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
    p.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    p.add_argument("--risk-per-trade-pct", type=float, default=DEFAULT_RISK_PER_TRADE_PCT)
    p.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument(
        "--min-efficiency-ratio", type=float, default=None,
        help="Skip tickers whose trendiness (strategies/trendiness.py) is below this "
             "threshold over the backtest window — see run_trend_screen.py. E.g. 0.05 "
             "would have excluded XOM/JNJ in the 2015-2024 AAPL/MSFT/JNJ/XOM/SPY basket.",
    )
    p.add_argument(
        "--regime-filter", action="store_true",
        help="Only take/hold trades when --regime-ticker is itself above its own "
             "--regime-window average (strategies/regime.py) — sits out, and exits "
             "existing positions, during broad-market chop/downtrends.",
    )
    p.add_argument("--regime-ticker", default="SPY")
    p.add_argument("--regime-window", type=int, default=DEFAULT_REGIME_WINDOW)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices_by_ticker, signals_by_ticker = {}, {}
    for ticker in args.tickers:
        try:
            prices = load_price_data(ticker, start=args.start, end=args.end)
        except ValueError as e:
            print(f"Skipping {ticker}: {e}")
            continue

        if args.min_efficiency_ratio is not None:
            er = efficiency_ratio(prices)
            if er < args.min_efficiency_ratio:
                print(f"Skipping {ticker}: efficiency ratio {er:.3f} below threshold {args.min_efficiency_ratio}")
                continue

        prices_by_ticker[ticker] = prices
        signals_by_ticker[ticker] = generate_signals(prices, args.short_window, args.long_window)

    if len(prices_by_ticker) < 2:
        print("Need at least 2 tickers with data to test diversification.")
        return

    if args.regime_filter:
        regime_prices = load_price_data(args.regime_ticker, start=args.start, end=args.end)
        regime_signal = market_regime_signal(regime_prices, window=args.regime_window)
        for ticker in signals_by_ticker:
            signals_by_ticker[ticker] = apply_regime_filter(signals_by_ticker[ticker], regime_signal)
        print(f"Regime filter on: only trading while {args.regime_ticker} is above its own {args.regime_window}-day average.")

    portfolio = run_portfolio_backtest(
        prices_by_ticker, signals_by_ticker,
        initial_capital=args.initial_capital, stop_loss_pct=args.stop_loss_pct,
        risk_per_trade_pct=args.risk_per_trade_pct, max_positions=args.max_positions,
    )
    benchmark_equity = run_portfolio_buy_and_hold(prices_by_ticker, initial_capital=args.initial_capital)

    portfolio_metrics = summarize(portfolio.equity_curve, portfolio.trades)
    benchmark_metrics = summarize(benchmark_equity, portfolio.trades.iloc[0:0])

    print(
        f"\nPortfolio ({args.short_window}/{args.long_window} MA, "
        f"max {args.max_positions} positions) across {list(prices_by_ticker)}, "
        f"{args.start} -> {args.end}\n"
    )
    header = f"{'metric':<20}{'portfolio':>15}{'equal_weight_bh':>18}"
    print(header)
    print("-" * len(header))
    for key in portfolio_metrics:
        print(f"{key:<20}{portfolio_metrics[key]:>15}{benchmark_metrics.get(key, '-'):>18}")

    if not portfolio.trades.empty:
        print(f"\nTrades by ticker:\n{portfolio.trades['ticker'].value_counts().to_string()}")
        print(f"\nLast 5 trades:\n{portfolio.trades.tail(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
