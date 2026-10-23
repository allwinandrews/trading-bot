"""CLI entry point: which tickers actually suit a trend-following strategy?

Usage:
    python run_trend_screen.py
    python run_trend_screen.py --tickers AAPL MSFT JNJ XOM SPY GOOGL AMZN TSLA KO PFE

Ranks tickers by Kaufman's Efficiency Ratio (strategies/trendiness.py) over
the given period — closer to 1.0 means the price moved in a cleaner,
straighter trend (a good fit for MA-crossover); closer to 0 means mostly
back-and-forth chop (a bad fit, likely to get whipsawed — this is exactly
what happened to XOM, see README "Does it actually have an edge?"). This is
a screen, not a backtest: use its output to choose which tickers to feed
into run_portfolio_backtest.py (via --min-efficiency-ratio), rather than
trading every ticker indiscriminately.
"""
from __future__ import annotations

import argparse

import pandas as pd

from data.loader import load_price_data
from strategies.trendiness import efficiency_ratio

DEFAULT_TICKERS = ["AAPL", "MSFT", "JNJ", "XOM", "SPY", "GOOGL", "AMZN", "TSLA", "KO", "PFE"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-01-01"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for ticker in args.tickers:
        try:
            prices = load_price_data(ticker, start=args.start, end=args.end)
        except ValueError as e:
            print(f"Skipping {ticker}: {e}")
            continue
        rows.append({"ticker": ticker, "efficiency_ratio": round(efficiency_ratio(prices), 3)})

    if not rows:
        print("No results — every ticker failed to load.")
        return

    results = pd.DataFrame(rows).sort_values("efficiency_ratio", ascending=False)
    print(f"\nTrendiness screen, {args.start} -> {args.end} (1.0 = clean trend, 0 = pure chop)\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
