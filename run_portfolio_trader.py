"""CLI entry point: run the portfolio bot — screens a universe of tickers
and manages several simultaneous positions from one shared capital pool.

Usage:
    python run_portfolio_trader.py --loop --interval-seconds 300
    python run_portfolio_trader.py --loop --max-positions 4 --initial-capital 500 --client-id 5

Unlike run_paper_trader.py, there's no single --ticker — the universe
(data/universe.py by default) is screened fresh every check via
strategies/screener.py. See execution/portfolio_trader.py for the full
entry/exit/sizing logic.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from data.universe import DEFAULT_UNIVERSE
from execution import alerts, trade_log
from execution.broker import DEFAULT_CLIENT_ID, IBBroker
from execution.portfolio_trader import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MARKET_TICKER,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
    run_once,
)
from risk.stops import DailyLossCircuitBreaker
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    p.add_argument("--short-window", type=int, default=DEFAULT_SHORT_WINDOW)
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW)
    p.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    p.add_argument("--risk-per-trade-pct", type=float, default=DEFAULT_RISK_PER_TRADE_PCT)
    p.add_argument("--market-ticker", type=str, default=DEFAULT_MARKET_TICKER)
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument(
        "--use-sentiment-filter", action="store_true",
        help="Veto entries with a clear negative-news signal via ai/sentiment_filter.py "
             "(needs GEMINI_API_KEY; silently skipped if not set — see that module's fail-open design)",
    )
    p.add_argument(
        "--max-daily-loss-pct", type=float, default=0.03,
        help="Circuit breaker: halt new entries once the portfolio is down this much for the day",
    )
    p.add_argument("--loop", action="store_true", help="Keep running instead of checking once and exiting")
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    p.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for this run's trade log/equity/ledger (default: execution/live_data)",
    )
    return p.parse_args()


def _run_check(args: argparse.Namespace, circuit_breaker: DailyLossCircuitBreaker, sentiment_filter) -> dict:
    log_dir = Path(args.log_dir) if args.log_dir else trade_log.DEFAULT_LOG_DIR
    with IBBroker(client_id=args.client_id) as broker:
        return run_once(
            broker,
            universe=DEFAULT_UNIVERSE,
            max_positions=args.max_positions,
            short_window=args.short_window,
            long_window=args.long_window,
            stop_loss_pct=args.stop_loss_pct,
            risk_per_trade_pct=args.risk_per_trade_pct,
            market_ticker=args.market_ticker,
            sentiment_filter=sentiment_filter,
            circuit_breaker=circuit_breaker,
            log_dir=log_dir,
            initial_capital=args.initial_capital,
        )


def main() -> None:
    args = parse_args()
    circuit_breaker = DailyLossCircuitBreaker(max_daily_loss_pct=args.max_daily_loss_pct)

    sentiment_filter = None
    if args.use_sentiment_filter:
        from ai.sentiment_filter import score_sentiment
        sentiment_filter = score_sentiment

    if not args.loop:
        result = _run_check(args, circuit_breaker, sentiment_filter)
        print(result)
        return

    print(f"Portfolio bot started — checking every {args.interval_seconds}s. Ctrl+C to stop.")
    while True:
        try:
            result = _run_check(args, circuit_breaker, sentiment_filter)
            print(result)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception("Check failed — will retry next interval")
            alerts.record_alert("error", f"Portfolio bot: check failed — {e}", source="portfolio_autopilot_loop")

        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("\nPortfolio bot stopped.")
            break


if __name__ == "__main__":
    main()
