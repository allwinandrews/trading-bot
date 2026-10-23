"""CLI entry point: run paper/live trading checks for a single ticker.

Usage:
    python run_paper_trader.py --ticker AAPL
    python run_paper_trader.py --ticker AAPL --stop-loss-pct 0.03 --risk-per-trade-pct 0.005
    python run_paper_trader.py --ticker AAPL --loop --interval-seconds 300

Connects to TWS/IB Gateway (host/port from .env, see .env.example — the
default port 4002 is IB Gateway's paper trading port), checks today's
signal, and enters/exits/holds accordingly.

By default runs once and exits — schedule it yourself (cron, Task
Scheduler) for once-per-trading-day checks. Pass --loop for "autopilot"
mode instead: it keeps re-checking every --interval-seconds until you stop
it with Ctrl+C. Since this is a daily-bar strategy, the trading *signal*
only meaningfully changes about once a day — looping more often than that
mainly keeps the equity/position readout (and the dashboard's Live Account
and Trade History pages) fresh, which is the point of "autopilot": you can
leave it running and watch it, not that it trades more often.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from execution import alerts, trade_log
from execution.broker import DEFAULT_CLIENT_ID, IBBroker
from execution.paper_trader import (
    DEFAULT_INITIAL_CAPITAL,
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
    p.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--short-window", type=int, default=DEFAULT_SHORT_WINDOW)
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW)
    p.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    p.add_argument("--risk-per-trade-pct", type=float, default=DEFAULT_RISK_PER_TRADE_PCT)
    p.add_argument(
        "--max-daily-loss-pct", type=float, default=0.03,
        help="Circuit breaker: halt new trades once the account is down this much for the day",
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Keep running (autopilot) instead of checking once and exiting",
    )
    p.add_argument(
        "--interval-seconds", type=int, default=300,
        help="How often to re-check when --loop is set (default 300s = 5 minutes)",
    )
    p.add_argument(
        "--client-id", type=int, default=DEFAULT_CLIENT_ID,
        help="IBKR API connection ID. Change this if running a one-off check while an "
             "autopilot loop (or another connection) is already using the default — "
             "two simultaneous connections with the same ID collide and time out. "
             "Dashboard pages use client_id=2 for the same reason.",
    )
    p.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for this run's trade log/equity history/ledger (default: "
             "execution/live_data). Give each concurrently-running strategy its own "
             "so their trades and equity curves don't get mixed together.",
    )
    p.add_argument(
        "--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
        help="Virtual starting capital for this strategy's own equity tracking — "
             "independent of the real account balance, since multiple strategies "
             "can share one real IBKR paper account (see execution/ledger.py).",
    )
    return p.parse_args()


def _run_check(args: argparse.Namespace, circuit_breaker: DailyLossCircuitBreaker) -> str:
    log_dir = Path(args.log_dir) if args.log_dir else trade_log.DEFAULT_LOG_DIR
    with IBBroker(client_id=args.client_id) as broker:
        return run_once(
            broker,
            ticker=args.ticker,
            short_window=args.short_window,
            long_window=args.long_window,
            stop_loss_pct=args.stop_loss_pct,
            risk_per_trade_pct=args.risk_per_trade_pct,
            circuit_breaker=circuit_breaker,
            log_dir=log_dir,
            initial_capital=args.initial_capital,
        )


def main() -> None:
    args = parse_args()
    circuit_breaker = DailyLossCircuitBreaker(max_daily_loss_pct=args.max_daily_loss_pct)

    if not args.loop:
        action = _run_check(args, circuit_breaker)
        print(f"{args.ticker}: {action}")
        return

    print(f"Autopilot started for {args.ticker} — checking every {args.interval_seconds}s. Ctrl+C to stop.")
    while True:
        try:
            action = _run_check(args, circuit_breaker)
            print(f"{args.ticker}: {action}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Don't let one bad check (a network blip, a data hiccup) kill an
            # unattended autopilot run — log it and try again next interval.
            logger.exception("Check failed — will retry next interval")
            alerts.record_alert(
                "error", f"{args.ticker}: check failed — {e}", source="autopilot_loop",
            )

        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("\nAutopilot stopped.")
            break


if __name__ == "__main__":
    main()
