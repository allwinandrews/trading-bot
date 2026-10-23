"""Live account view for the dashboard — real broker prices, not daily bars.

The trading logic deliberately runs on daily closes (that's what the
strategy was designed and backtested on), but showing those same stale
closes on the dashboard made it disagree with what IBKR itself displays,
which is confusing when you're trying to answer "am I up or down right
now". This pulls the broker's own current marks for display only — it never
feeds a trading decision.

Kept out of the view files so it can be tested without Streamlit, and so a
dead IB Gateway degrades to "prices unavailable" instead of a stack trace
on the page.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from execution import ledger, trade_log
from execution.broker import IBBroker

logger = logging.getLogger(__name__)

# Distinct from every bot client id (1 = legacy autopilot, 3 = legacy swing,
# 9 = portfolio bot) so a dashboard refresh never collides with a live
# trading connection.
DASHBOARD_CLIENT_ID = 2


def countdown_text(last_run_iso: str | None, interval_seconds: int, now: datetime | None = None) -> str:
    """Human-readable "when's the next one" for a fixed-interval job.

    The dashboard refreshes far more often than the bot actually runs, so
    without this the page looks live while showing numbers that are minutes
    old — this makes the gap visible instead of implied.
    """
    if not last_run_iso:
        return "waiting for first run"
    try:
        last_run = datetime.fromisoformat(str(last_run_iso))
    except ValueError:
        return "unknown"

    remaining = interval_seconds - (( now or datetime.now()) - last_run).total_seconds()
    if remaining <= 0:
        return "due now"
    minutes, seconds = divmod(int(remaining), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def fetch_live_marks(client_id: int = DASHBOARD_CLIENT_ID) -> dict[str, float] | None:
    """Current market price per held symbol, or None if the broker is unreachable."""
    try:
        with IBBroker(client_id=client_id) as broker:
            return broker.live_marks()
    except Exception as e:  # noqa: BLE001 — a dead Gateway must not break the page
        logger.warning("Live marks unavailable: %s", e)
        return None


def build_live_view(log_dir: Path, initial_capital: float, marks: dict[str, float] | None) -> dict:
    """Combine this bot's own open trades with live broker prices.

    Returns {"holdings": [...], "equity": float|None, "cash": float,
    "live": bool}. `equity` is this strategy's own virtual equity (its cash
    plus its positions at live prices) — not the whole IBKR account, which
    is shared with other strategies and orders of magnitude larger.
    """
    cash = ledger.get_cash(log_dir, initial_capital)
    trades = trade_log.load_trades(log_dir=log_dir)
    open_trades = trades[trades["exit_date"].isna()] if not trades.empty else trades

    holdings = []
    position_value = 0.0
    for _, row in open_trades.iterrows():
        ticker, shares, entry = row["ticker"], float(row["shares"]), float(row["entry_price"])
        mark = (marks or {}).get(ticker)
        value = shares * mark if mark else None
        if value is not None:
            position_value += value
        holdings.append({
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry,
            "price": mark,
            "value": value,
            "pnl": (mark - entry) * shares if mark else None,
            "pnl_pct": ((mark / entry) - 1) * 100 if mark else None,
        })

    every_holding_priced = all(h["price"] is not None for h in holdings)
    return {
        "holdings": holdings,
        "cash": cash,
        "equity": cash + position_value if every_holding_priced else None,
        "live": marks is not None and every_holding_priced,
    }
