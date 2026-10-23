"""Per-strategy virtual capital tracking for live/paper trading.

Multiple strategies can run against the same real IBKR paper account at
once (e.g. position trading on one ticker, swing trading on another).
broker.account_equity() returns the *whole account's* net liquidation —
identical for every strategy sharing that account — so it can't tell you
whether any one strategy is actually profitable on its own. This gives each
strategy its own virtual cash balance, seeded once from `initial_capital`
and updated on each real fill, so equity = cash + mark-to-market position
value can be compared strategy-to-strategy exactly like a backtest.

One small JSON file per strategy's log_dir (same directory trade_log.py
already writes to for that strategy), so tests can point at a temp
directory the same way they do for trade_log.
"""
from __future__ import annotations

import json
from pathlib import Path

LEDGER_FILENAME = "ledger.json"


def _ledger_path(log_dir: Path) -> Path:
    return log_dir / LEDGER_FILENAME


def get_cash(log_dir: Path, initial_capital: float) -> float:
    """Current virtual cash balance, seeding the ledger with `initial_capital` on first use."""
    path = _ledger_path(log_dir)
    if not path.exists():
        set_cash(log_dir, initial_capital)
        return initial_capital
    try:
        return float(json.loads(path.read_text())["cash"])
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        set_cash(log_dir, initial_capital)
        return initial_capital


def set_cash(log_dir: Path, cash: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    _ledger_path(log_dir).write_text(json.dumps({"cash": cash}))


def apply_buy(log_dir: Path, initial_capital: float, shares: float, price: float) -> float:
    """Deduct the cost of a buy fill from virtual cash. Returns the new balance."""
    cash = get_cash(log_dir, initial_capital) - shares * price
    set_cash(log_dir, cash)
    return cash


def apply_sell(log_dir: Path, initial_capital: float, shares: float, price: float) -> float:
    """Add the proceeds of a sell fill to virtual cash. Returns the new balance."""
    cash = get_cash(log_dir, initial_capital) + shares * price
    set_cash(log_dir, cash)
    return cash


def compute_equity(
    log_dir: Path, initial_capital: float, shares_held: float, last_price: float
) -> float:
    """Virtual cash plus the mark-to-market value of whatever's currently held."""
    return get_cash(log_dir, initial_capital) + shares_held * last_price


def compute_equity_multi(
    log_dir: Path, initial_capital: float, positions: dict[str, tuple[float, float]]
) -> float:
    """Like compute_equity, but for a portfolio holding several tickers at once.

    `positions` maps ticker -> (shares_held, last_price).
    """
    position_value = sum(shares * price for shares, price in positions.values())
    return get_cash(log_dir, initial_capital) + position_value
