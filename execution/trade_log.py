"""Persistent local record of what the live/paper trading loop has done.

Two small CSVs under execution/live_data/ (gitignored, regenerable — same
spirit as data/*.csv and results/):
  - equity_history.csv: one row per check, whatever the action — builds a
    real equity curve over time as the loop runs.
  - trades.csv: one row per round-trip trade, opened on entry and filled in
    on exit. Same column shape as backtester/engine.py's trade records, so
    backtester.metrics.summarize() and ai.trade_journal work on live trades
    exactly as they do on backtest trades — no separate live-metrics code.

Every function takes an optional `log_dir` (defaults to the real one) so
tests can point at a temp directory instead of touching real state.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_LOG_DIR = Path(__file__).parent / "live_data"

EQUITY_COLUMNS = ["timestamp", "ticker", "equity"]
TRADE_COLUMNS = [
    "ticker", "entry_date", "entry_price", "exit_date", "exit_price",
    "shares", "return_pct", "exit_reason",
]


def _equity_path(log_dir: Path) -> Path:
    return log_dir / "equity_history.csv"


def _trades_path(log_dir: Path) -> Path:
    return log_dir / "trades.csv"


def record_snapshot(
    ticker: str, equity: float, timestamp: datetime | None = None, log_dir: Path = DEFAULT_LOG_DIR
) -> None:
    """Append one equity reading — call this on every check, not just trades."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = _equity_path(log_dir)
    row = pd.DataFrame([{
        "timestamp": (timestamp or datetime.now()).isoformat(),
        "ticker": ticker,
        "equity": equity,
    }], columns=EQUITY_COLUMNS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def record_trade_open(
    ticker: str, entry_price: float, shares: int, entry_date: datetime | None = None,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> None:
    """Append a new open trade (exit fields blank until record_trade_close)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = _trades_path(log_dir)
    row = pd.DataFrame([{
        "ticker": ticker,
        "entry_date": (entry_date or datetime.now()).isoformat(),
        "entry_price": entry_price,
        "exit_date": "",
        "exit_price": "",
        "shares": shares,
        "return_pct": "",
        "exit_reason": "",
    }], columns=TRADE_COLUMNS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def record_trade_close(
    ticker: str, exit_price: float, exit_reason: str, exit_date: datetime | None = None,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> None:
    """Fill in the most recent still-open trade for `ticker`.

    Raises ValueError if there's no open trade to close — a live-trading
    bug (closing a position the log never saw opened) shouldn't fail
    silently.
    """
    path = _trades_path(log_dir)
    if not path.exists():
        raise ValueError(f"No trades logged yet — can't close a position in {ticker}")

    trades = pd.read_csv(path, dtype=object, keep_default_na=False)
    open_mask = (trades["ticker"] == ticker) & (trades["exit_date"] == "")
    if not open_mask.any():
        raise ValueError(f"No open trade found for {ticker} to close")

    idx = trades[open_mask].index[-1]
    entry_price = float(trades.loc[idx, "entry_price"])
    return_pct = (exit_price - entry_price) / entry_price * 100

    trades.loc[idx, "exit_date"] = (exit_date or datetime.now()).isoformat()
    trades.loc[idx, "exit_price"] = exit_price
    trades.loc[idx, "return_pct"] = round(return_pct, 4)
    trades.loc[idx, "exit_reason"] = exit_reason

    trades.to_csv(path, index=False)


def load_equity_curve(log_dir: Path = DEFAULT_LOG_DIR) -> pd.Series:
    """Equity over time across all checks, indexed by timestamp. Empty if nothing logged yet."""
    path = _equity_path(log_dir)
    if not path.exists():
        return pd.Series(dtype=float, name="equity")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.set_index("timestamp")["equity"].rename("equity")


def load_trades(log_dir: Path = DEFAULT_LOG_DIR) -> pd.DataFrame:
    """Completed and open trades, shaped like backtester/engine.py's trade records.

    Numeric columns are cast properly; open trades (no exit yet) keep NaN
    exit fields rather than empty strings, so callers can filter with
    `trades["exit_date"].notna()`.
    """
    path = _trades_path(log_dir)
    if not path.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)

    trades = pd.read_csv(path)
    for col in ("entry_price", "exit_price", "shares", "return_pct"):
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    return trades
