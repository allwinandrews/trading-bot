"""Historical price data loading, with a simple local CSV cache.

Single source of truth for getting price data into the rest of the project —
the backtester, notebooks, and (later) the paper-trading loop should all go
through here rather than each calling yfinance directly.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent

# Real incident: a background worker hung for two full days on a single
# yf.download() call despite yfinance's own internal timeout=10 default —
# that default doesn't reliably bound an internal retry/rate-limit loop.
# This is an external wall-clock backstop so a stuck download can never
# again silently freeze a process (including the live trading bot) with no
# indication beyond "still running" in the OS process list.
DOWNLOAD_TIMEOUT_SECONDS = 30


def _download_with_watchdog(*args, **kwargs):
    """Run yf.download with a hard external timeout.

    Python can't forcibly kill a thread, so on timeout the underlying call
    is abandoned (left running in the background, harmless — it just never
    reports back) rather than awaited — the point is that the *caller*
    reliably gets control back, not that the stuck request stops.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(yf.download, *args, **kwargs)
    try:
        return future.result(timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"Price data download timed out after {DOWNLOAD_TIMEOUT_SECONDS}s — "
            "Yahoo Finance may be rate-limiting or unresponsive."
        ) from None
    finally:
        pool.shutdown(wait=False)


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    return CACHE_DIR / f"{safe_ticker}_{start}_{end}_{interval}.csv"


def load_price_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data for `ticker` between `start` and `end` (YYYY-MM-DD).

    Caches to a CSV under data/ so repeated runs (e.g. re-running a backtest
    while tuning parameters) don't re-hit the network every time.
    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
    """
    path = _cache_path(ticker, start, end, interval)

    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = _download_with_watchdog(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker} between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    # yfinance can return MultiIndex columns even for a single ticker on some
    # versions — flatten to the plain OHLCV columns we expect everywhere else.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    if use_cache:
        df.to_csv(path)

    return df
