"""A broad, static universe of tickers for the portfolio bot to screen.

Not literally "every stock on the market" — that needs a paid listings feed
(index constituents, exchange symbol directories) this project doesn't have
wired up. This is a practical middle ground: a deliberately broad, varied
list spanning large caps down to small/micro caps and several sectors, so
screening picks *from* something wide rather than the user naming individual
tickers — see strategies/screener.py for the ranking that does the picking.

Extend this list freely; nothing else in the codebase assumes a fixed size
or order.
"""
from __future__ import annotations

DEFAULT_UNIVERSE = [
    # Mega/large-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "CRM", "ORCL", "ADBE",
    # Other large-cap, varied sectors
    "JPM", "V", "MA", "UNH", "HD", "PG", "KO", "PEP", "XOM", "CVX",
    "JNJ", "WMT", "DIS", "NFLX", "COST",
    # Mid-cap
    "SNAP", "PINS", "ETSY", "ROKU", "DKNG", "CHWY", "SOFI", "AFRM", "UPST", "COIN",
    # Small-cap / higher volatility
    "RIOT", "MARA", "PLTR", "LCID", "RIVN", "IONQ", "SMCI", "SOUN", "BBAI", "ACHR",
]
