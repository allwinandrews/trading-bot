"""Position sizing.

Fixed-fractional sizing: risk a fixed % of capital per trade, sized off the
stop-loss distance — not gut feel, and not the whole account on one signal.
This is the standard, sensible default (Kelly-style sizing is easy to
over-apply and is intentionally not offered here yet).
"""
from __future__ import annotations

import math


def shares_for_fixed_risk(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade_pct: float = 0.01,
) -> int:
    """How many whole shares to buy so a stop-out loses ~risk_per_trade_pct of capital.

    Also caps the position so it never costs more than the available capital,
    even if the stop distance is very tight.
    """
    if entry_price <= 0 or stop_price < 0:
        raise ValueError("entry_price and stop_price must be positive")
    if not (0 < risk_per_trade_pct <= 1):
        raise ValueError("risk_per_trade_pct must be between 0 and 1")

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        raise ValueError("stop_price must be below entry_price for a long position")

    dollars_at_risk = capital * risk_per_trade_pct
    shares_by_risk = math.floor(dollars_at_risk / risk_per_share)
    shares_by_capital = math.floor(capital / entry_price)

    return max(0, min(shares_by_risk, shares_by_capital))
