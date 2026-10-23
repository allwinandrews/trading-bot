"""What a trade actually costs at IBKR — not a flat percentage.

The backtests originally assumed a flat 0.1% commission. Real fills on the
live paper account came in at ~1.0% (e.g. $0.88 on an $88 trade), because
IBKR's US equity fixed tier is $0.005/share with a $1.00 *minimum* and a 1%
of trade value *cap* — and at the small trade sizes a £100-500 account
produces, the 1% cap is what binds. That 10x understatement made every
backtest meaningfully more optimistic than reality.

A flat percentage can't express this: the effective rate is ~1% on a $80
trade, 0.5% on a $200 trade, and 0.1% on a $1,000 trade. So model the fee
schedule directly instead, and let the trade size determine the rate.

Rates per IBKR's published US equity "Fixed" pricing; they change from time
to time, so treat these as a good approximation rather than a guarantee.
"""
from __future__ import annotations

PER_SHARE = 0.005
MINIMUM = 1.00
MAX_PCT_OF_TRADE = 0.01


def ibkr_commission(shares: float, price: float) -> float:
    """Commission for one fill of `shares` at `price`, in account currency."""
    trade_value = shares * price
    if trade_value <= 0:
        return 0.0
    return min(max(shares * PER_SHARE, MINIMUM), trade_value * MAX_PCT_OF_TRADE)


def effective_rate(shares: float, price: float) -> float:
    """Commission as a fraction of trade value — useful for reporting."""
    trade_value = shares * price
    if trade_value <= 0:
        return 0.0
    return ibkr_commission(shares, price) / trade_value
