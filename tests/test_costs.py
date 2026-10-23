import pytest

from backtester.costs import effective_rate, ibkr_commission


def test_small_trade_is_capped_at_one_percent():
    # Matches a real fill: $0.88 commission on a 1-share $88 KO trade.
    # Per-share ($0.005) and the $1 minimum would both exceed 1% here, so
    # the cap is what actually binds.
    assert ibkr_commission(shares=1, price=88.0) == pytest.approx(0.88)
    assert effective_rate(shares=1, price=88.0) == pytest.approx(0.01)


def test_medium_trade_pays_the_one_dollar_minimum():
    # $500 trade: per-share is trivial, 1% cap is $5, so the $1 floor wins.
    assert ibkr_commission(shares=10, price=50.0) == pytest.approx(1.00)
    assert effective_rate(shares=10, price=50.0) == pytest.approx(0.002)


def test_large_trade_pays_per_share():
    # 1,000 shares at $50: per-share ($5) beats the $1 minimum and sits
    # well under the 1% cap ($500).
    assert ibkr_commission(shares=1_000, price=50.0) == pytest.approx(5.00)


def test_effective_rate_falls_as_trade_size_grows():
    rates = [effective_rate(shares=n, price=50.0) for n in (1, 10, 100, 1_000)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]


def test_zero_or_negative_trade_costs_nothing():
    assert ibkr_commission(shares=0, price=50.0) == 0.0
    assert effective_rate(shares=0, price=50.0) == 0.0
