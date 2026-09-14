import pytest

from risk.position_sizing import shares_for_fixed_risk
from risk.stops import fixed_stop_price, trailing_stop_price


def test_fixed_stop_price():
    assert fixed_stop_price(100, 0.02) == pytest.approx(98)


def test_fixed_stop_price_rejects_bad_pct():
    with pytest.raises(ValueError):
        fixed_stop_price(100, 1.5)


def test_trailing_stop_price():
    assert trailing_stop_price(110, 0.02) == pytest.approx(107.8)


def test_shares_for_fixed_risk_basic():
    # $100k capital, 1% risk = $1000 at risk. Entry 100, stop 98 -> $2/share risk -> 500 shares.
    shares = shares_for_fixed_risk(capital=100_000, entry_price=100, stop_price=98, risk_per_trade_pct=0.01)
    assert shares == 500


def test_shares_for_fixed_risk_capped_by_capital():
    # Tiny stop distance would imply a huge share count — must be capped by available capital.
    shares = shares_for_fixed_risk(capital=1_000, entry_price=100, stop_price=99.9, risk_per_trade_pct=0.5)
    assert shares == 10  # floor(1000 / 100)


def test_shares_for_fixed_risk_rejects_stop_above_entry():
    with pytest.raises(ValueError):
        shares_for_fixed_risk(capital=100_000, entry_price=100, stop_price=101, risk_per_trade_pct=0.01)
