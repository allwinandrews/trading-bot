import pandas as pd
import pytest

from backtester.engine import run_backtest, run_buy_and_hold

DATES = pd.date_range("2024-01-01", periods=5, freq="B")


def _prices(lows):
    """5-bar synthetic OHLC series; only Low varies between the two test scenarios."""
    return pd.DataFrame(
        {
            "Open": [100, 100, 100, 101, 103],
            "High": [101, 101, 102, 103, 104],
            "Low": lows,
            "Close": [100, 100, 101, 102, 103],
        },
        index=DATES,
    )


def test_signal_exit_when_stop_never_hit():
    prices = _prices(lows=[99, 99, 99, 99, 102])
    signals = pd.Series([0, 1, 1, 0, 0], index=DATES)

    result = run_backtest(
        prices, signals,
        initial_capital=100_000, stop_loss_pct=0.02, risk_per_trade_pct=0.01, commission_pct=0.0,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "signal_exit"
    assert trade["entry_price"] == pytest.approx(100)   # bar2's Open
    assert trade["exit_price"] == pytest.approx(103)     # bar4's Open
    assert trade["shares"] == 500                        # $1000 risk / $2 stop distance

    # Price only went up, so the strategy should have made money.
    assert result.equity_curve.iloc[-1] > 100_000


def test_stop_loss_triggers_on_intrabar_low():
    # Same setup, but bar3's Low (97) breaches the 2% stop (98) after entry at 100.
    prices = _prices(lows=[99, 99, 99, 97, 102])
    signals = pd.Series([0, 1, 1, 0, 0], index=DATES)

    result = run_backtest(
        prices, signals,
        initial_capital=100_000, stop_loss_pct=0.02, risk_per_trade_pct=0.01, commission_pct=0.0,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(98)  # 2% below the $100 entry


def test_never_in_position_when_signal_always_flat():
    prices = _prices(lows=[99, 99, 99, 99, 99])
    signals = pd.Series([0, 0, 0, 0, 0], index=DATES)

    result = run_backtest(prices, signals, initial_capital=100_000)

    assert result.trades.empty
    assert (result.equity_curve == 100_000).all()


def test_buy_and_hold_tracks_close_price():
    prices = _prices(lows=[99, 99, 99, 99, 99])
    equity = run_buy_and_hold(prices, initial_capital=100_000)

    # Bought at bar0's Open (100) with all capital -> 1000 shares, $0 leftover.
    assert equity.iloc[0] == pytest.approx(100_000)
    assert equity.iloc[-1] == pytest.approx(1000 * prices["Close"].iloc[-1])
