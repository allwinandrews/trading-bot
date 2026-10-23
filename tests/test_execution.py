from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from execution import ledger, trade_log
from execution.paper_trader import _close_orphaned_log_entry, _ensure_stop_is_resting, decide_action, run_once


def test_decide_action_enters_when_flat_and_signal_on():
    assert decide_action(in_position=False, today_signal=1) == "enter"


def test_decide_action_exits_when_holding_and_signal_off():
    assert decide_action(in_position=True, today_signal=0) == "exit"


def test_decide_action_holds_when_flat_and_signal_off():
    assert decide_action(in_position=False, today_signal=0) == "hold"


def test_decide_action_holds_when_holding_and_signal_on():
    assert decide_action(in_position=True, today_signal=1) == "hold"


@patch("execution.paper_trader.alerts")
@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_refuses_to_trade_on_nan_price_data(mock_load_price_data, mock_trade_log, mock_alerts, tmp_path):
    # Reproduces the real incident: a connectivity drop returned NaN price
    # data, which was previously mis-read as a genuine "exit" signal and
    # triggered a real, unintended sell order against an open position.
    # alerts is mocked (not left to hit the real alerts.json) so running
    # this test doesn't pollute the actual dashboard's alert banner.
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [100, 101, 102, 103, float("nan")],
    }, index=dates)
    mock_load_price_data.return_value = prices

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.return_value = 1500  # currently holding a real position

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "data_error"
    broker.sell_market.assert_not_called()
    broker.buy_with_stop.assert_not_called()
    broker.cancel_open_orders.assert_not_called()
    mock_trade_log.record_trade_close.assert_not_called()
    mock_trade_log.record_trade_open.assert_not_called()
    mock_trade_log.record_snapshot.assert_called_once()


@patch("execution.paper_trader.alerts")
def test_ensure_stop_is_resting_rearms_when_stop_missing(mock_alerts):
    # Reproduces the real incident: IBKR's paper preset silently forces
    # stop orders to TIF=DAY, so a position held past one trading session
    # can end up with no resting stop and nothing noticing.
    broker = MagicMock()
    broker.open_orders.return_value = []  # no resting stop at all
    broker.all_positions.return_value = [{"symbol": "AAPL", "shares": 1500, "avg_cost": 329.62266}]

    _ensure_stop_is_resting(broker, "AAPL", stop_loss_pct=0.02)

    broker.place_stop.assert_called_once()
    args = broker.place_stop.call_args[0]
    assert args[0] == "AAPL"
    assert args[1] == 1500
    assert args[2] == pytest.approx(323.03, abs=0.01)
    mock_alerts.record_alert.assert_called_once()


@patch("execution.paper_trader.alerts")
def test_ensure_stop_is_resting_does_nothing_when_stop_already_resting(mock_alerts):
    broker = MagicMock()
    broker.open_orders.return_value = [
        {"symbol": "AAPL", "action": "SELL", "type": "STP", "quantity": 1500, "aux_price": 323.03,
         "status": "PreSubmitted", "filled": 0}
    ]
    broker.all_positions.return_value = [{"symbol": "AAPL", "shares": 1500, "avg_cost": 329.62266}]

    _ensure_stop_is_resting(broker, "AAPL", stop_loss_pct=0.02)

    broker.place_stop.assert_not_called()
    mock_alerts.record_alert.assert_not_called()


def test_ensure_stop_is_resting_does_nothing_when_flat():
    broker = MagicMock()
    broker.open_orders.return_value = []
    broker.all_positions.return_value = []

    _ensure_stop_is_resting(broker, "AAPL", stop_loss_pct=0.02)

    broker.place_stop.assert_not_called()


@patch("execution.paper_trader.alerts")
@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_rearms_missing_stop_while_holding(mock_load_price_data, mock_trade_log, mock_alerts, tmp_path):
    # End-to-end version of the same gap: a "hold" check with an open
    # position but no resting stop order should re-arm one automatically
    # rather than silently leaving the position unprotected until the next
    # signal-driven exit (which could be days away). alerts is mocked so
    # this doesn't write a real (fake) alert into the live alerts.json.
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [100, 101, 102, 103, 104],
    }, index=dates)
    mock_load_price_data.return_value = prices

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.return_value = 1500
    broker.open_orders.return_value = []  # stop expired / missing
    broker.all_positions.return_value = [{"symbol": "AAPL", "shares": 1500, "avg_cost": 100.0}]

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "hold"
    broker.place_stop.assert_called_once()
    broker.sell_market.assert_not_called()


def _bearish_prices():
    # short_window=2/long_window=3: last short_ma(100.0)=mean(101,99)=100.0
    # < long_ma=mean(102,101,99)=100.67 -> today's signal is 0 (exit).
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame({
        "Open": [100, 101, 102, 101, 99],
        "High": [101, 102, 103, 102, 100],
        "Low": [99, 100, 101, 100, 98],
        "Close": [100, 101, 102, 101, 99],
    }, index=dates)


@patch("execution.paper_trader.alerts")
@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_does_not_record_close_when_exit_sell_never_fills(
    mock_load_price_data, mock_trade_log, mock_alerts, tmp_path
):
    # Reproduces the real incident: a market sell placed outside trading
    # hours queued instead of filling. The old code recorded the trade as
    # closed anyway (falling back to an approximate price) even though the
    # position was still fully open — and had already cancelled its stop
    # moments earlier, leaving it both unprotected and wrongly marked flat.
    # alerts is mocked so the re-arm this triggers doesn't write a real
    # (fake) alert into the live alerts.json.
    mock_load_price_data.return_value = _bearish_prices()

    sell_trade = MagicMock()
    sell_trade.orderStatus.filled = 0

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.return_value = 1500
    broker.open_orders.return_value = []  # cancel_open_orders already cleared the stop
    broker.all_positions.return_value = [{"symbol": "AAPL", "shares": 1500, "avg_cost": 100.0}]
    broker.sell_market.return_value = sell_trade

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "hold"
    mock_trade_log.record_trade_close.assert_not_called()
    broker.place_stop.assert_called_once()  # re-armed instead of left unprotected


@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_records_close_when_exit_sell_fills(mock_load_price_data, mock_trade_log, tmp_path):
    mock_load_price_data.return_value = _bearish_prices()

    sell_trade = MagicMock()
    sell_trade.orderStatus.filled = 1500
    sell_trade.orderStatus.avgFillPrice = 98.5

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.return_value = 1500
    broker.sell_market.return_value = sell_trade

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "exit"
    mock_trade_log.record_trade_close.assert_called_once()
    assert mock_trade_log.record_trade_close.call_args.kwargs["exit_price"] == 98.5
    broker.place_stop.assert_not_called()  # fully closed — nothing left to protect


def _bullish_prices():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [100, 101, 102, 103, 104],
    }, index=dates)


@patch("execution.alerts.record_alert")
@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_reconciles_when_buy_raises_but_shares_actually_exist(
    mock_load_price_data, mock_trade_log, mock_record_alert, tmp_path
):
    # Reproduces the exact race found live: buy_with_stop raised (its own
    # cancel-confirmation couldn't confirm in time), but the order actually
    # filled for real moments later. Without this reconciliation, the bot
    # would hold a real position with zero record of it and zero stop.
    mock_load_price_data.return_value = _bullish_prices()

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.side_effect = [0, 500]  # flat when checked pre-entry, filled by the time we recheck
    broker.buy_with_stop.side_effect = RuntimeError("Buy order for AAPL had not filled after 15.0s")
    broker.open_orders.return_value = []
    broker.all_positions.return_value = [{"symbol": "AAPL", "shares": 500, "avg_cost": 100.0}]

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "enter"
    mock_trade_log.record_trade_open.assert_called_once()
    assert mock_trade_log.record_trade_open.call_args.kwargs["shares"] == 500
    broker.place_stop.assert_called_once()  # re-armed via _ensure_stop_is_resting


@patch("execution.paper_trader.trade_log")
@patch("execution.paper_trader.load_price_data")
def test_run_once_holds_when_buy_raises_and_no_shares_exist(mock_load_price_data, mock_trade_log, tmp_path):
    mock_load_price_data.return_value = _bullish_prices()

    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.position_shares.return_value = 0  # genuinely never filled
    broker.buy_with_stop.side_effect = RuntimeError("Buy order for AAPL had not filled after 15.0s")

    action = run_once(broker, ticker="AAPL", short_window=2, long_window=3, log_dir=tmp_path)

    assert action == "hold"
    mock_trade_log.record_trade_open.assert_not_called()


@patch("execution.alerts.record_alert")
def test_close_orphaned_log_entry_does_nothing_when_stop_still_resting(mock_alerts, tmp_path):
    # Reproduces the real incident: a mid-cycle connectivity hiccup ("positions
    # request timed out" then "Connectivity...lost") made position_shares()
    # read 0 for a genuinely-still-held position, with its protective stop
    # still resting untouched. That combination must be treated as an
    # unreliable read, not a real close.
    trade_log.record_trade_open("AAPL", entry_price=100.0, shares=10, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 0.0)

    broker = MagicMock()
    broker.open_orders.return_value = [
        {"symbol": "AAPL", "action": "SELL", "type": "STP", "quantity": 10, "aux_price": 95.0,
         "status": "PreSubmitted", "filled": 0}
    ]

    _close_orphaned_log_entry(broker, "AAPL", last_price=99.0, log_dir=tmp_path, initial_capital=1000.0)

    trades = trade_log.load_trades(log_dir=tmp_path)
    assert pd.isna(trades.loc[0, "exit_date"])  # still open
    assert ledger.get_cash(tmp_path, initial_capital=1000.0) == 0.0  # untouched


@patch("execution.alerts.record_alert")
def test_close_orphaned_log_entry_closes_and_credits_logged_shares(mock_alerts, tmp_path):
    trade_log.record_trade_open("AAPL", entry_price=100.0, shares=10, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 0.0)

    broker = MagicMock()
    broker.open_orders.return_value = []  # stop genuinely gone, not just unreported

    _close_orphaned_log_entry(broker, "AAPL", last_price=95.0, log_dir=tmp_path, initial_capital=1000.0)

    trades = trade_log.load_trades(log_dir=tmp_path)
    assert trades.loc[0, "exit_date"] is not pd.NaT
    assert trades.loc[0, "exit_price"] == 95.0
    # The real bug: this used to credit 0 shares regardless of how many
    # were actually logged as held, silently understating cash forever.
    assert ledger.get_cash(tmp_path, initial_capital=1000.0) == pytest.approx(10 * 95.0)
