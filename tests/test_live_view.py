from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from execution import ledger, trade_log
from execution.live_view import build_live_view, countdown_text, fetch_live_marks


def test_countdown_counts_down_from_the_last_run():
    last_run = datetime(2026, 1, 1, 12, 0, 0)
    now = last_run + timedelta(seconds=90)

    assert countdown_text(last_run.isoformat(), interval_seconds=300, now=now) == "3m 30s"


def test_countdown_drops_the_minutes_when_under_a_minute():
    last_run = datetime(2026, 1, 1, 12, 0, 0)
    now = last_run + timedelta(seconds=280)

    assert countdown_text(last_run.isoformat(), interval_seconds=300, now=now) == "20s"


def test_countdown_says_due_now_when_overdue():
    last_run = datetime(2026, 1, 1, 12, 0, 0)
    now = last_run + timedelta(seconds=400)

    assert countdown_text(last_run.isoformat(), interval_seconds=300, now=now) == "due now"


def test_countdown_handles_never_run_and_garbage_timestamps():
    assert countdown_text(None, interval_seconds=300) == "waiting for first run"
    assert countdown_text("not-a-date", interval_seconds=300) == "unknown"


def _open_position(tmp_path, ticker="KO", entry=88.0, shares=1):
    trade_log.record_trade_open(ticker, entry_price=entry, shares=shares, log_dir=tmp_path)


def test_build_live_view_marks_holdings_at_broker_prices(tmp_path):
    _open_position(tmp_path, "KO", entry=88.0, shares=1)
    _open_position(tmp_path, "SMCI", entry=39.09, shares=2)
    ledger.set_cash(tmp_path, 257.0)

    view = build_live_view(tmp_path, initial_capital=500.0, marks={"KO": 90.0, "SMCI": 41.0})

    assert view["live"] is True
    assert view["equity"] == pytest.approx(257.0 + 90.0 + 82.0)
    ko = next(h for h in view["holdings"] if h["ticker"] == "KO")
    assert ko["pnl"] == pytest.approx(2.0)
    assert ko["pnl_pct"] == pytest.approx((90 / 88 - 1) * 100)


def test_build_live_view_reports_not_live_when_broker_unreachable(tmp_path):
    _open_position(tmp_path)
    ledger.set_cash(tmp_path, 257.0)

    view = build_live_view(tmp_path, initial_capital=500.0, marks=None)

    assert view["live"] is False
    # Equity is withheld rather than quietly reported as cash-only — the
    # same failure mode that once recorded a fake 48% crash.
    assert view["equity"] is None
    assert view["holdings"][0]["price"] is None


def test_build_live_view_withholds_equity_if_any_holding_is_unpriced(tmp_path):
    _open_position(tmp_path, "KO", entry=88.0, shares=1)
    _open_position(tmp_path, "SMCI", entry=39.09, shares=2)
    ledger.set_cash(tmp_path, 257.0)

    view = build_live_view(tmp_path, initial_capital=500.0, marks={"KO": 90.0})  # SMCI missing

    assert view["live"] is False
    assert view["equity"] is None


def test_build_live_view_with_no_positions_is_all_cash(tmp_path):
    ledger.set_cash(tmp_path, 500.0)

    view = build_live_view(tmp_path, initial_capital=500.0, marks={})

    assert view["holdings"] == []
    assert view["equity"] == pytest.approx(500.0)
    assert view["live"] is True


@patch("execution.live_view.IBBroker")
def test_fetch_live_marks_returns_none_when_gateway_is_down(mock_broker_cls):
    mock_broker_cls.return_value.__enter__.side_effect = TimeoutError("gateway down")

    assert fetch_live_marks() is None


@patch("execution.live_view.IBBroker")
def test_fetch_live_marks_returns_broker_marks(mock_broker_cls):
    broker = MagicMock()
    broker.live_marks.return_value = {"KO": 87.83}
    mock_broker_cls.return_value.__enter__.return_value = broker

    assert fetch_live_marks() == {"KO": 87.83}
