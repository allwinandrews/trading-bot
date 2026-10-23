from datetime import datetime

import pytest

from execution.trade_log import (
    load_equity_curve,
    load_trades,
    record_snapshot,
    record_trade_close,
    record_trade_open,
)


def test_load_equity_curve_empty_when_nothing_logged(tmp_path):
    curve = load_equity_curve(log_dir=tmp_path)
    assert curve.empty


def test_record_and_load_equity_curve(tmp_path):
    record_snapshot("AAPL", 100_000.0, timestamp=datetime(2026, 1, 1, 9, 30), log_dir=tmp_path)
    record_snapshot("AAPL", 101_000.0, timestamp=datetime(2026, 1, 1, 9, 35), log_dir=tmp_path)

    curve = load_equity_curve(log_dir=tmp_path)
    assert list(curve) == [100_000.0, 101_000.0]


def test_load_trades_empty_when_nothing_logged(tmp_path):
    trades = load_trades(log_dir=tmp_path)
    assert trades.empty


def test_record_trade_open_and_close_roundtrip(tmp_path):
    record_trade_open(
        "AAPL", entry_price=100.0, shares=50, entry_date=datetime(2026, 1, 1), log_dir=tmp_path
    )
    record_trade_close(
        "AAPL", exit_price=110.0, exit_reason="signal_exit",
        exit_date=datetime(2026, 1, 5), log_dir=tmp_path,
    )

    trades = load_trades(log_dir=tmp_path)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["shares"] == 50
    assert row["exit_reason"] == "signal_exit"
    assert row["return_pct"] == pytest.approx(10.0)


def test_record_trade_close_without_open_raises(tmp_path):
    with pytest.raises(ValueError):
        record_trade_close("AAPL", exit_price=110.0, exit_reason="signal_exit", log_dir=tmp_path)


def test_record_trade_close_only_closes_the_open_one(tmp_path):
    # A closed trade followed by a new open trade — close must target the open one, not the first.
    record_trade_open(
        "AAPL", entry_price=100.0, shares=10, entry_date=datetime(2026, 1, 1), log_dir=tmp_path
    )
    record_trade_close(
        "AAPL", exit_price=105.0, exit_reason="signal_exit",
        exit_date=datetime(2026, 1, 2), log_dir=tmp_path,
    )
    record_trade_open(
        "AAPL", entry_price=200.0, shares=5, entry_date=datetime(2026, 1, 3), log_dir=tmp_path
    )
    record_trade_close(
        "AAPL", exit_price=190.0, exit_reason="stop_loss",
        exit_date=datetime(2026, 1, 4), log_dir=tmp_path,
    )

    trades = load_trades(log_dir=tmp_path)
    assert len(trades) == 2
    assert trades.iloc[0]["return_pct"] == pytest.approx(5.0)
    assert trades.iloc[1]["return_pct"] == pytest.approx(-5.0)
