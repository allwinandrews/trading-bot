from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from execution import ledger, screen_report, trade_log
from execution.portfolio_trader import run_once

DATES = pd.date_range("2024-01-01", periods=6, freq="B")
_KWARGS = dict(short_window=2, long_window=3, momentum_window=2, regime_window=2)


def _prices(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=DATES[: len(closes)])


UPTREND_MARKET = _prices([100, 101, 102, 103, 104, 106])
DOWNTREND = _prices([106, 104, 102, 101, 100, 98])
FAST_UPTREND = _prices([100, 102, 104, 106, 108, 112])
SLOW_UPTREND = _prices([100, 100.5, 101, 101.5, 102, 102.5])


def _mock_broker():
    positions = {}
    broker = MagicMock()
    broker.position_shares.side_effect = lambda t: positions.get(t, 0)

    def fake_buy(ticker, shares, stop_price):
        positions[ticker] = shares
        return MagicMock(), MagicMock()

    broker.buy_with_stop.side_effect = fake_buy
    broker.open_orders.return_value = []
    broker.all_positions.side_effect = lambda: [
        {"symbol": t, "shares": s, "avg_cost": 100.0} for t, s in positions.items() if s > 0
    ]
    broker._positions = positions  # exposed for test setup/inspection
    return broker


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_enters_top_ranked_candidates_up_to_max_positions(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"FAST": FAST_UPTREND, "SLOW": SLOW_UPTREND, "BAD": DOWNTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()

    result = run_once(
        broker, universe=["FAST", "SLOW", "BAD"], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert set(result["entered"]) == {"FAST", "SLOW"}
    assert "BAD" not in result["entered"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert set(trades["ticker"]) == {"FAST", "SLOW"}
    assert ledger.get_cash(tmp_path, initial_capital=10_000.0) < 10_000.0


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_records_its_reasoning_for_the_research_page(mock_load, mock_alerts, tmp_path):
    # The Research page shows what the bot actually saw and decided; that
    # only works if each cycle persists its own reasoning rather than
    # throwing it away when the subprocess finishes.
    prices_by_ticker = {"FAST": FAST_UPTREND, "BAD": DOWNTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()

    run_once(
        broker, universe=["FAST", "BAD"], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    report = screen_report.load_screen_report(tmp_path)
    assert report["regime_ok"] is True
    assert report["universe_size"] == 2
    assert [c["ticker"] for c in report["candidates"]] == ["FAST"]
    assert report["excluded"]["BAD"] == "failed the trend/momentum screen"
    assert "bought" in report["decisions"]["FAST"]
    assert report["entered"] == ["FAST"]


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_explains_a_sentiment_veto_in_its_report(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"FAST": FAST_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()
    sentiment = lambda ticker: {"verdict": "negative", "reason": "Fraud lawsuit filed."}

    run_once(
        broker, universe=["FAST"], max_positions=2, sentiment_filter=sentiment,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    report = screen_report.load_screen_report(tmp_path)
    assert "vetoed on news sentiment" in report["decisions"]["FAST"]
    assert "Fraud lawsuit" in report["decisions"]["FAST"]


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_respects_negative_sentiment_veto(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"FAST": FAST_UPTREND, "SLOW": SLOW_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()
    sentiment_filter = lambda ticker: {"verdict": "negative"} if ticker == "FAST" else {"verdict": "neutral"}

    result = run_once(
        broker, universe=["FAST", "SLOW"], max_positions=2,
        sentiment_filter=sentiment_filter,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert "FAST" in result["skipped"]
    assert "FAST" not in result["entered"]
    assert "SLOW" in result["entered"]


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_never_enters_ticker_already_held_by_another_strategy(mock_load, mock_alerts, tmp_path):
    # Reproduces the exact real risk this guards against: this portfolio
    # bot's own trade log has never opened AAPL, but the real IBKR account
    # already holds AAPL for a different strategy sharing that account.
    prices_by_ticker = {"AAPL": FAST_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()
    broker._positions["AAPL"] = 1500  # already held by a different, unrelated strategy

    result = run_once(
        broker, universe=["AAPL"], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["entered"] == []
    broker.buy_with_stop.assert_not_called()


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_exits_held_ticker_on_signal_off(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"BAD": DOWNTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    trade_log.record_trade_open("BAD", entry_price=106.0, shares=10, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 9_000.0)

    broker = _mock_broker()
    broker._positions["BAD"] = 10
    sell_trade = MagicMock()
    sell_trade.orderStatus.filled = 10
    sell_trade.orderStatus.avgFillPrice = 98.0
    broker.sell_market.return_value = sell_trade

    result = run_once(
        broker, universe=[], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["exited"] == ["BAD"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert trades.loc[0, "exit_date"] is not pd.NaT
    assert ledger.get_cash(tmp_path, initial_capital=10_000.0) == pytest.approx(9_000.0 + 10 * 98.0)


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_does_not_close_when_flat_but_stop_still_resting(mock_load, mock_alerts, tmp_path):
    # Reproduces the real incident: a mid-cycle connectivity hiccup ("positions
    # request timed out" then "Connectivity...lost") made position_shares()
    # read 0 for a genuinely-still-held position, in the same cycle its stop
    # was still resting untouched. That combination must be treated as an
    # unreliable read, not a real close — KO and SMCI both got wrongly
    # recorded as "closed" this way in production.
    prices_by_ticker = {"KO": DOWNTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    trade_log.record_trade_open("KO", entry_price=88.25, shares=1, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 400.0)

    broker = _mock_broker()
    # broker._positions["KO"] left unset -> position_shares() reads 0,
    # simulating the bad snapshot from the connectivity hiccup.
    broker.open_orders.return_value = [
        {"symbol": "KO", "action": "SELL", "type": "STP", "quantity": 1, "aux_price": 83.84,
         "status": "PreSubmitted", "filled": 0}
    ]

    result = run_once(
        broker, universe=[], max_positions=2,
        initial_capital=500.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["exited"] == []
    assert "KO" in result["held"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert pd.isna(trades.loc[0, "exit_date"])
    assert ledger.get_cash(tmp_path, initial_capital=500.0) == pytest.approx(400.0)  # untouched


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_closes_and_credits_logged_shares_when_stop_genuinely_gone(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"KO": DOWNTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    trade_log.record_trade_open("KO", entry_price=88.25, shares=1, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 400.0)

    broker = _mock_broker()
    broker.open_orders.return_value = []  # stop genuinely gone, not just unreported

    result = run_once(
        broker, universe=[], max_positions=2,
        initial_capital=500.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["exited"] == ["KO"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert trades.loc[0, "exit_date"] is not pd.NaT
    last_close = float(DOWNTREND["Close"].iloc[-1])
    # The real bug: this used to credit 0 shares regardless of how many
    # were actually logged as held, silently understating cash forever.
    assert ledger.get_cash(tmp_path, initial_capital=500.0) == pytest.approx(400.0 + 1 * last_close)


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_reconciles_when_buy_raises_but_shares_actually_exist(mock_load, mock_alerts, tmp_path):
    # Same real race as paper_trader.py's equivalent test: buy_with_stop
    # raised, but the order actually filled moments later. Without this,
    # the portfolio bot would hold an untracked, unprotected position.
    prices_by_ticker = {"FAST": FAST_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()
    broker.buy_with_stop.side_effect = RuntimeError("Buy order for FAST had not filled after 15.0s")
    broker._positions["FAST"] = 23  # what all_positions() reports once the fill is real

    calls = {"FAST": 0}

    def position_shares_after_failed_buy(t):
        # First check (the "already held?" exclusion guard) sees 0 — not
        # yet entered. Every check after that (the post-raise reconciliation
        # re-check, then the final equity calc) sees the real fill.
        if t != "FAST":
            return broker._positions.get(t, 0)
        calls["FAST"] += 1
        return 0 if calls["FAST"] == 1 else 23

    broker.position_shares.side_effect = position_shares_after_failed_buy

    result = run_once(
        broker, universe=["FAST"], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["entered"] == ["FAST"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert trades.loc[0, "ticker"] == "FAST"
    assert trades.loc[0, "shares"] == 23
    broker.place_stop.assert_called_once()


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_skips_when_buy_raises_and_no_shares_exist(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"FAST": FAST_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    broker = _mock_broker()
    broker.buy_with_stop.side_effect = RuntimeError("Buy order for FAST had not filled after 15.0s")

    result = run_once(
        broker, universe=["FAST"], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["entered"] == []
    assert "FAST" in result["skipped"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    assert trades.empty


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_equity_does_not_crash_when_a_holdings_price_is_unavailable(mock_load, mock_alerts, tmp_path):
    # Reproduces a real bug: a held position whose price couldn't be fetched
    # that cycle dropped out of the equity sum entirely — valuing it at zero
    # and recording a 48% "crash" in the equity curve that never happened.
    # It should fall back to the entry price instead: stale, but honest.
    def prices_for(ticker, **kw):
        if ticker == "SPY":
            return UPTREND_MARKET
        raise ValueError(f"No data returned for {ticker}")  # price feed down for holdings

    mock_load.side_effect = prices_for

    trade_log.record_trade_open("FAST", entry_price=100.0, shares=2, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 300.0)

    broker = _mock_broker()
    broker._positions["FAST"] = 2

    run_once(
        broker, universe=[], max_positions=3,
        initial_capital=500.0, log_dir=tmp_path, **_KWARGS,
    )

    equity_curve = trade_log.load_equity_curve(log_dir=tmp_path)
    # £300 cash + 2 shares marked at the £100 entry price = £500, NOT £300.
    assert equity_curve.iloc[-1] == pytest.approx(500.0)


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_rearms_stop_for_held_position_still_trending(mock_load, mock_alerts, tmp_path):
    prices_by_ticker = {"FAST": FAST_UPTREND, "SPY": UPTREND_MARKET}
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    trade_log.record_trade_open("FAST", entry_price=100.0, shares=10, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 9_000.0)

    broker = _mock_broker()
    broker._positions["FAST"] = 10
    broker.open_orders.return_value = []  # no resting stop -> should be re-armed

    result = run_once(
        broker, universe=[], max_positions=2,
        initial_capital=10_000.0, log_dir=tmp_path, **_KWARGS,
    )

    assert result["held"] == ["FAST"]
    broker.place_stop.assert_called_once()


@patch("execution.alerts.record_alert")
@patch("execution.portfolio_trader.load_price_data")
def test_run_once_lets_the_last_open_slot_use_more_than_a_flat_fraction_of_cash(mock_load, mock_alerts, tmp_path):
    # With 2 of 3 slots already filled, sizing the last slot off the same
    # fixed 25%-of-cash fraction used when all 3 slots were still competing
    # for it would leave a $112 candidate unaffordable against $300 cash
    # (25% = $75) even though the whole $300 is available for this one
    # remaining slot. Sizing should scale up as fewer slots remain.
    prices_by_ticker = {
        "HELD1": FAST_UPTREND, "HELD2": FAST_UPTREND, "CAND": FAST_UPTREND, "SPY": UPTREND_MARKET,
    }
    mock_load.side_effect = lambda ticker, **kw: prices_by_ticker[ticker]

    trade_log.record_trade_open("HELD1", entry_price=100.0, shares=1, log_dir=tmp_path)
    trade_log.record_trade_open("HELD2", entry_price=100.0, shares=1, log_dir=tmp_path)
    ledger.set_cash(tmp_path, 300.0)

    broker = _mock_broker()
    broker._positions["HELD1"] = 1
    broker._positions["HELD2"] = 1

    result = run_once(
        broker, universe=["CAND"], max_positions=3,
        initial_capital=300.0, log_dir=tmp_path, **_KWARGS,
    )

    assert "CAND" in result["entered"]
    trades = trade_log.load_trades(log_dir=tmp_path)
    cand_row = trades[trades["ticker"] == "CAND"].iloc[0]
    assert cand_row["shares"] == 2
