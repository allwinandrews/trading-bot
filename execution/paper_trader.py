"""Live signal loop: check today's signal for one ticker and act on it once.

Mirrors backtester/engine.py's per-bar logic (signal -> stop -> size ->
order) but against a live broker connection instead of a simulation. This
module does not loop or sleep itself — call `run_once` on a schedule (e.g. a
scheduled task shortly after each day's open), once per ticker per day.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path

from data.loader import load_price_data
from execution import alerts, ledger, trade_log
from execution.broker import IBBroker
from risk.position_sizing import shares_for_fixed_risk
from risk.stops import DailyLossCircuitBreaker, fixed_stop_price
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW, generate_signals

logger = logging.getLogger(__name__)

DEFAULT_STOP_LOSS_PCT = 0.02
DEFAULT_RISK_PER_TRADE_PCT = 0.01
DEFAULT_INITIAL_CAPITAL = 100_000.0

# Extra trading days of history requested beyond the long window, so the
# moving averages have room to warm up even across weekends/holidays.
_LOOKBACK_BUFFER_DAYS = 30


def _has_resting_stop(broker: IBBroker, ticker: str) -> bool:
    """Is a protective SELL stop currently resting for `ticker`?

    Shared by _ensure_stop_is_resting (re-arm if missing) and the
    orphaned-close detection below (a stop still resting is strong evidence
    a "position reads as flat" signal is a bad data snapshot, not a real
    close — see the incident note on that check).
    """
    open_stop_symbols = {
        o["symbol"] for o in broker.open_orders()
        if o["type"] in ("STP", "STP LMT") and o["action"] == "SELL"
    }
    return ticker in open_stop_symbols


def _ensure_stop_is_resting(broker: IBBroker, ticker: str, stop_loss_pct: float) -> None:
    """If we hold `ticker` but no protective stop order is resting for it, re-arm one.

    Discovered the hard way: IBKR's paper-account preset silently forces
    stop orders to TIF=DAY regardless of what's requested, and a DAY order
    auto-cancels at end of session — so a position that survives past one
    trading day could quietly end up completely unprotected with no error,
    no alert, nothing. broker.buy_with_stop/place_stop now request GTC
    explicitly, but this check exists so a stop going missing for *any*
    reason (that fix being wrong, a manual cancel, IBKR-side weirdness)
    gets caught and fixed automatically instead of relying on nobody
    noticing being the safety net.
    """
    if _has_resting_stop(broker, ticker):
        return

    position = next((p for p in broker.all_positions() if p["symbol"] == ticker), None)
    if position is None or position["shares"] <= 0:
        return

    stop_price = fixed_stop_price(position["avg_cost"], stop_loss_pct)
    broker.place_stop(ticker, int(position["shares"]), stop_price)
    logger.warning(
        "%s: held %d shares with no resting stop order — re-armed one @ %.2f",
        ticker, int(position["shares"]), stop_price,
    )
    alerts.record_alert(
        "warning",
        f"{ticker}: was holding a position with no protective stop resting (likely daily "
        f"expiry or a manual cancel) — automatically re-armed a stop @ {stop_price:.2f}.",
        source="_ensure_stop_is_resting",
    )


def _close_orphaned_log_entry(
    broker: IBBroker, ticker: str, last_price: float, log_dir: Path, initial_capital: float
) -> None:
    """Close a logged open trade the broker no longer shows a position for.

    Happens when the resting stop-loss order fills on IBKR's side without
    the bot's own signal logic ever calling sell_market — otherwise that
    trade would stay "open" in the log forever. `last_price` is an
    approximation of the actual stop fill price (the exact fill isn't
    queried here), so these show up tagged distinctly from signal exits.

    Real incident this guards against: a mid-cycle connectivity hiccup
    (IBKR logged "positions request timed out" then "Connectivity...lost")
    made position_shares() read 0 for two genuinely-still-held positions in
    the same cycle. Both got recorded as closed even though their stops
    were still resting untouched. A resting stop is strong evidence the
    broker's "flat" reading was a bad snapshot, not a real close — treat it
    as unreliable and retry next cycle rather than trusting a single read.
    """
    if _has_resting_stop(broker, ticker):
        logger.warning(
            "%s: reads as flat but a stop is still resting — treating as an unreliable "
            "read this cycle, not a real close", ticker,
        )
        return

    open_trades = trade_log.load_trades(log_dir=log_dir)
    if open_trades.empty:
        return
    still_open = (open_trades["ticker"] == ticker) & open_trades["exit_date"].isna()
    if not still_open.any():
        return
    shares = float(open_trades.loc[still_open, "shares"].iloc[-1])

    trade_log.record_trade_close(
        ticker, exit_price=last_price, exit_reason="stop_loss_or_external", log_dir=log_dir
    )
    ledger.apply_sell(log_dir, initial_capital, shares, last_price)
    logger.info(
        "%s position was already flat (stop-loss or external close) — logged exit @ ~%.2f",
        ticker, last_price,
    )
    alerts.record_alert(
        "warning",
        f"{ticker}: position closed outside the bot's own signal logic (stop-loss or external) "
        f"— logged exit @ ~{last_price:.2f}. Worth checking the account directly.",
        source="_close_orphaned_log_entry",
    )


def decide_action(in_position: bool, today_signal: int) -> str:
    """Pure decision logic: 'enter', 'exit', or 'hold'.

    Kept separate from broker calls so it's unit-testable without a live
    connection.
    """
    if not in_position and today_signal == 1:
        return "enter"
    if in_position and today_signal == 0:
        return "exit"
    return "hold"


def run_once(
    broker: IBBroker,
    ticker: str,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    circuit_breaker: DailyLossCircuitBreaker | None = None,
    log_dir: Path = trade_log.DEFAULT_LOG_DIR,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> str:
    """Check today's signal for `ticker` and act on it once via `broker`.

    Returns the action taken: 'enter', 'exit', 'hold', or 'halted'.

    `log_dir` and `initial_capital` let more than one strategy run against
    the same real IBKR paper account at once (e.g. comparing a fast swing
    strategy against the slower default) without their results getting
    mixed up: each strategy gets its own trade log directory and its own
    virtual capital, tracked via execution/ledger.py rather than
    broker.account_equity() — which returns the *whole account's* net
    liquidation and so would report identical "equity" for every strategy
    sharing that account, making a fair comparison impossible.
    """
    lookback_days = long_window * 2 + _LOOKBACK_BUFFER_DAYS
    prices = load_price_data(
        ticker,
        start=(date.today() - timedelta(days=lookback_days)).isoformat(),
        end=date.today().isoformat(),
        use_cache=False,
    )
    signals = generate_signals(prices, short_window, long_window)
    today_signal = int(signals.iloc[-1])
    shares_held = broker.position_shares(ticker)

    close = prices["Close"]
    last_price = float(close.iloc[-1])
    short_ma = float(close.rolling(window=short_window).mean().iloc[-1])
    long_ma = float(close.rolling(window=long_window).mean().iloc[-1])

    if math.isnan(last_price) or math.isnan(short_ma) or math.isnan(long_ma):
        # A connectivity drop or data-feed hiccup can hand back NaN prices —
        # NaN comparisons silently evaluate False, which would otherwise
        # look like a real "trend turned off" signal and trigger a bogus
        # exit. Refuse to act on invalid data rather than trade on it. Mark
        # the held position (if any) at the last known-good close instead of
        # today's NaN one, so a data glitch doesn't show up as a fake dip in
        # the equity curve.
        logger.warning(
            "%s: price data invalid (NaN) this check — skipping all trading logic rather than "
            "act on it. Will retry next interval.", ticker,
        )
        alerts.record_alert(
            "warning",
            f"{ticker}: price data came back invalid (NaN) this check — likely a connectivity "
            "drop. Skipped trading logic rather than act on bad data.",
            source="run_once",
        )
        valid_closes = close.dropna()
        mark_price = float(valid_closes.iloc[-1]) if not valid_closes.empty else 0.0
        equity = ledger.compute_equity(log_dir, initial_capital, shares_held, mark_price)
        trade_log.record_snapshot(ticker, equity=equity, log_dir=log_dir)
        return "data_error"

    equity = ledger.compute_equity(log_dir, initial_capital, shares_held, last_price)

    if circuit_breaker is not None and circuit_breaker.update(equity):
        logger.warning("Circuit breaker tripped (daily loss limit) — no new trades today.")
        return "halted"

    logger.info(
        "%s status: price=%.2f short_ma(%d)=%.2f long_ma(%d)=%.2f signal=%d position=%d shares equity=%.2f",
        ticker, last_price, short_window, short_ma, long_window, long_ma, today_signal, shares_held, equity,
    )

    if shares_held == 0:
        _close_orphaned_log_entry(broker, ticker, last_price, log_dir, initial_capital)

    action = decide_action(in_position=shares_held > 0, today_signal=today_signal)

    if shares_held > 0 and action != "exit":
        # Only worth checking when we're not about to cancel-and-sell anyway.
        _ensure_stop_is_resting(broker, ticker, stop_loss_pct)

    if action == "enter":
        stop_price = fixed_stop_price(last_price, stop_loss_pct)
        shares = shares_for_fixed_risk(equity, last_price, stop_price, risk_per_trade_pct)
        if shares > 0:
            try:
                broker.buy_with_stop(ticker, shares, stop_price)
            except RuntimeError:
                # buy_with_stop already cancelled the order and alerted at
                # the broker level if it couldn't confirm that cancellation
                # — but a real incident showed the confirmation itself can
                # race (an order flagged "Cancelled" silently resubmits and
                # fills moments later). Double-check before giving up: if
                # real shares exist despite the exception, reconcile now
                # instead of leaving trade_log/ledger silently wrong.
                actual_shares = broker.position_shares(ticker)
                if actual_shares > 0:
                    logger.error(
                        "%s: buy raised but %d real shares exist — reconciling instead of losing track",
                        ticker, actual_shares,
                    )
                    trade_log.record_trade_open(
                        ticker, entry_price=last_price, shares=actual_shares, log_dir=log_dir
                    )
                    ledger.apply_buy(log_dir, initial_capital, actual_shares, last_price)
                    _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
                    equity = ledger.compute_equity(log_dir, initial_capital, actual_shares, last_price)
                else:
                    action = "hold"
            else:
                trade_log.record_trade_open(ticker, entry_price=last_price, shares=shares, log_dir=log_dir)
                ledger.apply_buy(log_dir, initial_capital, shares, last_price)
                equity = ledger.compute_equity(log_dir, initial_capital, shares, last_price)
                logger.info(
                    "Entered %s: %d shares @ ~%.2f, stop %.2f", ticker, shares, last_price, stop_price
                )
        else:
            action = "hold"  # position sizing rounded to zero shares — nothing to do
    elif action == "exit":
        broker.cancel_open_orders(ticker)
        sell_trade = broker.sell_market(ticker, shares_held)
        filled_qty = int(sell_trade.orderStatus.filled)

        if filled_qty <= 0:
            # The sell didn't happen (e.g. queued outside trading hours and
            # cancelled by sell_market's own timeout, per the real incident
            # that motivated this) — we just cancelled the protective stop
            # above, so the position is now resting unprotected. Re-arm
            # immediately rather than record a close that never happened;
            # decide_action will retry the exit again next cycle.
            logger.warning(
                "%s: exit sell didn't fill — re-arming stop, will retry exit next cycle", ticker,
            )
            _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
            action = "hold"
        else:
            exit_price = sell_trade.orderStatus.avgFillPrice or last_price
            trade_log.record_trade_close(
                ticker, exit_price=exit_price, exit_reason="signal_exit", log_dir=log_dir
            )
            ledger.apply_sell(log_dir, initial_capital, filled_qty, exit_price)
            remaining = shares_held - filled_qty
            equity = ledger.compute_equity(log_dir, initial_capital, remaining, exit_price)
            if remaining > 0:
                # Partial fill: still holding some shares with no resting
                # stop (cancelled above for the whole original quantity).
                logger.warning(
                    "%s: only %d/%d sold — re-arming stop for the %d shares still held",
                    ticker, filled_qty, shares_held, remaining,
                )
                _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
            logger.info("Exited %s: sold %d shares @ ~%.2f", ticker, filled_qty, exit_price)

    trade_log.record_snapshot(ticker, equity=equity, log_dir=log_dir)
    return action
