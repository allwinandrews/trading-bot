"""Thin wrapper around Interactive Brokers' API (via ib_insync).

Handles connecting to TWS/IB Gateway and placing/querying orders for one
account. The same code path works for paper and live trading — only the
port differs (see .env.example): IB Gateway paper defaults to 4002 (live
4001), TWS paper to 7497 (live 7496). Kept separate from the decision logic
in paper_trader.py so that logic can be unit-tested against a stub broker
without a live connection.
"""
from __future__ import annotations

import asyncio
import logging
import os

# ib_insync's dependency (eventkit) grabs an asyncio event loop for the
# *current thread* the moment it's imported. That's fine in a plain script
# (main thread always has one), but Streamlit runs each page in its own
# worker thread with no event loop by default, which makes the import
# itself raise "There is no current event loop in thread ...". Ensure one
# exists before importing, so this module works from either context.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, MarketOrder, Stock, StopOrder, Trade

from execution import alerts

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002  # IB Gateway paper trading port
DEFAULT_CLIENT_ID = 1
DEFAULT_FILL_TIMEOUT = 15.0  # seconds to wait for a market order to fill


def _ensure_event_loop() -> None:
    """Make sure the *current* thread has an asyncio event loop set.

    The import-time guard above only covers the thread that happened to
    import this module first. Streamlit reruns a page's script — including
    fresh IBBroker() construction and .connect() calls — on whatever thread
    it feels like, which may be a different one each time, so this same
    check needs to run again at actual point of use, not just at import.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class IBBroker:
    """Connects to IBKR via TWS/IB Gateway and places/queries orders for one account."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int = DEFAULT_CLIENT_ID,
    ):
        _ensure_event_loop()
        self.host = host or os.getenv("IB_HOST", DEFAULT_HOST)
        self.port = port or int(os.getenv("IB_PORT", DEFAULT_PORT))
        self.client_id = client_id
        self.ib = IB()

    def connect(self) -> None:
        """Connect, cleaning up after ourselves if it fails partway through.

        Real gap this closes: ib.connect() can get far enough to register
        this clientId with IB Gateway (the TCP/API handshake succeeds) and
        then still raise — e.g. a startup request like reqPositions timing
        out. Used via `with IBBroker(...) as broker:`, a connect() that
        raises means __enter__ never returns, so __exit__/disconnect()
        never runs either — the socket leaks and Gateway keeps showing that
        client ID as connected. Catch, disconnect if a partial connection
        was made, then re-raise so the caller still sees the real failure.
        """
        _ensure_event_loop()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception:
            if self.ib.isConnected():
                self.ib.disconnect()
            raise

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self) -> "IBBroker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def account_equity(self) -> float:
        """Net liquidation value of the connected account."""
        for row in self.ib.accountSummary():
            if row.tag == "NetLiquidation":
                return float(row.value)
        raise RuntimeError("NetLiquidation not found in account summary — is the account connected?")

    def position_shares(self, symbol: str) -> int:
        """Current whole-share position in `symbol` (0 if flat)."""
        for pos in self.ib.positions():
            if pos.contract.symbol == symbol:
                return int(pos.position)
        return 0

    def live_marks(self) -> dict[str, float]:
        """Current market price per held symbol, as IBKR itself values them.

        Sourced from ib.portfolio(), which IBKR pushes on connect — this
        needs no streaming market-data subscription and is materially
        fresher than the daily closes the strategy logic runs on. Used for
        display so the dashboard agrees with what the broker shows, rather
        than quoting yesterday's close as if it were current.
        """
        return {
            item.contract.symbol: float(item.marketPrice)
            for item in self.ib.portfolio()
            if item.marketPrice and item.marketPrice > 0
        }

    def all_positions(self) -> list[dict]:
        """Every open position in the account, for display purposes."""
        return [
            {"symbol": p.contract.symbol, "shares": p.position, "avg_cost": p.avgCost}
            for p in self.ib.positions()
        ]

    def open_orders(self) -> list[dict]:
        """Every resting/working order in the account, for display purposes.

        Explicitly requests orders from *all* client IDs, not just this
        connection's own. IBKR's reqOpenOrders (what connect() calls
        automatically) only returns orders placed by the same clientId —
        so a stop placed by the autopilot (clientId 1) was invisible to the
        dashboard (clientId 2) or any one-off check using another ID, and
        looked exactly like a missing stop even when it was resting fine.
        """
        self.ib.reqAllOpenOrders()
        self.ib.sleep(1)
        return [
            {
                "symbol": t.contract.symbol,
                "action": t.order.action,
                "type": t.order.orderType,
                "quantity": t.order.totalQuantity,
                "aux_price": t.order.auxPrice,
                "status": t.orderStatus.status,
                "filled": t.orderStatus.filled,
            }
            for t in self.ib.openTrades()
        ]

    def _wait_for_fill(self, trade: Trade, fill_timeout: float) -> None:
        """Pump the event loop until `trade` is done (filled/cancelled/rejected) or times out."""
        waited = 0.0
        poll_interval = 0.5
        while not trade.isDone() and waited < fill_timeout:
            self.ib.sleep(poll_interval)
            waited += poll_interval

    def _wait_for_acknowledgement(self, trade: Trade, timeout: float = 5.0) -> None:
        """Pump the event loop until `trade` has actually reached IBKR's system.

        placeOrder() is fire-and-forget — it returns a Trade object before
        the order has necessarily been transmitted, let alone acknowledged.
        A resting order (like a stop-loss) never becomes isDone(), so this
        checks for status leaving the empty/PendingSubmit state instead.
        Without this, disconnecting right after placeOrder() (exactly what
        the connect -> act -> disconnect pattern in run_once() does) can
        sever the connection before the order was ever actually placed —
        silently leaving a position with no protective stop.
        """
        waited = 0.0
        poll_interval = 0.5
        while trade.orderStatus.status in ("", "PendingSubmit") and waited < timeout:
            self.ib.sleep(poll_interval)
            waited += poll_interval
        if trade.orderStatus.status in ("", "PendingSubmit"):
            logger.warning(
                "Order for %s still not acknowledged by IBKR after %.1fs (status=%r) — "
                "it may not actually be resting; check open orders before assuming it's protected.",
                trade.contract.symbol, timeout, trade.orderStatus.status,
            )

    def _force_cancel(self, trade: Trade, symbol: str, side: str) -> None:
        """Cancel `trade` and confirm it's actually gone — don't trust a
        single isDone()/status snapshot.

        Real incident this guards against: IBKR's paper preset can flash an
        order to "Cancelled" and then silently auto-resubmit it to
        PreSubmitted moments later (the same TIF dance documented in
        buy_with_stop above). A cancel call gated on `not trade.isDone()`
        caught the order mid-flicker at "Cancelled" and skipped cancelling
        — the order then resubmitted and sat live, untracked, for hours.
        So: always issue the cancel (a no-op if it's genuinely already
        gone), then poll briefly and raise a loud alert if it still isn't
        confirmed dead — silence here is exactly how the last one hid.
        """
        self.ib.cancelOrder(trade.order)
        waited = 0.0
        poll_interval = 0.5
        confirm_timeout = 5.0
        while trade.orderStatus.status not in ("Cancelled", "ApiCancelled") and waited < confirm_timeout:
            self.ib.sleep(poll_interval)
            waited += poll_interval

        if trade.orderStatus.filled > 0:
            logger.error(
                "%s %s order filled (%s shares) despite being cancelled after timeout — "
                "check the account directly, this position is likely untracked.",
                symbol, side, trade.orderStatus.filled,
            )
            alerts.record_alert(
                "error",
                f"{symbol}: {side} order filled ({trade.orderStatus.filled} shares) despite being "
                "cancelled after a fill timeout — this position may be untracked. Check the account.",
                source="broker._force_cancel",
            )
        elif trade.orderStatus.status not in ("Cancelled", "ApiCancelled"):
            logger.error(
                "%s %s order still not confirmed cancelled after %.1fs (status=%s) — "
                "may still be live; check the account directly.",
                symbol, side, confirm_timeout, trade.orderStatus.status,
            )
            alerts.record_alert(
                "error",
                f"{symbol}: {side} order could not be confirmed cancelled (status="
                f"{trade.orderStatus.status}) — it may still fill later untracked. Check the account.",
                source="broker._force_cancel",
            )

    def place_stop(self, symbol: str, quantity: int, stop_price: float) -> Trade:
        """Place a protective stop-loss sell that lasts until filled or cancelled.

        Explicitly GTC (good-til-cancelled) — IBKR's paper-account preset
        otherwise silently defaults stop orders to TIF=DAY, which auto-
        cancels at the end of every trading session. For a strategy that can
        hold positions for weeks, that meant the "protective" stop was
        quietly expiring overnight, every night, with nothing re-arming it —
        a real incident, not a hypothetical one. See run_once()'s
        _ensure_stop_is_resting for the second half of the fix: actively
        checking the stop is still there, not just trusting it was once.
        """
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = StopOrder("SELL", quantity, stop_price)
        order.tif = "GTC"
        stop_trade = self.ib.placeOrder(contract, order)
        self._wait_for_acknowledgement(stop_trade)
        return stop_trade

    def buy_with_stop(
        self,
        symbol: str,
        quantity: int,
        stop_price: float,
        fill_timeout: float = DEFAULT_FILL_TIMEOUT,
    ) -> tuple[Trade, Trade]:
        """Submit a market buy, wait for it to fill, then place a stop-loss
        sell sized to whatever quantity actually filled.

        Not a true atomic bracket order (the two legs aren't OCA-linked) —
        a known v1 simplification, same spirit as backtester/engine.py's.
        Waits up to `fill_timeout` seconds for the buy to complete rather
        than assuming it fills instantly: a large or illiquid order can take
        longer than a blind fixed sleep, and placing a stop for more shares
        than are actually held would misrepresent the protection in place.

        If it still hasn't filled by then, the order is cancelled before
        raising — real incident this guards against: a market order placed
        outside regular trading hours sits queued rather than rejected
        (IBKR: "will not be placed at the exchange until <next open>"). The
        old code gave up locally and raised, but left that order resting on
        IBKR's side; it filled for real, hours later, with no trade logged
        and no stop ever placed. The *next* check then saw position_shares()
        still at 0 (order hadn't filled yet) and placed a second, identical
        buy — which also queued and later filled, doubling the position.
        Cancelling here means "we gave up" and "no position exists" actually
        stay true instead of silently diverging from what IBKR does later.
        """
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        buy_trade = self.ib.placeOrder(contract, MarketOrder("BUY", quantity))
        self._wait_for_fill(buy_trade, fill_timeout)

        filled_qty = int(buy_trade.orderStatus.filled)
        if filled_qty <= 0:
            self._force_cancel(buy_trade, symbol, "buy")
            raise RuntimeError(
                f"Buy order for {symbol} had not filled after {fill_timeout}s "
                f"(status={buy_trade.orderStatus.status}) — cancelled rather than left resting"
            )
        if filled_qty < quantity:
            logger.warning(
                "%s buy only %d/%d filled after %.1fs — placing stop for %d shares actually held",
                symbol, filled_qty, quantity, fill_timeout, filled_qty,
            )

        stop_trade = self.place_stop(symbol, filled_qty, stop_price)
        return buy_trade, stop_trade

    def sell_market(self, symbol: str, quantity: int, fill_timeout: float = DEFAULT_FILL_TIMEOUT) -> Trade:
        """Submit a market sell and wait for it to fill (or time out) before returning.

        Waiting matters here specifically so `trade.orderStatus.avgFillPrice`
        is populated for the caller to log the actual exit price, not just
        the price at the moment the order was sent.

        An order still unfilled at the timeout is cancelled rather than left
        resting — same reasoning as buy_with_stop: outside regular trading
        hours an order queues instead of rejecting, so "gave up waiting"
        must also mean "this didn't happen" on IBKR's side, or a caller that
        treats the timeout as a failed exit (paper_trader.py does, since it
        must not log a close or drop the protective stop for a sale that
        never happened) would silently diverge from a fill arriving later.
        """
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        trade = self.ib.placeOrder(contract, MarketOrder("SELL", quantity))
        self._wait_for_fill(trade, fill_timeout)

        if trade.orderStatus.filled <= 0:
            logger.warning(
                "%s sell order for %d shares had not filled after %.1fs (status=%s) — cancelling",
                symbol, quantity, fill_timeout, trade.orderStatus.status,
            )
            self._force_cancel(trade, symbol, "sell")
        return trade

    def cancel_open_orders(self, symbol: str) -> None:
        """Cancel any open orders (e.g. a resting stop) for `symbol`."""
        for trade in self.ib.openTrades():
            if trade.contract.symbol == symbol:
                self.ib.cancelOrder(trade.order)
