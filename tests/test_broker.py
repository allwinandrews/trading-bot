from unittest.mock import MagicMock, patch

from execution.broker import IBBroker


def _broker_with_mock_ib() -> IBBroker:
    broker = IBBroker()
    broker.ib = MagicMock()
    # isDone()/orderStatus.status defaults so _wait_for_fill/_wait_for_acknowledgement
    # return immediately instead of polling a mock forever.
    broker.ib.placeOrder.return_value = MagicMock()
    broker.ib.placeOrder.return_value.orderStatus.status = "Submitted"
    return broker


def test_connect_disconnects_a_partial_connection_before_raising():
    # Real gap found live: ib.connect() can register the clientId with IB
    # Gateway (TCP/API handshake succeeds) and then still raise — e.g. a
    # startup request timing out. Used via `with IBBroker(...) as broker:`,
    # that meant __enter__ never returned, so disconnect() never ran either
    # — the socket leaked and Gateway kept showing that client as connected.
    broker = IBBroker()
    broker.ib = MagicMock()
    broker.ib.isConnected.return_value = True  # the partial connection that leaked
    broker.ib.connect.side_effect = TimeoutError("startup request timed out")

    try:
        broker.connect()
        assert False, "expected TimeoutError"
    except TimeoutError:
        pass

    broker.ib.disconnect.assert_called_once()


def test_connect_does_not_disconnect_when_never_actually_connected():
    broker = IBBroker()
    broker.ib = MagicMock()
    broker.ib.isConnected.return_value = False  # connect() failed before any handshake
    broker.ib.connect.side_effect = ConnectionRefusedError("Gateway not running")

    try:
        broker.connect()
        assert False, "expected ConnectionRefusedError"
    except ConnectionRefusedError:
        pass

    broker.ib.disconnect.assert_not_called()


def test_place_stop_sets_gtc_time_in_force():
    # Reproduces the real incident: IBKR's paper-account preset silently
    # forces stop orders to TIF=DAY, which auto-cancels at end of session —
    # a position that survived past one trading day ended up completely
    # unprotected with no error or alert. Stops must request GTC explicitly.
    broker = _broker_with_mock_ib()

    broker.place_stop("AAPL", 1500, 323.03)

    placed_order = broker.ib.placeOrder.call_args[0][1]
    assert placed_order.action == "SELL"
    assert placed_order.totalQuantity == 1500
    assert placed_order.auxPrice == 323.03
    assert placed_order.tif == "GTC"


def test_buy_with_stop_places_gtc_stop_for_filled_quantity():
    broker = _broker_with_mock_ib()
    buy_trade = MagicMock()
    buy_trade.isDone.return_value = True
    buy_trade.orderStatus.filled = 1500

    stop_trade = MagicMock()
    stop_trade.orderStatus.status = "Submitted"

    broker.ib.placeOrder.side_effect = [buy_trade, stop_trade]

    result_buy, result_stop = broker.buy_with_stop("AAPL", 1500, 323.03)

    assert result_buy is buy_trade
    assert result_stop is stop_trade
    stop_order = broker.ib.placeOrder.call_args_list[1][0][1]
    assert stop_order.tif == "GTC"


@patch("execution.alerts.record_alert")
def test_buy_with_stop_cancels_order_that_never_fills(mock_record_alert):
    # Reproduces the real incident: a market order placed outside trading
    # hours queues instead of rejecting ("will not be placed until <next
    # open>"). The old code gave up locally and raised, but left that order
    # resting — it filled for real hours later with no trade logged and no
    # stop ever placed, and the next check (seeing 0 shares) placed a
    # second, identical buy that also filled, doubling the position.
    broker = _broker_with_mock_ib()
    buy_trade = MagicMock()
    buy_trade.isDone.return_value = False
    buy_trade.orderStatus.filled = 0
    buy_trade.orderStatus.status = "Cancelled"  # confirms cleanly once cancelled
    broker.ib.placeOrder.return_value = buy_trade

    try:
        broker.buy_with_stop("RIOT", 12, 20.70)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    broker.ib.cancelOrder.assert_called_once_with(buy_trade.order)
    mock_record_alert.assert_not_called()  # cancel confirmed cleanly, nothing to escalate


@patch("execution.alerts.record_alert")
def test_buy_with_stop_always_cancels_even_when_status_looks_already_done(mock_record_alert):
    # Reproduces the actual gap found live: IBKR flashed an order to
    # "Cancelled" (isDone() == True) and then silently auto-resubmitted it
    # to PreSubmitted moments later — a cancel gated on `not isDone()`
    # skipped cancelling entirely, leaving that resubmitted order live and
    # untracked. The cancel must always be attempted regardless of the
    # snapshot status.
    broker = _broker_with_mock_ib()
    buy_trade = MagicMock()
    buy_trade.isDone.return_value = True  # looks terminal, but isn't really
    buy_trade.orderStatus.filled = 0
    buy_trade.orderStatus.status = "Cancelled"
    broker.ib.placeOrder.return_value = buy_trade

    try:
        broker.buy_with_stop("KO", 1, 60.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    broker.ib.cancelOrder.assert_called_once_with(buy_trade.order)


@patch("execution.alerts.record_alert")
def test_buy_with_stop_alerts_when_cancel_cannot_be_confirmed(mock_record_alert):
    broker = _broker_with_mock_ib()
    buy_trade = MagicMock()
    buy_trade.isDone.return_value = False
    buy_trade.orderStatus.filled = 0
    buy_trade.orderStatus.status = "PreSubmitted"  # never actually reaches Cancelled
    broker.ib.placeOrder.return_value = buy_trade

    try:
        broker.buy_with_stop("KO", 1, 60.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    mock_record_alert.assert_called_once()
    assert mock_record_alert.call_args[0][0] == "error"


@patch("execution.alerts.record_alert")
def test_sell_market_cancels_order_that_never_fills(mock_record_alert):
    broker = _broker_with_mock_ib()
    sell_trade = MagicMock()
    sell_trade.isDone.return_value = False
    sell_trade.orderStatus.filled = 0
    sell_trade.orderStatus.status = "Cancelled"
    broker.ib.placeOrder.return_value = sell_trade

    result = broker.sell_market("RIOT", 24)

    assert result is sell_trade
    broker.ib.cancelOrder.assert_called_once_with(sell_trade.order)
    mock_record_alert.assert_not_called()


@patch("execution.alerts.record_alert")
def test_force_cancel_alerts_if_order_fills_despite_being_cancelled(mock_record_alert):
    # The exact race that hid the real incident: the order still shows 0
    # filled when the caller gives up waiting and decides to cancel it, but
    # it actually fills for real moments later, during the confirmation
    # poll — simulated here via ib.sleep's side effect. Tested directly
    # against _force_cancel (not through sell_market/buy_with_stop) so it
    # isn't coupled to _wait_for_fill's own iteration count.
    broker = _broker_with_mock_ib()
    trade = MagicMock()
    trade.orderStatus.filled = 0
    trade.orderStatus.status = "PreSubmitted"

    def fill_it_anyway(*_args, **_kwargs):
        trade.orderStatus.filled = 24
        trade.orderStatus.status = "Filled"

    broker.ib.sleep.side_effect = fill_it_anyway

    broker._force_cancel(trade, "RIOT", "sell")

    broker.ib.cancelOrder.assert_called_once_with(trade.order)
    mock_record_alert.assert_called_once()
    assert mock_record_alert.call_args[0][0] == "error"


def test_sell_market_does_not_cancel_when_filled():
    broker = _broker_with_mock_ib()
    sell_trade = MagicMock()
    sell_trade.isDone.return_value = True
    sell_trade.orderStatus.filled = 24
    broker.ib.placeOrder.return_value = sell_trade

    broker.sell_market("RIOT", 24)

    broker.ib.cancelOrder.assert_not_called()
