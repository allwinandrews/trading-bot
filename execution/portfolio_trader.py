"""Live/paper signal loop for the portfolio bot: screen a broad universe,
manage several simultaneous positions from one shared capital pool.

Same spirit and safety patterns as paper_trader.py (NaN guard, GTC-stop
re-arming, don't record a close that didn't actually fill) but for many
tickers sharing one pot instead of one ticker with its own dedicated
capital. Call `run_once` on a schedule, same as paper_trader.run_once.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.loader import load_price_data
from execution import alerts, ledger, trade_log
from execution.broker import IBBroker
from execution.paper_trader import _ensure_stop_is_resting, _has_resting_stop
from risk.position_sizing import shares_for_fixed_risk
from risk.stops import DailyLossCircuitBreaker, fixed_stop_price
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW, generate_signals
from execution.screen_report import save_screen_report
from strategies.regime import DEFAULT_REGIME_WINDOW, market_regime_signal
from strategies.screener import DEFAULT_MOMENTUM_WINDOW, rank_candidates

logger = logging.getLogger(__name__)

DEFAULT_MARKET_TICKER = "SPY"
DEFAULT_MAX_POSITIONS = 4
DEFAULT_STOP_LOSS_PCT = 0.05
# Deliberately much smaller than paper_trader's single-ticker default
# (0.01) for the same reason backtester/portfolio_engine.py uses a smaller
# figure: with stop_loss_pct=5%, risk_per_trade_pct=0.0125 works out to
# roughly 25% of available cash per position — enough headroom for
# DEFAULT_MAX_POSITIONS positions to actually coexist instead of the first
# candidate swallowing the whole (already small) pot.
DEFAULT_RISK_PER_TRADE_PCT = 0.0125
DEFAULT_INITIAL_CAPITAL = 500.0

_LOOKBACK_BUFFER_DAYS = 30


def _load_prices_safe(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    try:
        return load_price_data(
            ticker,
            start=(date.today() - timedelta(days=lookback_days)).isoformat(),
            end=date.today().isoformat(),
            use_cache=False,
        )
    except ValueError as e:
        logger.warning("%s: price fetch failed — excluding from this cycle (%s)", ticker, e)
        return None


def _positions_to_mark(
    broker: IBBroker, tickers: list[str], mark_price: dict[str, float], log_dir: Path
) -> dict[str, tuple[float, float]]:
    """Map each still-held ticker to (shares, price) for the equity calculation.

    A holding whose price couldn't be fetched this cycle must NOT silently
    drop out of the sum — that's a real bug this fixes: three held positions
    vanished from one cycle's equity because their prices were briefly
    unavailable, recording a 48% "crash" in the equity curve that never
    happened. Falls back to the price the position was opened at, which is
    stale but honest, rather than implicitly valuing it at zero.
    """
    open_trades = trade_log.load_trades(log_dir=log_dir)
    entry_prices: dict[str, float] = {}
    if not open_trades.empty:
        still_open = open_trades[open_trades["exit_date"].isna()]
        entry_prices = dict(zip(still_open["ticker"], still_open["entry_price"]))

    positions: dict[str, tuple[float, float]] = {}
    for ticker in set(tickers):
        shares = broker.position_shares(ticker)
        if shares <= 0:
            continue
        price = mark_price.get(ticker) or entry_prices.get(ticker)
        if price:
            positions[ticker] = (shares, float(price))
        else:
            logger.warning(
                "%s: held but no price available to mark it — excluded from this equity reading",
                ticker,
            )
    return positions


def _open_tickers(log_dir: Path) -> dict[str, float]:
    """Tickers this portfolio bot itself currently has an open trade in,
    mapped to the shares the log says it holds — not the whole real account
    (which may hold other tickers belonging to other strategies sharing the
    same IBKR paper account). The share count is the source of truth for
    crediting the ledger if a position turns out to have closed: by the
    time that's discovered, the live broker read is already 0/unreliable,
    so it can't be used for "how many shares to credit back"."""
    trades = trade_log.load_trades(log_dir=log_dir)
    if trades.empty:
        return {}
    open_rows = trades[trades["exit_date"].isna()]
    return dict(zip(open_rows["ticker"], open_rows["shares"]))


def run_once(
    broker: IBBroker,
    universe: list[str],
    max_positions: int = DEFAULT_MAX_POSITIONS,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    regime_window: int = DEFAULT_REGIME_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    market_ticker: str = DEFAULT_MARKET_TICKER,
    sentiment_filter=None,
    circuit_breaker: DailyLossCircuitBreaker | None = None,
    log_dir: Path = trade_log.DEFAULT_LOG_DIR,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> dict:
    """Check exits for currently-held tickers, then fill open slots from the
    screened universe. Returns {"exited": [...], "entered": [...], "held": [...], "skipped": [...]}.

    `sentiment_filter`, if given, is a callable(ticker) -> {"verdict": ...}
    (see ai/sentiment_filter.py) — a "negative" verdict skips that candidate
    without touching the ranking; never required (defaults to no filtering
    at all, matching the fail-open stance sentiment_filter.score_sentiment
    itself takes when unavailable).
    """
    lookback_days = long_window * 2 + _LOOKBACK_BUFFER_DAYS
    held_tickers = _open_tickers(log_dir)

    result = {"exited": [], "entered": [], "held": [], "skipped": []}
    mark_price: dict[str, float] = {}  # ticker -> price used to mark it, for the final equity snapshot
    holdings_analysis: list[dict] = []  # per-holding reasoning, for the Research page

    # --- 1. Manage existing positions: exit on signal-off, else re-arm stop. ---
    for ticker in held_tickers:
        prices = _load_prices_safe(ticker, lookback_days)
        shares_held = broker.position_shares(ticker)

        if shares_held <= 0:
            # A stop may have filled (or an external close happened) — but
            # don't trust a single "reads as flat" snapshot on its own. Real
            # incident: a mid-cycle connectivity hiccup ("positions request
            # timed out" then "Connectivity...lost") made position_shares()
            # read 0 for two genuinely-still-held positions in this exact
            # branch, in the same cycle — both got wrongly recorded as
            # closed while their stops sat untouched. A stop still resting
            # is strong evidence the broker read was bad, not that the
            # position is gone — hold and retry rather than close.
            if _has_resting_stop(broker, ticker):
                logger.warning(
                    "%s: reads as flat but a stop is still resting — treating as an "
                    "unreliable read this cycle, not a real close", ticker,
                )
                result["held"].append(ticker)
                if prices is not None and not prices["Close"].isna().any():
                    mark_price[ticker] = float(prices["Close"].iloc[-1])
                continue

            if prices is not None and not prices["Close"].isna().any():
                last_price = float(prices["Close"].iloc[-1])
                trade_log.record_trade_close(
                    ticker, exit_price=last_price, exit_reason="stop_loss_or_external", log_dir=log_dir
                )
                ledger.apply_sell(log_dir, initial_capital, held_tickers[ticker], last_price)
                alerts.record_alert(
                    "warning",
                    f"{ticker}: position closed outside the bot's own signal logic "
                    f"(stop-loss or external) — logged exit @ ~{last_price:.2f}.",
                    source="portfolio_trader",
                )
                result["exited"].append(ticker)
            continue

        if prices is None or prices["Close"].isna().any():
            logger.warning("%s: NaN/missing price data — holding without a signal check this cycle", ticker)
            result["held"].append(ticker)
            holdings_analysis.append({
                "ticker": ticker, "shares": shares_held, "last_price": None,
                "trend_still_on": None, "action": "held — no usable price data to check the signal",
            })
            continue

        last_price = float(prices["Close"].iloc[-1])
        signal = int(generate_signals(prices, short_window, long_window).iloc[-1])
        holdings_analysis.append({
            "ticker": ticker, "shares": shares_held, "last_price": last_price,
            "trend_still_on": bool(signal),
            "action": "trend intact — holding" if signal else "trend broke — selling",
        })

        if signal == 0:
            broker.cancel_open_orders(ticker)
            sell_trade = broker.sell_market(ticker, shares_held)
            filled_qty = int(sell_trade.orderStatus.filled)
            if filled_qty <= 0:
                logger.warning("%s: exit sell didn't fill — re-arming stop, will retry next cycle", ticker)
                _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
                result["held"].append(ticker)
                mark_price[ticker] = last_price
            else:
                exit_price = sell_trade.orderStatus.avgFillPrice or last_price
                trade_log.record_trade_close(
                    ticker, exit_price=exit_price, exit_reason="signal_exit", log_dir=log_dir
                )
                ledger.apply_sell(log_dir, initial_capital, filled_qty, exit_price)
                result["exited"].append(ticker)
                remaining = shares_held - filled_qty
                if remaining > 0:
                    _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
                    result["held"].append(ticker)
                    mark_price[ticker] = last_price
        else:
            _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
            result["held"].append(ticker)
            mark_price[ticker] = last_price

    # --- 2. Compute portfolio equity so far (cash + mark-to-market of what's held). ---
    equity = ledger.compute_equity_multi(
        log_dir, initial_capital,
        _positions_to_mark(broker, result["held"], mark_price, log_dir),
    )

    # `report` records the reasoning behind this cycle, not just its outcome,
    # so the Research page can show the actual numbers the bot acted on —
    # see execution/screen_report.py. Built before the circuit-breaker check
    # so a halted cycle still explains itself rather than leaving the page
    # showing the previous cycle's stale reasoning.
    report: dict = {
        "equity": equity,
        "held": list(result["held"]),
        "max_positions": max_positions,
        "universe_size": len(universe),
        "regime_ok": None,
        "market_ticker": market_ticker,
        "halted": False,
        "holdings_analysis": holdings_analysis,
        "screened": True,
        "candidates": [],
        "excluded": {},
        "decisions": {},
    }

    if circuit_breaker is not None and circuit_breaker.update(equity):
        logger.warning("Portfolio circuit breaker tripped (daily loss limit) — no new entries today.")
        trade_log.record_snapshot("PORTFOLIO", equity=equity, log_dir=log_dir)
        report["halted"] = True
        save_screen_report(log_dir, report)
        return result

    # --- 3. Fill open slots from the screened universe. ---
    open_slots = max_positions - len(result["held"])
    if open_slots <= 0:
        # Screening 45 tickers costs 45 price fetches; there's nothing to do
        # with the answer when every slot is taken, so it's skipped — the
        # Research page says so rather than showing a mysteriously empty screen.
        report["screened"] = False
    if open_slots > 0:
        market_prices = _load_prices_safe(market_ticker, lookback_days)
        if market_prices is None:
            logger.warning("Market regime data (%s) unavailable — skipping new entries this cycle", market_ticker)
            report["regime_ok"] = None
            report["excluded"]["_market_data_unavailable"] = market_ticker
        else:
            report["regime_ok"] = bool(
                market_regime_signal(market_prices, window=regime_window).iloc[-1]
            )
            excluded = set(result["held"]) | set(held_tickers) | set(result["exited"])
            candidates_prices = {}
            for ticker in universe:
                if ticker in excluded:
                    report["excluded"][ticker] = "already held by this bot"
                    continue
                if broker.position_shares(ticker) > 0:
                    # Real position already exists for this ticker under a
                    # different strategy sharing this account — never touch
                    # it, to avoid corrupting another instance's tracking.
                    report["excluded"][ticker] = "held outside this bot"
                    continue
                prices = _load_prices_safe(ticker, lookback_days)
                if prices is None or prices["Close"].isna().any():
                    report["excluded"][ticker] = "no usable price data"
                    continue
                candidates_prices[ticker] = prices

            ranked = rank_candidates(
                candidates_prices, market_prices,
                short_window=short_window, long_window=long_window, momentum_window=momentum_window,
                regime_window=regime_window,
            )
            report["candidates"] = ranked.to_dict("records")
            screened_out = set(candidates_prices) - set(ranked["ticker"]) if not ranked.empty else set(candidates_prices)
            for ticker in screened_out:
                report["excluded"][ticker] = "failed the trend/momentum screen"

            for _, row in ranked.iterrows():
                ticker = row["ticker"]
                if open_slots <= 0:
                    report["decisions"][ticker] = "ranked, but no free position slot"
                    continue

                if sentiment_filter is not None:
                    sentiment = sentiment_filter(ticker)
                    if sentiment.get("verdict") == "negative":
                        logger.info("%s: skipped — negative sentiment veto", ticker)
                        result["skipped"].append(ticker)
                        report["decisions"][ticker] = (
                            f"vetoed on news sentiment — {sentiment.get('reason', '')}".strip()
                        )
                        continue

                last_price = row["last_price"]
                cash = ledger.get_cash(log_dir, initial_capital)
                stop_price = fixed_stop_price(last_price, stop_loss_pct)
                # risk_per_trade_pct is calibrated assuming all max_positions
                # slots are still competing for the same cash pool. Once some
                # slots are already filled, that fixed fraction leaves the
                # rest of the cash stranded — so scale it up in proportion to
                # how few slots remain, letting the last open slot use up to
                # (almost) all of what's left instead of the same small share
                # it would have gotten on a fresh, all-slots-open cycle.
                effective_risk_pct = min(risk_per_trade_pct * max_positions / open_slots, 1.0)
                shares = shares_for_fixed_risk(cash, last_price, stop_price, effective_risk_pct)
                if shares <= 0:
                    result["skipped"].append(ticker)
                    report["decisions"][ticker] = (
                        f"too expensive to size — £{cash:,.2f} cash can't buy a share at {last_price:,.2f}"
                    )
                    continue

                try:
                    broker.buy_with_stop(ticker, shares, stop_price)
                except RuntimeError:
                    # Same reconciliation as paper_trader.py's enter branch:
                    # buy_with_stop already cancelled + alerted at the
                    # broker level, but a real incident showed the
                    # cancel-confirmation itself can race with a late fill.
                    actual_shares = broker.position_shares(ticker)
                    if actual_shares <= 0:
                        result["skipped"].append(ticker)
                        report["decisions"][ticker] = "order didn't fill (cancelled) — will retry next cycle"
                        continue
                    logger.error(
                        "%s: buy raised but %d real shares exist — reconciling instead of losing track",
                        ticker, actual_shares,
                    )
                    trade_log.record_trade_open(
                        ticker, entry_price=last_price, shares=actual_shares, log_dir=log_dir
                    )
                    ledger.apply_buy(log_dir, initial_capital, actual_shares, last_price)
                    _ensure_stop_is_resting(broker, ticker, stop_loss_pct)
                    result["entered"].append(ticker)
                    mark_price[ticker] = last_price
                    open_slots -= 1
                    report["decisions"][ticker] = f"bought {actual_shares} shares (reconciled after a late fill)"
                    continue

                trade_log.record_trade_open(ticker, entry_price=last_price, shares=shares, log_dir=log_dir)
                ledger.apply_buy(log_dir, initial_capital, shares, last_price)
                result["entered"].append(ticker)
                mark_price[ticker] = last_price
                open_slots -= 1
                report["decisions"][ticker] = f"bought {shares} shares @ ~{last_price:,.2f}, stop {stop_price:,.2f}"
                logger.info(
                    "Entered %s: %d shares @ ~%.2f, stop %.2f (score %.3f)",
                    ticker, shares, last_price, stop_price, row["score"],
                )

    equity = ledger.compute_equity_multi(
        log_dir, initial_capital,
        _positions_to_mark(broker, result["held"] + result["entered"], mark_price, log_dir),
    )
    trade_log.record_snapshot("PORTFOLIO", equity=equity, log_dir=log_dir)

    report["equity"] = equity
    report["entered"] = list(result["entered"])
    report["exited"] = list(result["exited"])
    save_screen_report(log_dir, report)
    return result
