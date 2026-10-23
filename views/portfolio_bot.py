"""Portfolio Bot page: what the bot is actually holding and how it's doing.

Run via `streamlit run dashboard.py` — see dashboard.py for the navigation
setup. The reasoning *behind* these trades lives on the Research page
(views/research.py); this page is deliberately just controls and outcomes,
laid out to fit one screen without scrolling.
"""
from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from backtester.metrics import summarize
from data.universe import DEFAULT_UNIVERSE
from execution.alerts import acknowledge_all, load_alerts
from execution.autopilot_control import (
    get_status,
    list_strategies,
    read_recent_log,
    start_autopilot,
    stop_autopilot,
    trade_log_dir,
)
from execution.live_view import build_live_view, countdown_text, fetch_live_marks
from execution.screen_report import load_screen_report
from execution.portfolio_trader import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
)
from execution.trade_log import load_equity_curve, load_trades

load_dotenv()

_STRATEGY = "portfolio"
PRICE_REFRESH_SECONDS = 20  # must match _cached_marks' cache ttl


@st.cache_data(ttl=20, show_spinner="Fetching live prices…")
def _cached_marks() -> tuple[dict[str, float] | None, datetime]:
    """Broker prices plus when they were actually fetched, cached ~20s.

    Each call opens a real IBKR connection, so an uncached fragment rerun
    every 15s would hammer the Gateway — and the page redraws more often
    than prices meaningfully move anyway. The timestamp rides along so the
    page can show how old the prices really are rather than implying they're
    as fresh as the redraw.
    """
    return fetch_live_marks(), datetime.now()


def _initial_capital() -> float:
    config = list_strategies().get(_STRATEGY, {}).get("config", {})
    return float(config.get("initial_capital") or DEFAULT_INITIAL_CAPITAL)


def _equity_chart(equity_curve) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode="lines", name="Equity"))
    fig.update_layout(
        xaxis_title=None, yaxis_title="Equity",
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        showlegend=False,
    )
    return fig


@st.fragment(run_every="5s")
def _running_status() -> None:
    """Its own fragment so the countdown actually counts down — the rest of
    the sidebar is static controls that don't need re-rendering."""
    status = get_status(_STRATEGY)
    if not status["running"]:
        st.info("Not running")
        return

    st.success(f"Running — PID {status['pid']}")
    report = load_screen_report(trade_log_dir(_STRATEGY))
    next_in = countdown_text((report or {}).get("timestamp"), int(status["interval_seconds"]))
    st.caption(
        f"Next check in **{next_in}** · every {status['interval_seconds']}s  \n"
        f"Started {status['started_at'][:16].replace('T', ' ')}"
    )
    if st.button("Stop", type="primary", use_container_width=True):
        stop_autopilot(_STRATEGY)
        st.rerun()


def _sidebar_controls() -> None:
    status = get_status(_STRATEGY)

    with st.sidebar:
        st.header("Portfolio Bot")
        _running_status()

        if not status["running"]:
            st.info("Not running")
            initial_capital = st.number_input(
                "Starting capital ($/£)", min_value=50.0, value=DEFAULT_INITIAL_CAPITAL, step=50.0,
                help="Virtual capital for position sizing and equity tracking.",
            )
            max_positions = st.number_input(
                "Max positions", min_value=1, max_value=10, value=DEFAULT_MAX_POSITIONS,
            )
            client_id = st.number_input("IBKR client ID", min_value=3, value=9, step=1)
            stop_loss_pct = st.number_input(
                "Stop-loss %", min_value=1.0, max_value=20.0, value=DEFAULT_STOP_LOSS_PCT * 100, step=0.5,
            ) / 100
            risk_per_trade_pct = st.number_input(
                "Risk per trade %", min_value=0.1, max_value=10.0,
                value=DEFAULT_RISK_PER_TRADE_PCT * 100, step=0.25,
            ) / 100
            interval_seconds = st.number_input("Check interval (s)", min_value=30, value=300, step=30)
            use_sentiment_filter = st.checkbox("News-sentiment veto (needs GEMINI_API_KEY)", value=False)

            if st.button("Start", type="primary", use_container_width=True):
                extra_args = [
                    "--max-positions", str(int(max_positions)),
                    "--stop-loss-pct", str(stop_loss_pct),
                    "--risk-per-trade-pct", str(risk_per_trade_pct),
                ]
                if use_sentiment_filter:
                    extra_args.append("--use-sentiment-filter")
                try:
                    start_autopilot(
                        None, int(interval_seconds), strategy=_STRATEGY,
                        client_id=int(client_id), initial_capital=float(initial_capital),
                        script="run_portfolio_trader.py", extra_args=extra_args,
                    )
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    st.rerun()

        with st.expander(f"Universe ({len(DEFAULT_UNIVERSE)} tickers)"):
            st.caption(", ".join(DEFAULT_UNIVERSE))


@st.fragment(run_every="15s")
def _main_panel() -> None:
    log_dir = trade_log_dir(_STRATEGY)

    unacked = load_alerts(unacknowledged_only=True)
    if unacked:
        errors = [a for a in unacked if a["severity"] == "error"]
        with st.container(border=True):
            render = st.error if errors else st.warning
            render(f"**{len(unacked)} unacknowledged alert(s)**", icon="🚨" if errors else "⚠️")
            for a in unacked[:5]:
                st.markdown(f"**{a['timestamp'][:16].replace('T', ' ')}** — {a['message']}")
            if st.button("Acknowledge all"):
                acknowledge_all()
                st.rerun()

    equity_curve = load_equity_curve(log_dir=log_dir)
    trades = load_trades(log_dir=log_dir)

    if equity_curve.empty:
        st.info("No history yet — results appear here once the bot has run at least one check.")
        return

    closed_trades = trades[trades["exit_date"].notna()]
    open_trades = trades[trades["exit_date"].isna()]
    metrics = summarize(equity_curve, closed_trades) if not closed_trades.empty else None

    # Prefer the broker's own current marks over the strategy's daily closes
    # for anything displayed, so this page agrees with what IBKR shows.
    marks, marks_at = _cached_marks()
    live = build_live_view(log_dir, _initial_capital(), marks)
    started_at = equity_curve.iloc[0]
    latest_equity = live["equity"] if live["equity"] is not None else equity_curve.iloc[-1]
    change = latest_equity - started_at
    change_pct = change / started_at * 100

    cols = st.columns(6)
    cols[0].metric(
        "Equity" if live["live"] else "Equity (stale)",
        f"{latest_equity:,.2f}", delta=f"{change_pct:+.2f}%",
        help=(
            f"Cash plus positions at IBKR's prices as of {marks_at:%H:%M:%S}, "
            f"refreshed every ~{PRICE_REFRESH_SECONDS}s. Change is {change:+,.2f} "
            "since tracking began."
        ) if live["live"] else "Live broker prices unavailable — showing the last completed check",
    )
    if live["live"]:
        next_price_in = countdown_text(marks_at.isoformat(), PRICE_REFRESH_SECONDS)
        cols[0].caption(f"{marks_at:%H:%M:%S} · next in {next_price_in}")
    else:
        cols[0].caption("prices unavailable")
    cols[1].metric("Holding", len(open_trades))
    cols[2].metric("Closed trades", metrics["num_trades"] if metrics else 0)
    cols[3].metric("Win rate", f"{metrics['win_rate_pct']:.0f}%" if metrics else "—")
    cols[4].metric("Avg trade", f"{metrics['avg_trade_return_pct']:+.1f}%" if metrics else "—")
    cols[5].metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%" if metrics else "—")

    chart_col, side_col = st.columns([2, 1])
    with chart_col:
        st.plotly_chart(_equity_chart(equity_curve), use_container_width=True, config={"displayModeBar": False})
    with side_col:
        with st.container(height=260, border=True):
            st.markdown("**Holdings**" + ("" if live["live"] else " *(prices unavailable)*"))
            if not live["holdings"]:
                st.caption("None — waiting for the screen to surface a candidate.")
            else:
                for h in live["holdings"]:
                    if h["price"] is None:
                        st.markdown(f"`{h['ticker']}` · {h['shares']:.0f} sh @ {h['entry_price']:.2f}")
                        continue
                    arrow = "🟢" if h["pnl"] >= 0 else "🔴"
                    st.markdown(
                        f"{arrow} `{h['ticker']}` {h['shares']:.0f} sh · "
                        f"{h['price']:,.2f} · {h['pnl']:+,.2f} ({h['pnl_pct']:+.1f}%)"
                    )
            if not closed_trades.empty:
                st.markdown("**Recent closes**")
                for _, row in closed_trades.tail(5).iloc[::-1].iterrows():
                    arrow = "🟢" if row["return_pct"] > 0 else "🔴"
                    st.markdown(f"{arrow} `{row['ticker']}` {row['return_pct']:+.1f}%")

    with st.expander("Recent log output"):
        st.code(read_recent_log(_STRATEGY, lines=30), language=None)


st.title("Portfolio Bot")
_sidebar_controls()
_main_panel()
