"""Streamlit dashboard: run a backtest interactively and see the results.

Usage:
    streamlit run dashboard.py

Calls the same backtester.runner.execute_backtest used by run_backtest.py —
no logic is duplicated here, just presentation.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ai.trade_journal import generate_trade_journal
from backtester.engine import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_LOSS_PCT,
)
from backtester.runner import execute_backtest
from strategies.moving_average import DEFAULT_LONG_WINDOW, DEFAULT_SHORT_WINDOW

load_dotenv()

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")


@st.cache_data(show_spinner="Running backtest...")
def _cached_run(ticker, start, end, short_window, long_window,
                 initial_capital, stop_loss_pct, risk_per_trade_pct, commission_pct):
    # Streamlit reruns the whole script on every widget interaction; caching on
    # the exact input combo avoids re-downloading data / re-simulating for
    # unrelated UI changes (e.g. toggling the AI journal checkbox).
    return execute_backtest(
        ticker=ticker, start=str(start), end=str(end),
        short_window=short_window, long_window=long_window,
        initial_capital=initial_capital, stop_loss_pct=stop_loss_pct,
        risk_per_trade_pct=risk_per_trade_pct, commission_pct=commission_pct,
    )


def _equity_curve_chart(run) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run.strategy_result.equity_curve.index, y=run.strategy_result.equity_curve,
        name="Strategy", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=run.benchmark_equity.index, y=run.benchmark_equity,
        name="Buy & Hold", mode="lines",
    ))
    fig.update_layout(
        title="Equity curve: strategy vs. buy & hold",
        xaxis_title="Date", yaxis_title="Equity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _metrics_table(run) -> pd.DataFrame:
    return pd.DataFrame({
        "strategy": run.strategy_metrics,
        "buy_and_hold": run.benchmark_metrics,
    })


def main() -> None:
    st.title("Trading Bot Dashboard")
    st.caption(
        "Backtest a moving-average crossover strategy with stop-loss and "
        "position sizing, and compare it against buy-and-hold."
    )

    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Ticker", value="AAPL")
        col1, col2 = st.columns(2)
        start = col1.date_input("Start", value=pd.Timestamp("2018-01-01"))
        end = col2.date_input("End", value=pd.Timestamp("2024-01-01"))

        st.subheader("Strategy")
        short_window = st.slider("Short MA window", 2, 100, DEFAULT_SHORT_WINDOW)
        long_window = st.slider("Long MA window", 5, 250, DEFAULT_LONG_WINDOW)

        st.subheader("Risk")
        stop_loss_pct = st.slider("Stop-loss %", 0.5, 15.0, DEFAULT_STOP_LOSS_PCT * 100) / 100
        risk_per_trade_pct = st.slider("Risk per trade %", 0.1, 10.0, DEFAULT_RISK_PER_TRADE_PCT * 100) / 100

        st.subheader("Account")
        initial_capital = st.number_input("Initial capital", value=DEFAULT_INITIAL_CAPITAL, step=10_000.0)
        commission_pct = st.number_input(
            "Commission %", value=DEFAULT_COMMISSION_PCT * 100, step=0.01, format="%.3f"
        ) / 100

        want_ai_journal = st.checkbox("Generate AI trade journal (uses ANTHROPIC_API_KEY)")
        run_clicked = st.button("Run backtest", type="primary")

    if not run_clicked:
        st.info("Set your parameters in the sidebar and click **Run backtest**.")
        return

    if short_window >= long_window:
        st.error("Short MA window must be smaller than the long MA window.")
        return

    try:
        run = _cached_run(
            ticker, start, end, short_window, long_window,
            initial_capital, stop_loss_pct, risk_per_trade_pct, commission_pct,
        )
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader(run.strategy_desc)
    st.plotly_chart(_equity_curve_chart(run), use_container_width=True)

    st.subheader("Metrics: strategy vs. buy & hold")
    st.dataframe(_metrics_table(run), use_container_width=True)

    st.subheader("Trade log")
    if run.strategy_result.trades.empty:
        st.write("No trades were taken in this window.")
    else:
        st.dataframe(run.strategy_result.trades, use_container_width=True)
        st.download_button(
            "Download trades as CSV",
            run.strategy_result.trades.to_csv(index=False),
            file_name=f"{ticker}_trades.csv",
        )

    if want_ai_journal:
        st.subheader("AI trade journal")
        try:
            journal = generate_trade_journal(
                ticker, run.strategy_desc, run.strategy_metrics, run.strategy_result.trades
            )
        except RuntimeError as e:
            st.warning(str(e))
        else:
            st.markdown(journal)


if __name__ == "__main__":
    main()
