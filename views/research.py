"""Research page: what the bot looked at, calculated, and decided — and why.

Reads the report the bot writes each cycle (execution/screen_report.py)
rather than recomputing anything here. That matters: recomputing would be
slow (45 price fetches) and, worse, could show numbers the bot never
actually acted on. Everything here is what it really saw.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from execution.autopilot_control import get_status, trade_log_dir
from execution.live_view import countdown_text, fetch_live_marks
from execution.screen_report import load_screen_report

load_dotenv()

_STRATEGY = "portfolio"
PRICE_REFRESH_SECONDS = 20  # must match _cached_marks' cache ttl


@st.cache_data(ttl=20, show_spinner="Fetching live prices…")
def _cached_marks() -> tuple[dict[str, float] | None, datetime]:
    """Broker prices plus when they were fetched, refetched at most every
    20s — each call opens a real IBKR connection, and the page redraws
    more often than prices actually move."""
    return fetch_live_marks(), datetime.now()


def _decision_badge(ticker: str, decisions: dict) -> str:
    note = decisions.get(ticker)
    if note is None:
        return "—"
    return note


@st.fragment(run_every="15s")
def _research_panel() -> None:
    report = load_screen_report(trade_log_dir(_STRATEGY))

    if report is None:
        st.info(
            "Nothing yet — this fills in once the bot completes a full check. "
            "If it's running, give it a cycle (up to 5 minutes)."
        )
        return

    ran_at = str(report.get("timestamp", ""))[:19].replace("T", " ")
    status = get_status(_STRATEGY)
    if status["running"]:
        next_in = countdown_text(report.get("timestamp"), int(status["interval_seconds"]))
        st.caption(f"Last analysis: {ran_at} · next in **{next_in}**")
    else:
        st.caption(f"Last analysis: {ran_at} · bot is stopped, so this won't update")

    regime_ok = report.get("regime_ok")
    market = report.get("market_ticker", "SPY")
    screened = report.get("screened", True)

    cols = st.columns(4)
    if regime_ok is True:
        cols[0].metric("Market regime", "Risk-on", help=f"{market} is above its 200-day average — buying allowed")
    elif regime_ok is False:
        cols[0].metric("Market regime", "Risk-off", help=f"{market} is below its 200-day average — no new buys")
    elif not screened:
        cols[0].metric("Market regime", "Not checked", help="No free slots, so the market call wasn't needed")
    else:
        cols[0].metric("Market regime", "Unknown", help="Market data was unavailable this cycle")

    # When every slot is full the bot skips screening entirely, so reporting
    # "45 screened / 0 passed" would describe work it never did.
    cols[1].metric("Universe screened", report.get("universe_size", 0) if screened else "—")
    cols[2].metric("Passed screen", len(report.get("candidates", [])) if screened else "—")
    cols[3].metric(
        "Slots", f"{len(report.get('held', []))}/{report.get('max_positions', 0)}",
        help="Positions held vs the maximum allowed at once",
    )

    if report.get("halted"):
        st.error("Circuit breaker tripped this cycle — daily loss limit hit, no new entries taken.")
    elif regime_ok is False:
        st.warning(
            f"{market} is below its long-term average, so the bot is sitting out — "
            "it only buys individual stocks when the broad market is itself trending up."
        )

    holdings = report.get("holdings_analysis", [])
    if holdings:
        st.subheader("Current holdings — is the reason for holding still true?")
        marks, marks_at = _cached_marks()
        marks = marks or {}
        if marks:
            next_price_in = countdown_text(marks_at.isoformat(), PRICE_REFRESH_SECONDS)
            st.caption(f"Live prices as of {marks_at:%H:%M:%S} · next in {next_price_in}")
        hdf = pd.DataFrame(holdings)
        # The bot decides on daily closes, but showing those as "price" made
        # this page disagree with IBKR; show both so neither is misleading.
        hdf["live_price"] = hdf["ticker"].map(
            lambda t: round(marks[t], 2) if t in marks else None
        )
        hdf["trend_still_on"] = hdf["trend_still_on"].map({True: "Yes", False: "No", None: "Unknown"})
        hdf = hdf.rename(columns={
            "ticker": "Ticker", "shares": "Shares", "last_price": "Price it decided on",
            "live_price": "Live price", "trend_still_on": "Trend still up?", "action": "Decision",
        })
        st.dataframe(
            hdf[["Ticker", "Shares", "Price it decided on", "Live price", "Trend still up?", "Decision"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "The bot trades on daily closing prices (what the strategy was designed and tested on); "
            "the live price is IBKR's current mark, shown so this page agrees with your broker."
        )

    st.subheader("Candidates that passed the screen")
    st.caption(
        "Score = trend quality (efficiency ratio, 0-1) × momentum (90-day % change). "
        "Only stocks already trending up *and* cleared by the market-regime gate get this far, "
        "ranked best first."
    )

    candidates = report.get("candidates", [])
    decisions = report.get("decisions", {})
    if not report.get("screened", True):
        st.info(
            f"No screening this cycle — all {report.get('max_positions', 0)} position slots are "
            "full, so there's nothing to buy even if something qualified. It resumes screening "
            "as soon as a position closes."
        )
    elif not candidates:
        st.write("Nothing passed the screen this cycle — no stock was both trending and clean enough to trade.")
    else:
        df = pd.DataFrame(candidates)
        df["decision"] = df["ticker"].map(lambda t: _decision_badge(t, decisions))
        df = df.rename(columns={
            "ticker": "Ticker", "last_price": "Price", "efficiency_ratio": "Trend quality",
            "momentum_pct": "Momentum %", "score": "Score", "decision": "What the bot did",
        })
        st.dataframe(
            df[["Ticker", "Price", "Trend quality", "Momentum %", "Score", "What the bot did"]],
            use_container_width=True, hide_index=True,
        )

    excluded = report.get("excluded", {})
    if excluded:
        with st.expander(f"Why the other {len(excluded)} were excluded"):
            reasons: dict[str, list[str]] = {}
            for ticker, reason in excluded.items():
                reasons.setdefault(reason, []).append(ticker)
            for reason, tickers in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                st.markdown(f"**{reason}** ({len(tickers)}): {', '.join(sorted(tickers))}")

    entered, exited = report.get("entered", []), report.get("exited", [])
    if entered or exited:
        st.subheader("Actions taken this cycle")
        if entered:
            st.success(f"Bought: {', '.join(entered)}")
        if exited:
            st.info(f"Sold: {', '.join(exited)}")


st.title("Research & Analysis")
st.caption(
    "Everything the bot calculated on its most recent check — the market call, every stock it "
    "scored, and the reason behind each decision. This is the working, not the result."
)
_research_panel()
