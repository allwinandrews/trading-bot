"""Scenario Lab: a standing experiment you watch rather than operate.

A background worker (run_scenario_lab.py) keeps running the same set of
strategy settings across different baskets and market periods, accumulating
comparable studies. This page has no controls on purpose — the findings are
only meaningful if the experiment runs the same way every time, and a lab
you keep adjusting mid-flight produces results you can't compare.

Everything here is historical simulation. Nothing on this page trades.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from backtester.scenarios import leaderboard, load_status, load_studies
from execution.autopilot_control import get_status, start_autopilot

load_dotenv()

_LAB = "scenario_lab"


def _ensure_running() -> None:
    """Start the lab if it isn't going — no button, it just runs."""
    if get_status(_LAB)["running"]:
        return
    try:
        start_autopilot(
            None, 900, strategy=_LAB, script="run_scenario_lab.py", extra_args=["--capital", "500"],
        )
    except RuntimeError:
        pass  # already starting elsewhere; the status line will catch up


def _progress_header(studies: dict, status: dict) -> None:
    total_combinations = 20  # 4 baskets x 5 periods, see run_scenario_lab.py
    done = len(studies)

    cols = st.columns([2, 1, 1])
    with cols[0]:
        state = status.get("state")
        if state == "running":
            st.info(
                f"**Studying now:** {status.get('basket')} over {status.get('period')}…",
                icon=":material/science:",
            )
        elif state == "error":
            st.error(f"Last study failed: {status.get('error', 'unknown error')}")
        elif done:
            st.success(
                f"**Idle** — {done} of {total_combinations} studies complete.",
                icon=":material/check_circle:",
            )
        else:
            st.info("Starting up — the first study takes a minute or two.", icon=":material/hourglass:")

    cols[1].metric("Studies completed", f"{done}/{total_combinations}")
    cols[2].metric(
        "Settings compared", len(next(iter(studies.values()))["rows"]) if studies else 0,
        help="Strategy configurations tested in each study, including the buy & hold benchmark",
    )
    st.progress(min(done / total_combinations, 1.0))


def _verdict_across_studies(board: pd.DataFrame) -> None:
    """The honest headline: did anything consistently beat doing nothing?"""
    if board.empty:
        return

    strategies = board[board["scenario"] != "Buy & hold (benchmark)"]
    if strategies.empty:
        return

    consistent = strategies[strategies["beat_benchmark_pct"] >= 50]
    if consistent.empty:
        best = strategies.iloc[0]
        st.warning(
            f"**No setting has beaten buy & hold in most studies yet.** The strongest so far — "
            f"“{best['scenario']}” — averaged {best['avg_cagr_pct']:.1f}%/yr and beat the benchmark "
            f"in {best['beat_benchmark_pct']:.0f}% of studies. Trend-following typically trades "
            f"return for smaller drawdowns ({best['avg_drawdown_pct']:.0f}% average), which is a "
            f"real benefit — but it isn't the same thing as making more money."
        )
    else:
        best = consistent.iloc[0]
        st.success(
            f"**“{best['scenario']}” has beaten buy & hold in {best['beat_benchmark_pct']:.0f}% of "
            f"studies**, averaging {best['avg_cagr_pct']:.1f}%/yr with a "
            f"{best['avg_drawdown_pct']:.0f}% average drawdown. Worth watching as more studies "
            f"land — consistency across conditions is what separates an edge from luck."
        )


def _leaderboard_table(board: pd.DataFrame) -> None:
    if board.empty:
        return

    display = board.rename(columns={
        "scenario": "Setting", "studies": "Studies", "avg_cagr_pct": "Avg return/yr",
        "avg_drawdown_pct": "Avg drawdown", "avg_fees": "Avg fees",
        "beat_benchmark_pct": "Beat benchmark",
    })
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Avg return/yr": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg drawdown": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg fees": st.column_config.NumberColumn(format="$%.2f"),
            "Beat benchmark": st.column_config.ProgressColumn(
                "Beat benchmark", format="%.0f%%", min_value=0, max_value=100,
            ),
        },
    )


def _risk_return_chart(board: pd.DataFrame) -> go.Figure:
    strategies = board[board["scenario"] != "Buy & hold (benchmark)"]
    benchmark = board[board["scenario"] == "Buy & hold (benchmark)"]

    notable = set(strategies.nlargest(2, "avg_cagr_pct")["scenario"]) | set(
        strategies.nlargest(1, "avg_fees")["scenario"]
    )
    labels = [s if s in notable else "" for s in strategies["scenario"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategies["avg_drawdown_pct"].abs(), y=strategies["avg_cagr_pct"],
        mode="markers+text", name="Strategy settings",
        text=labels, textposition="top center", textfont=dict(size=10),
        marker=dict(
            size=14, color=strategies["beat_benchmark_pct"],
            colorscale="RdYlGn", cmin=0, cmax=100, showscale=True,
            colorbar=dict(title="Beat<br>benchmark %", thickness=12),
            line=dict(width=1, color="rgba(255,255,255,0.35)"),
        ),
        customdata=strategies[["scenario", "studies", "avg_fees"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{y:.1f}%/yr · %{x:.1f}% drawdown"
            "<br>%{customdata[1]} studies · $%{customdata[2]:.2f} avg fees<extra></extra>"
        ),
    ))
    if not benchmark.empty:
        bh = benchmark.iloc[0]
        fig.add_trace(go.Scatter(
            x=[abs(bh["avg_drawdown_pct"])], y=[bh["avg_cagr_pct"]],
            mode="markers+text", name="Buy & hold",
            text=["Buy & hold"], textposition="bottom center",
            marker=dict(size=18, symbol="star", color="#4dd0e1"),
            hovertemplate="<b>Buy & hold</b><br>%{y:.1f}%/yr · %{x:.1f}% drawdown<extra></extra>",
        ))
        fig.add_hline(
            y=bh["avg_cagr_pct"], line_dash="dot", line_color="#4dd0e1", opacity=0.5,
            annotation_text="below this line lost to doing nothing",
            annotation_position="bottom right", annotation_font_size=10,
        )

    fig.update_layout(
        xaxis_title="Average worst drawdown (%)", yaxis_title="Average return (%/yr)",
        margin=dict(l=70, r=10, t=40, b=50), height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _latest_study(studies: dict) -> None:
    if not studies:
        return

    key, study = next(iter(studies.items()))
    st.subheader("Most recent study")
    st.caption(f"{key} · ${float(study['context']['capital']):,.0f} · run {study['ran_at'][:16].replace('T', ' ')}")

    df = pd.DataFrame(study["rows"])
    df["Equity"] = df["equity_curve"]
    display = df[["scenario", "Equity", "cagr_pct", "max_drawdown_pct", "num_trades",
                  "win_rate_pct", "commission_paid"]].rename(columns={
        "scenario": "Setting", "cagr_pct": "Return/yr", "max_drawdown_pct": "Max drawdown",
        "num_trades": "Trades", "win_rate_pct": "Win %", "commission_paid": "Fees paid",
    })
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Equity": st.column_config.LineChartColumn("Equity curve", width="small"),
            "Return/yr": st.column_config.NumberColumn(format="%.1f%%"),
            "Max drawdown": st.column_config.NumberColumn(format="%.1f%%"),
            "Win %": st.column_config.NumberColumn(format="%.0f%%"),
            "Fees paid": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


@st.fragment(run_every="20s")
def _lab() -> None:
    studies = load_studies()
    status = load_status()

    _progress_header(studies, status)

    if not studies:
        st.info(
            "No completed studies yet. The lab runs one basket-and-period combination at a time "
            "and records the result here — the first should appear within a couple of minutes."
        )
        return

    board = leaderboard(studies)
    _verdict_across_studies(board)
    st.plotly_chart(_risk_return_chart(board), use_container_width=True, config={"displayModeBar": False})

    st.subheader("How each setting has done, averaged across every study")
    st.caption(
        "“Beat benchmark” is the column that matters — a setting that wins once and loses "
        "four times hasn't found an edge."
    )
    _leaderboard_table(board)
    _latest_study(studies)


st.title("Scenario Lab")
st.caption(
    "A standing experiment: the same strategy settings, run across different baskets and market "
    "conditions with the same $500, to see which trade-offs hold up. Runs by itself — nothing to "
    "configure, nothing here trades."
)

_ensure_running()
_lab()
