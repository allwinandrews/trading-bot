"""Run the same capital through many strategy configurations and compare.

The point isn't to find "the best settings" — with a handful of backtests
over one period that's just overfitting with extra steps. The point is to
see the *shape* of the trade-offs at a specific account size: how much
faster trading costs you in commission, what a tighter stop does to the
number of round trips, whether holding 1 position or 5 changes anything
once fees are real, and — the question that matters most — whether any of
it beats simply buying and holding the same basket.

Every scenario runs on identical data with identical capital, so
differences are attributable to the configuration rather than luck of the
draw on tickers or dates.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtester.metrics import summarize
from backtester.portfolio_engine import run_portfolio_backtest, run_portfolio_buy_and_hold
from strategies.moving_average import generate_signals
from strategies.regime import apply_regime_filter, market_regime_signal

RESULTS_FILE = Path(__file__).parent.parent / "execution" / "live_data" / "scenario_results.json"


SPARKLINE_POINTS = 120


def _sparkline(curve: pd.Series, points: int = SPARKLINE_POINTS) -> list[float]:
    """Thin an equity curve down to something a table cell can draw.

    A decade of daily bars is ~2,500 points per scenario; at a dozen
    scenarios that's a payload nobody can see the benefit of in a 100px
    cell, and it makes the saved results file needlessly large.
    """
    if len(curve) <= points:
        return [float(v) for v in curve]
    # ceil, not floor: floor division overshoots the cap (1000//120 = 8
    # leaves 125 points, not 120).
    step = math.ceil(len(curve) / points)
    return [float(v) for v in curve.iloc[::step]]


@dataclass(frozen=True)
class Scenario:
    name: str
    short_window: int
    long_window: int
    max_positions: int
    stop_loss_pct: float
    regime_filter: bool = True
    risk_per_trade_pct: float = 0.0025


def default_scenarios() -> list[Scenario]:
    """A deliberately small, readable sweep over the levers that actually matter.

    Kept short on purpose: every extra scenario is another chance to fool
    yourself by picking the winner after the fact. These vary one thing at a
    time from a common baseline rather than exploring every combination.
    """
    baseline = Scenario(
        name="Baseline: slow trend, 3 positions",
        short_window=50, long_window=200, max_positions=3, stop_loss_pct=0.05,
    )
    return [
        baseline,
        replace(baseline, name="Concentrated: 1 position", max_positions=1),
        replace(baseline, name="Concentrated: 2 positions", max_positions=2),
        replace(baseline, name="Spread: 5 positions", max_positions=5),
        replace(baseline, name="Faster trend (20/100)", short_window=20, long_window=100),
        replace(baseline, name="Swing speed (10/30)", short_window=10, long_window=30),
        replace(baseline, name="Tight stop (2%)", stop_loss_pct=0.02),
        replace(baseline, name="Wide stop (10%)", stop_loss_pct=0.10),
        replace(baseline, name="No market-regime filter", regime_filter=False),
        replace(baseline, name="Bigger bets (2x risk/trade)", risk_per_trade_pct=0.005),
    ]


def _signals_for(
    prices_by_ticker: dict[str, pd.DataFrame], market_prices: pd.DataFrame, scenario: Scenario
) -> dict[str, pd.Series]:
    regime = market_regime_signal(market_prices)
    signals = {}
    for ticker, prices in prices_by_ticker.items():
        raw = generate_signals(prices, scenario.short_window, scenario.long_window)
        signals[ticker] = apply_regime_filter(raw, regime) if scenario.regime_filter else raw
    return signals


def run_scenario(
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    scenario: Scenario,
    initial_capital: float,
) -> dict:
    """One scenario's headline numbers, including what it paid in fees."""
    signals = _signals_for(prices_by_ticker, market_prices, scenario)
    result = run_portfolio_backtest(
        prices_by_ticker, signals,
        initial_capital=initial_capital,
        stop_loss_pct=scenario.stop_loss_pct,
        risk_per_trade_pct=scenario.risk_per_trade_pct,
        max_positions=scenario.max_positions,
    )
    metrics = summarize(result.equity_curve, result.trades)
    final_equity = float(result.equity_curve.iloc[-1])

    return {
        "scenario": scenario.name,
        "positions": scenario.max_positions,
        "windows": f"{scenario.short_window}/{scenario.long_window}",
        "stop_loss_pct": scenario.stop_loss_pct * 100,
        "regime_filter": scenario.regime_filter,
        "final_equity": final_equity,
        "total_return_pct": metrics["total_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "num_trades": metrics["num_trades"],
        "win_rate_pct": metrics["win_rate_pct"],
        "commission_paid": round(result.total_commission, 2),
        # What the fees cost you relative to what you started with — the
        # number that makes small-account drag concrete.
        "commission_pct_of_capital": round(result.total_commission / initial_capital * 100, 2),
        "equity_curve": _sparkline(result.equity_curve),
    }


def buy_and_hold_row(
    prices_by_ticker: dict[str, pd.DataFrame], initial_capital: float
) -> dict:
    """The benchmark every scenario has to beat to be worth the effort."""
    curve = run_portfolio_buy_and_hold(prices_by_ticker, initial_capital=initial_capital)
    metrics = summarize(curve, pd.DataFrame(columns=["return_pct"]))
    return {
        "scenario": "Buy & hold (benchmark)",
        "positions": len(prices_by_ticker),
        "windows": "—",
        "stop_loss_pct": 0.0,
        "regime_filter": False,
        "final_equity": float(curve.iloc[-1]),
        "total_return_pct": metrics["total_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "num_trades": 0,
        "win_rate_pct": 0.0,
        "commission_paid": 0.0,
        "commission_pct_of_capital": 0.0,
        "equity_curve": _sparkline(curve),
    }


def run_scenarios(
    prices_by_ticker: dict[str, pd.DataFrame],
    market_prices: pd.DataFrame,
    scenarios: list[Scenario],
    initial_capital: float,
    include_benchmark: bool = True,
    progress=None,
) -> pd.DataFrame:
    """Every scenario over identical data, best CAGR first.

    `progress` is an optional callable(done, total, name) so a UI can report
    where it's up to — a full sweep is slow enough that silence looks broken.
    """
    rows = []
    total = len(scenarios) + (1 if include_benchmark else 0)

    for i, scenario in enumerate(scenarios, start=1):
        if progress:
            progress(i - 1, total, scenario.name)
        rows.append(run_scenario(prices_by_ticker, market_prices, scenario, initial_capital))

    if include_benchmark:
        if progress:
            progress(len(scenarios), total, "Buy & hold")
        rows.append(buy_and_hold_row(prices_by_ticker, initial_capital))

    if progress:
        progress(total, total, "done")

    return pd.DataFrame(rows).sort_values("cagr_pct", ascending=False).reset_index(drop=True)


def study_key(basket: str, period: str) -> str:
    return f"{basket} | {period}"


def _read_store(path: Path) -> dict:
    if not path.exists():
        return {"studies": {}, "status": {}}
    try:
        payload = json.loads(path.read_text())
        payload.setdefault("studies", {})
        payload.setdefault("status", {})
        return payload
    except (json.JSONDecodeError, OSError):
        return {"studies": {}, "status": {}}


def _write_store(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, default=str))


def save_study(results: pd.DataFrame, context: dict, path: Path = RESULTS_FILE) -> None:
    """Record one completed sweep, keyed by basket+period.

    Studies accumulate rather than overwrite: the useful question isn't
    "what did the last run say" but "does the same setting keep winning
    across different baskets and market conditions, or did it just get
    lucky once".
    """
    store = _read_store(path)
    store["studies"][study_key(context["basket"], context["period"])] = {
        "ran_at": datetime.now().isoformat(),
        "context": context,
        "rows": results.to_dict("records"),
    }
    _write_store(store, path)


def load_studies(path: Path = RESULTS_FILE) -> dict[str, dict]:
    """Every sweep completed so far, newest-run first."""
    studies = _read_store(path)["studies"]
    return dict(sorted(studies.items(), key=lambda kv: kv[1].get("ran_at", ""), reverse=True))


def save_status(status: dict, path: Path = RESULTS_FILE) -> None:
    """What the lab is working on right now, so the page can say so."""
    store = _read_store(path)
    store["status"] = {"updated_at": datetime.now().isoformat(), **status}
    _write_store(store, path)


def load_status(path: Path = RESULTS_FILE) -> dict:
    return _read_store(path)["status"]


def leaderboard(studies: dict[str, dict]) -> pd.DataFrame:
    """How each setting did on average across every study run so far.

    `beat_benchmark` is the column that matters: a setting that wins on one
    basket and loses on four others hasn't found an edge, it's found noise.
    """
    rows = []
    for key, study in studies.items():
        df = pd.DataFrame(study["rows"])
        if df.empty or "Buy & hold (benchmark)" not in set(df["scenario"]):
            continue
        benchmark_cagr = float(df[df["scenario"] == "Buy & hold (benchmark)"]["cagr_pct"].iloc[0])
        for _, row in df.iterrows():
            rows.append({
                "scenario": row["scenario"],
                "study": key,
                "cagr_pct": row["cagr_pct"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "commission_paid": row["commission_paid"],
                "beat_benchmark": bool(row["cagr_pct"] > benchmark_cagr),
            })

    if not rows:
        return pd.DataFrame(columns=["scenario", "studies", "avg_cagr_pct", "avg_drawdown_pct",
                                     "avg_fees", "beat_benchmark_pct"])

    long = pd.DataFrame(rows)
    summary = long.groupby("scenario").agg(
        studies=("study", "count"),
        avg_cagr_pct=("cagr_pct", "mean"),
        avg_drawdown_pct=("max_drawdown_pct", "mean"),
        avg_fees=("commission_paid", "mean"),
        beat_benchmark_pct=("beat_benchmark", lambda s: s.mean() * 100),
    ).reset_index()
    return summary.sort_values("avg_cagr_pct", ascending=False).reset_index(drop=True)
