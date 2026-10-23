"""Background worker for the Scenario Lab: keeps studying, on its own.

Usage:
    python run_scenario_lab.py --loop

Cycles through every basket x period combination, running the same set of
strategy settings over each and recording the results. Nothing here trades
or touches the broker — it's pure historical simulation, so it's safe to
leave running and safe to kill at any point.

Deliberately has no inputs to tune mid-flight: the value is in accumulating
comparable studies across many conditions, and a lab you keep fiddling with
produces results you can't compare.
"""
from __future__ import annotations

import argparse
import logging
import time

from backtester.scenarios import (
    default_scenarios,
    load_studies,
    run_scenarios,
    save_status,
    save_study,
    study_key,
)
from data.loader import load_price_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASKETS = {
    "Mega-cap tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Blue chips": ["JNJ", "PG", "KO", "WMT", "XOM"],
    "High volatility": ["NVDA", "AMD", "PLTR", "RIOT", "COIN"],
    "Mixed (broad)": ["AAPL", "MSFT", "JNJ", "KO", "XOM", "NVDA", "AMD", "WMT"],
}

PERIODS = {
    "Bull market (2015-2020)": ("2015-01-01", "2020-01-01"),
    "Covid crash & recovery (2020-2022)": ("2020-01-01", "2022-01-01"),
    "Bear market (2022)": ("2022-01-01", "2023-01-01"),
    "Recent (2022-2024)": ("2022-01-01", "2024-01-01"),
    "Long run (2015-2024)": ("2015-01-01", "2024-01-01"),
}

DEFAULT_CAPITAL = 500.0


def _combinations() -> list[tuple[str, str]]:
    return [(b, p) for b in BASKETS for p in PERIODS]


def _next_combination() -> tuple[str, str]:
    """Whichever combination is least recently studied — so a restart
    continues the sweep rather than redoing what's already known."""
    studies = load_studies()
    unstudied = [c for c in _combinations() if study_key(*c) not in studies]
    if unstudied:
        return unstudied[0]
    # All covered: refresh the oldest, so long-running labs stay current.
    oldest = min(studies.items(), key=lambda kv: kv[1].get("ran_at", ""))
    context = oldest[1]["context"]
    return context["basket"], context["period"]


def run_one(basket: str, period: str, capital: float) -> None:
    tickers = BASKETS[basket]
    start, end = PERIODS[period]

    save_status({"state": "running", "basket": basket, "period": period})
    logger.info("Studying %s over %s", basket, period)

    # A ticker with no data for this period (e.g. before its IPO — PLTR
    # 2020, COIN 2021) must not sink the whole study: real incident where
    # "High volatility | Bull market (2015-2020)" failed every single retry
    # forever because PLTR didn't exist yet, permanently blocking the lab
    # at that combination and never reaching the other studies.
    prices = {}
    for t in tickers:
        try:
            prices[t] = load_price_data(t, start, end)
        except ValueError as e:
            logger.warning("%s: no data for %s — excluding from this study (%s)", t, period, e)
    if len(prices) < 2:
        raise ValueError(f"Too few tickers had data for {basket} over {period} to run a study")

    market = load_price_data("SPY", start, end)

    results = run_scenarios(prices, market, default_scenarios(), initial_capital=capital)
    save_study(results, {"basket": basket, "period": period, "capital": capital})

    best = results.iloc[0]
    logger.info("Done: best was %s at %.1f%%/yr", best["scenario"], best["cagr_pct"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop", action="store_true", help="Keep studying instead of running once")
    p.add_argument("--interval-seconds", type=int, default=900)
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    p.add_argument(
        "--log-dir", default=None,
        help="Accepted and ignored. execution/autopilot_control.py passes this to every "
             "background script it launches; the lab keeps its results in the shared "
             "scenario store (backtester/scenarios.py) rather than a per-strategy log dir.",
    )
    args = p.parse_args()

    while True:
        basket, period = _next_combination()
        try:
            run_one(basket, period, args.capital)
        except Exception as e:  # noqa: BLE001 — one bad basket shouldn't end the lab
            logger.exception("Study failed — moving on")
            save_status({"state": "error", "basket": basket, "period": period, "error": str(e)})

        # Always clear "running" — otherwise a one-shot run leaves the page
        # claiming a study is in progress forever.
        save_status({
            "state": "idle", "studied": len(load_studies()), "total": len(_combinations()),
            "next_in_seconds": args.interval_seconds if args.loop else None,
        })
        if not args.loop:
            return
        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("\nScenario lab stopped.")
            return


if __name__ == "__main__":
    main()
