import pandas as pd
import pytest

from backtester.scenarios import (
    SPARKLINE_POINTS,
    Scenario,
    _sparkline,
    default_scenarios,
    leaderboard,
    load_status,
    load_studies,
    run_scenario,
    run_scenarios,
    save_status,
    save_study,
)

DATES = pd.date_range("2020-01-01", periods=260, freq="B")


def _trending_prices(start: float, step: float) -> pd.DataFrame:
    closes = [start + step * i for i in range(len(DATES))]
    return pd.DataFrame(
        {"Open": closes, "High": [c * 1.01 for c in closes],
         "Low": [c * 0.99 for c in closes], "Close": closes},
        index=DATES,
    )


PRICES = {"AAA": _trending_prices(100, 0.4), "BBB": _trending_prices(50, 0.2)}
MARKET = _trending_prices(300, 0.5)

_FAST = Scenario(
    name="fast", short_window=5, long_window=20, max_positions=2,
    stop_loss_pct=0.05, regime_filter=False,
)


def test_run_scenario_reports_headline_metrics_and_fees():
    row = run_scenario(PRICES, MARKET, _FAST, initial_capital=500.0)

    assert row["scenario"] == "fast"
    assert row["windows"] == "5/20"
    assert row["positions"] == 2
    assert row["commission_paid"] >= 0
    assert row["final_equity"] > 0
    # Curve is thinned for the table sparkline, not the full daily series.
    assert 0 < len(row["equity_curve"]) <= SPARKLINE_POINTS


def test_sparkline_thins_long_curves_but_leaves_short_ones_alone():
    long_curve = pd.Series(range(1000), dtype=float)
    short_curve = pd.Series(range(10), dtype=float)

    assert len(_sparkline(long_curve)) <= SPARKLINE_POINTS
    assert _sparkline(short_curve) == [float(v) for v in range(10)]


def test_studies_survive_a_save_and_load_round_trip(tmp_path):
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0)
    path = tmp_path / "results.json"

    save_study(df, {"basket": "Blue chips", "period": "Bear market (2022)", "capital": 500.0}, path=path)
    studies = load_studies(path)

    assert "Blue chips | Bear market (2022)" in studies
    saved = studies["Blue chips | Bear market (2022)"]
    assert saved["context"]["capital"] == 500.0
    assert len(saved["rows"]) == len(df)


def test_studies_accumulate_rather_than_overwrite_each_other(tmp_path):
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0)
    path = tmp_path / "results.json"

    save_study(df, {"basket": "A", "period": "P1", "capital": 500.0}, path=path)
    save_study(df, {"basket": "B", "period": "P2", "capital": 500.0}, path=path)

    # The point of the lab is comparing across conditions, so a second study
    # must not replace the first.
    assert set(load_studies(path)) == {"A | P1", "B | P2"}


def test_status_round_trips_alongside_studies(tmp_path):
    path = tmp_path / "results.json"
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0)

    save_study(df, {"basket": "A", "period": "P1", "capital": 500.0}, path=path)
    save_status({"state": "running", "basket": "A"}, path=path)

    assert load_status(path)["state"] == "running"
    assert "A | P1" in load_studies(path)  # writing status didn't clobber studies


def test_missing_or_corrupt_store_reads_as_empty(tmp_path):
    assert load_studies(tmp_path / "nope.json") == {}
    assert load_status(tmp_path / "nope.json") == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json")
    assert load_studies(corrupt) == {}


def test_leaderboard_scores_settings_by_how_often_they_beat_the_benchmark(tmp_path):
    path = tmp_path / "results.json"
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0)
    save_study(df, {"basket": "A", "period": "P1", "capital": 500.0}, path=path)
    save_study(df, {"basket": "B", "period": "P2", "capital": 500.0}, path=path)

    board = leaderboard(load_studies(path))

    assert set(board["scenario"]) == {"fast", "Buy & hold (benchmark)"}
    assert (board["studies"] == 2).all()  # each setting seen in both studies
    assert board["beat_benchmark_pct"].between(0, 100).all()


def test_leaderboard_of_nothing_is_empty_not_an_error():
    assert leaderboard({}).empty


def test_commission_is_actually_charged_when_trades_happen():
    row = run_scenario(PRICES, MARKET, _FAST, initial_capital=500.0)

    if row["num_trades"] > 0 or row["final_equity"] != 500.0:
        assert row["commission_paid"] > 0
        assert row["commission_pct_of_capital"] > 0


def test_run_scenarios_sorts_by_cagr_and_includes_benchmark():
    scenarios = [_FAST, Scenario("slow", 20, 60, 1, 0.05, regime_filter=False)]

    df = run_scenarios(PRICES, MARKET, scenarios, initial_capital=500.0)

    assert "Buy & hold (benchmark)" in set(df["scenario"])
    assert len(df) == 3
    assert list(df["cagr_pct"]) == sorted(df["cagr_pct"], reverse=True)


def test_run_scenarios_can_skip_the_benchmark():
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0, include_benchmark=False)

    assert list(df["scenario"]) == ["fast"]


def test_run_scenarios_reports_progress():
    seen = []
    run_scenarios(
        PRICES, MARKET, [_FAST], initial_capital=500.0,
        progress=lambda done, total, name: seen.append((done, total, name)),
    )

    assert seen[0] == (0, 2, "fast")
    assert seen[-1] == (2, 2, "done")


def test_default_scenarios_vary_one_lever_at_a_time_from_a_baseline():
    scenarios = default_scenarios()
    baseline = scenarios[0]

    assert len(scenarios) >= 8
    assert len({s.name for s in scenarios}) == len(scenarios)  # no duplicate names
    # Every variant differs from the baseline in at least one lever.
    for s in scenarios[1:]:
        assert (
            s.max_positions != baseline.max_positions
            or (s.short_window, s.long_window) != (baseline.short_window, baseline.long_window)
            or s.stop_loss_pct != baseline.stop_loss_pct
            or s.regime_filter != baseline.regime_filter
            or s.risk_per_trade_pct != baseline.risk_per_trade_pct
        )


def test_benchmark_pays_no_commission():
    df = run_scenarios(PRICES, MARKET, [_FAST], initial_capital=500.0)
    benchmark = df[df["scenario"] == "Buy & hold (benchmark)"].iloc[0]

    assert benchmark["commission_paid"] == pytest.approx(0.0)
    assert benchmark["num_trades"] == 0
