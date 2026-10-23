import pandas as pd
import pytest

from strategies.screener import rank_candidates

DATES = pd.date_range("2024-01-01", periods=6, freq="B")


def _prices(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=DATES[: len(closes)])


UPTREND_MARKET = _prices([100, 101, 102, 103, 104, 106])
DOWNTREND_MARKET = _prices([106, 104, 102, 101, 100, 98])

_COMMON_KWARGS = dict(short_window=2, long_window=3, momentum_window=2, regime_window=2)


def test_rank_candidates_includes_qualifying_uptrend_ticker():
    prices_by_ticker = {"AAA": _prices([100, 101, 102, 103, 104, 106])}

    result = rank_candidates(prices_by_ticker, UPTREND_MARKET, **_COMMON_KWARGS)

    assert list(result["ticker"]) == ["AAA"]
    assert result.loc[0, "efficiency_ratio"] == pytest.approx(1.0)
    assert result.loc[0, "momentum_pct"] == pytest.approx((106 / 104 - 1) * 100, abs=0.01)
    assert result.loc[0, "score"] > 0


def test_rank_candidates_excludes_downtrend_ticker():
    prices_by_ticker = {"BBB": _prices([106, 104, 102, 101, 100, 98])}

    result = rank_candidates(prices_by_ticker, UPTREND_MARKET, **_COMMON_KWARGS)

    assert result.empty


def test_rank_candidates_excludes_ticker_with_nan_price():
    prices_by_ticker = {"CCC": _prices([100, 101, 102, 103, 104, float("nan")])}

    result = rank_candidates(prices_by_ticker, UPTREND_MARKET, **_COMMON_KWARGS)

    assert result.empty


def test_rank_candidates_excludes_ticker_with_insufficient_history():
    prices_by_ticker = {"DDD": _prices([100, 106])}  # shorter than long_window=3

    result = rank_candidates(prices_by_ticker, UPTREND_MARKET, **_COMMON_KWARGS)

    assert result.empty


def test_rank_candidates_excludes_when_market_regime_is_off():
    prices_by_ticker = {"AAA": _prices([100, 101, 102, 103, 104, 106])}

    result = rank_candidates(prices_by_ticker, DOWNTREND_MARKET, **_COMMON_KWARGS)

    assert result.empty


def test_rank_candidates_sorts_multiple_qualifiers_by_score_descending():
    prices_by_ticker = {
        "SLOW": _prices([100, 100.5, 101, 101.5, 102, 102.5]),   # gentle, steady uptrend
        "FAST": _prices([100, 102, 104, 106, 108, 112]),          # steeper uptrend, more momentum
    }

    result = rank_candidates(prices_by_ticker, UPTREND_MARKET, **_COMMON_KWARGS)

    assert list(result["ticker"]) == ["FAST", "SLOW"]
    assert result.loc[0, "score"] > result.loc[1, "score"]


def test_rank_candidates_returns_empty_dataframe_with_expected_columns_when_nothing_qualifies():
    result = rank_candidates({}, UPTREND_MARKET, **_COMMON_KWARGS)

    assert result.empty
    assert list(result.columns) == ["ticker", "last_price", "efficiency_ratio", "momentum_pct", "score"]
