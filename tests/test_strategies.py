import pandas as pd
import pytest

from strategies.moving_average import generate_signals


def test_generate_signals_flat_during_warmup():
    # long_window=3, so the first 2 bars can't have a long MA yet -> must be flat.
    prices = pd.DataFrame({"Close": [10, 10, 10, 10, 10]})
    signals = generate_signals(prices, short_window=1, long_window=3)
    assert (signals.iloc[:2] == 0).all()


def test_generate_signals_goes_long_on_crossover():
    # A rising price series: short MA should cross above long MA and stay long.
    close = pd.Series([10, 10, 10, 11, 13, 16, 20])
    prices = pd.DataFrame({"Close": close})
    signals = generate_signals(prices, short_window=2, long_window=3)
    # By the end of a clear uptrend, the short MA should be above the long MA.
    assert signals.iloc[-1] == 1


def test_generate_signals_rejects_bad_windows():
    prices = pd.DataFrame({"Close": [1, 2, 3]})
    with pytest.raises(ValueError):
        generate_signals(prices, short_window=10, long_window=5)
