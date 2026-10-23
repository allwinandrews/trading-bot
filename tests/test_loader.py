import time
from unittest.mock import patch

import pandas as pd
import pytest

from data import loader


@patch("data.loader.DOWNLOAD_TIMEOUT_SECONDS", 0.2)
@patch("data.loader.yf.download")
def test_watchdog_times_out_a_hung_download_instead_of_blocking_forever(mock_download):
    # Reproduces a real incident: a background worker hung for two full days
    # on one yf.download() call despite yfinance's own internal timeout=10
    # default not actually bounding it. The caller must get control back
    # quickly regardless of how long the underlying call actually takes.
    def hangs(*args, **kwargs):
        time.sleep(5)
        return pd.DataFrame({"Close": [1.0]})

    mock_download.side_effect = hangs

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        loader._download_with_watchdog("AAPL")
    elapsed = time.monotonic() - started

    assert elapsed < 2.0  # bounded by the watchdog, not by the 5s hang


@patch("data.loader.yf.download")
def test_watchdog_passes_through_a_normal_fast_download(mock_download):
    expected = pd.DataFrame({"Close": [100.0, 101.0]})
    mock_download.return_value = expected

    result = loader._download_with_watchdog("AAPL")

    assert result is expected


@patch("data.loader.DOWNLOAD_TIMEOUT_SECONDS", 0.2)
@patch("data.loader.yf.download")
def test_load_price_data_surfaces_the_timeout_to_callers(mock_download, tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)
    mock_download.side_effect = lambda *a, **kw: time.sleep(5)

    with pytest.raises(TimeoutError):
        loader.load_price_data("AAPL", "2020-01-01", "2020-02-01", use_cache=False)
