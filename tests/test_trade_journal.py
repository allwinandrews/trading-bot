from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.genai import errors as genai_errors

from ai.trade_journal import _build_prompt, generate_trade_journal


def test_generate_trade_journal_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate_trade_journal("AAPL", "MA crossover", {"total_return_pct": 10}, pd.DataFrame())


def test_build_prompt_includes_key_details():
    trades = pd.DataFrame(
        [{"entry_date": "2024-01-01", "return_pct": 5.0, "exit_reason": "signal_exit"}]
    )
    prompt = _build_prompt("AAPL", "MA crossover", {"total_return_pct": 10}, trades)
    assert "AAPL" in prompt
    assert "MA crossover" in prompt
    assert "signal_exit" in prompt


def test_build_prompt_handles_no_trades():
    prompt = _build_prompt("AAPL", "MA crossover", {"total_return_pct": 0}, pd.DataFrame())
    assert "(no trades)" in prompt


@patch("ai.trade_journal.genai.Client")
def test_generate_trade_journal_calls_api_and_returns_text(mock_client_cls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked journal text."
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    result = generate_trade_journal("AAPL", "MA crossover", {"total_return_pct": 10}, pd.DataFrame())

    assert result == "Mocked journal text."
    mock_client_cls.assert_called_once_with(api_key="test-key")
    mock_client.models.generate_content.assert_called_once()


@patch("ai.trade_journal.genai.Client")
def test_generate_trade_journal_wraps_server_overload_as_runtime_error(mock_client_cls, monkeypatch):
    # Reproduces the real 503 crash seen in the dashboard: the SDK's own
    # ServerError must not leak past this function uncaught.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
    )
    mock_client_cls.return_value = mock_client

    with pytest.raises(RuntimeError, match="overloaded|UNAVAILABLE"):
        generate_trade_journal("AAPL", "MA crossover", {"total_return_pct": 10}, pd.DataFrame())
