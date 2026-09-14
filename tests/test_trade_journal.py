from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai.trade_journal import _build_prompt, generate_trade_journal


def test_generate_trade_journal_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
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


@patch("ai.trade_journal.Anthropic")
def test_generate_trade_journal_calls_api_and_returns_text(mock_anthropic_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked journal text.")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    result = generate_trade_journal("AAPL", "MA crossover", {"total_return_pct": 10}, pd.DataFrame())

    assert result == "Mocked journal text."
    mock_anthropic_cls.assert_called_once_with(api_key="test-key")
    mock_client.messages.create.assert_called_once()
