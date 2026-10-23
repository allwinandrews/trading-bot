from unittest.mock import MagicMock, patch

from ai.sentiment_filter import score_sentiment


def test_score_sentiment_neutral_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = score_sentiment("AAPL")

    assert result["verdict"] == "neutral"


@patch("ai.sentiment_filter.get_recent_news")
def test_score_sentiment_neutral_when_no_news(mock_get_news, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_get_news.return_value = []

    result = score_sentiment("AAPL")

    assert result["verdict"] == "neutral"


@patch("ai.sentiment_filter.get_recent_news")
def test_score_sentiment_neutral_when_news_fetch_raises(mock_get_news, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_get_news.side_effect = RuntimeError("network blip")

    result = score_sentiment("AAPL")

    assert result["verdict"] == "neutral"


@patch("ai.sentiment_filter.genai.Client")
@patch("ai.sentiment_filter.get_recent_news")
def test_score_sentiment_parses_negative_verdict(mock_get_news, mock_client_cls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_get_news.return_value = [{"title": "Company X hit with fraud lawsuit"}]

    mock_response = MagicMock()
    mock_response.text = "negative\nMajor fraud lawsuit just filed."
    mock_client_cls.return_value.models.generate_content.return_value = mock_response

    result = score_sentiment("XCO")

    assert result["verdict"] == "negative"
    assert "fraud" in result["reason"].lower()


@patch("ai.sentiment_filter.genai.Client")
@patch("ai.sentiment_filter.get_recent_news")
def test_score_sentiment_falls_back_to_neutral_on_unrecognized_response(
    mock_get_news, mock_client_cls, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_get_news.return_value = [{"title": "Some headline"}]

    mock_response = MagicMock()
    mock_response.text = "I'm not sure, could go either way"
    mock_client_cls.return_value.models.generate_content.return_value = mock_response

    result = score_sentiment("AAPL")

    assert result["verdict"] == "neutral"


@patch("ai.sentiment_filter.genai.Client")
@patch("ai.sentiment_filter.get_recent_news")
def test_score_sentiment_neutral_on_api_error(mock_get_news, mock_client_cls, monkeypatch):
    from google.genai import errors as genai_errors

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_get_news.return_value = [{"title": "Some headline"}]
    mock_client_cls.return_value.models.generate_content.side_effect = genai_errors.ServerError(
        503, {"error": {"message": "overloaded"}}
    )

    result = score_sentiment("AAPL")

    assert result["verdict"] == "neutral"
