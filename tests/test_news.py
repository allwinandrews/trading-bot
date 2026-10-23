from unittest.mock import MagicMock, patch

from data.news import get_recent_news


@patch("data.news.yf.Ticker")
def test_get_recent_news_parses_items(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {
            "content": {
                "title": "Some headline",
                "summary": "Some summary",
                "pubDate": "2026-01-01T00:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/a"},
            }
        }
    ]
    mock_ticker_cls.return_value = mock_ticker

    items = get_recent_news("AAPL")

    assert len(items) == 1
    assert items[0] == {
        "title": "Some headline",
        "summary": "Some summary",
        "publisher": "Reuters",
        "published_at": "2026-01-01T00:00:00Z",
        "url": "https://example.com/a",
    }


@patch("data.news.yf.Ticker")
def test_get_recent_news_falls_back_to_click_through_url(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"content": {"title": "X", "clickThroughUrl": {"url": "https://example.com/b"}}}]
    mock_ticker_cls.return_value = mock_ticker

    items = get_recent_news("AAPL")
    assert items[0]["url"] == "https://example.com/b"


@patch("data.news.yf.Ticker")
def test_get_recent_news_handles_no_news(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = []
    mock_ticker_cls.return_value = mock_ticker

    assert get_recent_news("XYZ") == []


@patch("data.news.yf.Ticker")
def test_get_recent_news_handles_none_news(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = None
    mock_ticker_cls.return_value = mock_ticker

    assert get_recent_news("XYZ") == []


@patch("data.news.yf.Ticker")
def test_get_recent_news_respects_limit(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"content": {"title": f"Headline {i}"}} for i in range(20)]
    mock_ticker_cls.return_value = mock_ticker

    items = get_recent_news("AAPL", limit=5)
    assert len(items) == 5
