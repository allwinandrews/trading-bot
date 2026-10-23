"""Recent news headlines for a ticker, via yfinance's free news feed.

Purely informational — nothing here feeds into any trading decision. See
README: AI/news-based signals need their own separate, rigorous validation
before ever being wired into live trades, not bolted on directly. This
exists so you can see relevant headlines while watching the bot work, not
to influence what it does.
"""
from __future__ import annotations

import yfinance as yf


def get_recent_news(ticker: str, limit: int = 10) -> list[dict]:
    """Return up to `limit` recent news items for `ticker`.

    Each item: {"title", "summary", "publisher", "published_at", "url"}.
    Returns an empty list (not an error) if yfinance has nothing right now
    — news availability is inherently spotty, that's not a failure.
    """
    raw = yf.Ticker(ticker).news or []
    items = []
    for entry in raw[:limit]:
        content = entry.get("content", {})
        url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url", "")
        provider = content.get("provider") or {}
        items.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "publisher": provider.get("displayName", ""),
            "published_at": content.get("pubDate", ""),
            "url": url,
        })
    return items
