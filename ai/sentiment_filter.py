"""News-sentiment veto for the portfolio bot's entry decisions.

Deliberately a *filter*, not a decision-maker: the statistical screener
(strategies/screener.py) already decided a ticker is a good candidate on
trend/momentum grounds. This only ever asks "is there a clear reason from
recent news to skip it anyway" — it can block an entry, it never creates one
and never sizes or times a trade. Matches the project's existing stance on
AI (see data/news.py, ai/trade_journal.py): useful as a narrator/filter, not
as the thing making the actual call.

Fails open: any problem (no API key, no news, an API error) returns
"neutral", which never blocks an entry. A missing sentiment check should
never be why a good candidate gets skipped.
"""
from __future__ import annotations

import logging
import os

from google import genai
from google.genai import errors as genai_errors

from data.news import get_recent_news

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

VERDICT_POSITIVE = "positive"
VERDICT_NEUTRAL = "neutral"
VERDICT_NEGATIVE = "negative"

_VALID_VERDICTS = {VERDICT_POSITIVE, VERDICT_NEUTRAL, VERDICT_NEGATIVE}

_PROMPT_TEMPLATE = """You are screening recent news for {ticker} for a single question: \
is there a clear, material red flag (fraud, major lawsuit, regulatory action, guidance cut, \
leadership scandal, etc.) that would make it unwise to open a new long position right now?

Recent headlines:
{headlines}

Respond with exactly one word on the first line: positive, neutral, or negative.
- negative: a clear, material red flag is present in these headlines.
- positive: clearly good news (e.g. a strong earnings beat, positive guidance).
- neutral: nothing headline-worthy either way, or the headlines are too thin/irrelevant to judge.
Then a second line with a one-sentence reason. Default to neutral when genuinely unsure — \
don't manufacture a signal from routine or ambiguous headlines."""


def score_sentiment(ticker: str, model: str = DEFAULT_MODEL) -> dict:
    """{"verdict": "positive"|"neutral"|"negative", "reason": str}.

    Always returns a verdict — never raises. Falls back to "neutral" (the
    non-blocking default) for every failure mode: missing API key, no news
    available, or a Gemini API error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"verdict": VERDICT_NEUTRAL, "reason": "GEMINI_API_KEY not set — sentiment check skipped."}

    try:
        news = get_recent_news(ticker, limit=5)
    except Exception as e:
        logger.warning("%s: news fetch failed — treating as neutral (%s)", ticker, e)
        return {"verdict": VERDICT_NEUTRAL, "reason": f"News fetch failed: {e}"}

    if not news:
        return {"verdict": VERDICT_NEUTRAL, "reason": "No recent news available."}

    headlines = "\n".join(f"- {item['title']}" for item in news if item.get("title"))
    if not headlines:
        return {"verdict": VERDICT_NEUTRAL, "reason": "No usable headlines."}

    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, headlines=headlines)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
    except (genai_errors.ServerError, genai_errors.APIError) as e:
        logger.warning("%s: sentiment check failed — treating as neutral (%s)", ticker, e)
        return {"verdict": VERDICT_NEUTRAL, "reason": f"Sentiment check failed: {e}"}

    lines = (response.text or "").strip().splitlines()
    verdict = lines[0].strip().lower() if lines else ""
    reason = lines[1].strip() if len(lines) > 1 else ""

    if verdict not in _VALID_VERDICTS:
        logger.warning("%s: unexpected sentiment response %r — treating as neutral", ticker, verdict)
        return {"verdict": VERDICT_NEUTRAL, "reason": "Unrecognized model response."}

    return {"verdict": verdict, "reason": reason}
