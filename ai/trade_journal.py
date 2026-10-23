"""AI-written trade journal.

Turns a backtest's trade log into a plain-English review: what happened, why,
and what's worth being skeptical of. This is where "AI" earns its place in
this project — not predicting prices, but explaining decisions after the
fact so there's something to actually learn from a run, not just a number.

Requires GEMINI_API_KEY (see .env.example). Treated as optional everywhere
it's called from — see run_backtest.py's --ai-journal flag — so a missing key
never breaks a plain backtest run.
"""
from __future__ import annotations

import os

import pandas as pd
from google import genai
from google.genai import errors as genai_errors

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def _build_prompt(ticker: str, strategy_desc: str, metrics: dict, trades: pd.DataFrame) -> str:
    trades_text = trades.to_string(index=False) if not trades.empty else "(no trades)"
    metrics_text = "\n".join(f"- {k}: {v}" for k, v in metrics.items())

    return f"""You are a skeptical trading analyst reviewing a backtest, not a promoter of it.

Strategy: {strategy_desc}
Ticker: {ticker}

Summary metrics:
{metrics_text}

Trade log:
{trades_text}

Write a short trade journal (under 400 words) covering:
1. What actually happened, in plain English — no jargon dump.
2. One or two specific notable trades from the log (best and worst), and why they likely played out that way.
3. Any pattern worth being skeptical of (e.g. repeated stop-losses in a choppy period, or the \
strategy lagging a strong trend).
4. One honest, concrete suggestion for what to investigate or tune next.

Be direct and specific to the numbers given — don't hedge with generic disclaimers, and don't \
oversell the result."""


def generate_trade_journal(
    ticker: str,
    strategy_desc: str,
    metrics: dict,
    trades: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> str:
    """Returns the journal text.

    Raises RuntimeError (not an API-specific exception) for both a missing
    GEMINI_API_KEY and any failure from the Gemini API itself (rate limits,
    temporary 503 overload, etc.), so callers only ever need to catch one
    clear thing regardless of what went wrong underneath.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set - copy .env.example to .env and add your key, "
            "or omit --ai-journal."
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(ticker, strategy_desc, metrics, trades)

    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except genai_errors.ServerError as e:
        raise RuntimeError(
            f"Gemini is temporarily overloaded ({e.code} {e.status}) - try again in a moment."
        ) from e
    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini API error ({e.code} {e.status}): {e.message}") from e

    return response.text
