"""Persisted record of what the bot actually looked at on its last check.

The screening work (scoring every ticker in the universe, the market-regime
gate, why each candidate was taken or skipped) happens inside the bot's
background process and used to be thrown away the moment it finished — the
dashboard could show the *outcome* of a decision but never the reasoning.
This writes that reasoning to disk each cycle so the Research page can show
the actual numbers the bot acted on, rather than recomputing them (which
would be slow and, worse, might not match what the bot really saw).

One file per strategy log_dir, overwritten each cycle — this is "what is it
doing right now", not an archive.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPORT_FILENAME = "last_screen.json"


def _report_path(log_dir: Path) -> Path:
    return log_dir / REPORT_FILENAME


def save_screen_report(log_dir: Path, report: dict) -> None:
    """Write this cycle's screening report, stamped with when it ran."""
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now().isoformat(), **report}
    _report_path(log_dir).write_text(json.dumps(payload, default=str))


def load_screen_report(log_dir: Path) -> dict | None:
    """The last saved report, or None if the bot hasn't completed a cycle yet."""
    path = _report_path(log_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
