"""Alerts for problems the bot hit on its own — invalid data, state
mismatches, failed checks — surfaced prominently on the dashboard instead
of only sitting in a log file nobody's watching.

Real incident this exists for: a NaN-data bug silently cancelled a real
protective stop and left a position unprotected for hours, only noticed
because someone happened to spot "+nan%" in the UI. These alerts are meant
to be impossible to miss instead — a banner, not a log line.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).parent / "live_data"
ALERTS_FILE = STATE_DIR / "alerts.json"

MAX_ALERTS = 200  # keep the file bounded, oldest fall off


def record_alert(severity: str, message: str, source: str = "") -> None:
    """severity: 'warning' or 'error'."""
    alerts = _read_alerts()
    alerts.append({
        "id": f"{datetime.now().isoformat()}-{len(alerts)}",
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "message": message,
        "source": source,
        "acknowledged": False,
    })
    _write_alerts(alerts[-MAX_ALERTS:])


def load_alerts(unacknowledged_only: bool = False) -> list[dict]:
    """Newest first."""
    alerts = _read_alerts()
    if unacknowledged_only:
        alerts = [a for a in alerts if not a["acknowledged"]]
    return list(reversed(alerts))


def acknowledge_alert(alert_id: str) -> None:
    alerts = _read_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            a["acknowledged"] = True
    _write_alerts(alerts)


def acknowledge_all() -> None:
    alerts = _read_alerts()
    for a in alerts:
        a["acknowledged"] = True
    _write_alerts(alerts)


def _read_alerts() -> list[dict]:
    if not ALERTS_FILE.exists():
        return []
    try:
        return json.loads(ALERTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _write_alerts(alerts: list[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_FILE.write_text(json.dumps(alerts))
