"""Persistent checklist for "is this bot ready to trade real money yet."

The canonical checklist items live here, not in the dashboard page, so
there's one source of truth. Each item is either:
  - "auto": verifiable from real project state (a file exists, a trade
    count, an env var) — computed fresh every time, never hand-ticked.
  - manual (no "auto" key): genuinely subjective (risk tolerance,
    willingness to not intervene) — no code can determine these, so they
    stay a plain checkbox. State for manual items is a small JSON file on
    disk (like execution/autopilot_control.py's), so progress survives
    dashboard restarts and isn't tied to one browser session.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STATE_DIR = PROJECT_ROOT / "execution" / "live_data"
STATE_FILE = STATE_DIR / "readiness_checklist.json"
VALIDATION_START_FILE = STATE_DIR / "validation_started_at.json"

VALIDATION_TARGET_WEEKS = 4.0
TRADE_COUNT_TARGET = 15


def _file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle.lower() in path.read_text(errors="ignore").lower()


def _check_filters_wired() -> bool:
    """Has the live execution path actually been updated to use the tested filters?

    A lightweight but real check — looks for actual usage in the source,
    not just that the strategy modules exist somewhere in the repo.
    """
    path = PROJECT_ROOT / "execution" / "paper_trader.py"
    return _file_contains(path, "regime") and _file_contains(path, "trendiness")


def _check_rotation_explored() -> bool:
    return (PROJECT_ROOT / "strategies" / "momentum_rotation.py").exists()


def _check_vol_sizing_explored() -> bool:
    return _file_contains(PROJECT_ROOT / "risk" / "position_sizing.py", "volatility")


def _check_news_filter_tested() -> bool:
    return _file_contains(PROJECT_ROOT / "execution" / "paper_trader.py", "news")


def _check_account_id_set() -> bool:
    return bool(os.environ.get("IB_ACCOUNT_ID", "").strip())


def get_validation_days(autopilot_running: bool) -> float | None:
    """Days since autopilot was first ever observed running. None if never started.

    An approximation, not a strict "continuous, uninterrupted" measure — it's
    wall-clock time since first observed, not accounting for gaps if
    autopilot was stopped in between. Good enough as a validation-window
    proxy; the "no manual intervention" half of that criterion still needs
    a human judgment call, not just this number.
    """
    data = {}
    if VALIDATION_START_FILE.exists():
        try:
            data = json.loads(VALIDATION_START_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    if "started_at" not in data:
        if not autopilot_running:
            return None
        data["started_at"] = datetime.now().isoformat()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATION_START_FILE.write_text(json.dumps(data))

    started = datetime.fromisoformat(data["started_at"])
    return (datetime.now() - started).total_seconds() / 86400


PENDING_ITEMS = [
    {"id": "wire_filters", "auto": True, "text": "Wire the trendiness + regime filters into the live execution path",
     "note": "Autopilot currently runs plain MA50/200 — not the config that got Sharpe to 0.99"},
    {"id": "explore_rotation", "auto": True, "text": "Explore momentum/relative-strength rotation as an alternative strategy family",
     "note": "Auto-detected: checks for strategies/momentum_rotation.py"},
    {"id": "explore_vol_sizing", "auto": True, "text": "Explore inverse-volatility position sizing",
     "note": "Auto-detected: checks risk/position_sizing.py for volatility-based sizing"},
    {"id": "test_news_filter", "auto": True, "text": "Test news/sentiment as an actual entry filter",
     "note": "Auto-detected: checks whether execution/paper_trader.py actually uses news data"},
    {"id": "set_account_id", "auto": True, "text": "Set IB_ACCOUNT_ID in .env",
     "note": "Auto-detected from your environment"},
]

PHASE_1_VALIDATION = [
    {"id": "p1-config", "auto": True, "text": "Live autopilot runs the validated config (trendiness + regime filters)",
     "note": "Same check as \"wire the filters\" above"},
    {"id": "p1-uptime", "auto": True, "text": f"{VALIDATION_TARGET_WEEKS:.0f}+ weeks since autopilot first started validating",
     "note": "Survives market opens/closes, IB Gateway restarts, weekends — see the live day count above"},
    {"id": "p1-trades", "auto": True, "text": f"{TRADE_COUNT_TARGET}+ real closed trades accumulated",
     "note": "Below this, win rate and average return are statistical noise"},
    {"id": "p1-nobugs", "text": "No new code bugs surface during the window",
     "note": "3 real bugs were caught and fixed on day one: stop-loss rounding, a clientId collision, an uncaught API crash"},
    {"id": "p1-tracks", "text": "Live Sharpe / drawdown roughly tracks the backtest's expectations",
     "note": "Large divergence (worse fills, more slippage) is a reason to keep paper trading, not push through"},
]

PHASE_2_GO_LIVE = [
    {"id": "p2-phase1", "auto": True, "text": "Every item in Phase 1 is checked", "note": ""},
    {"id": "p2-lose", "text": "Fully prepared to lose 100% of whatever capital is committed",
     "note": "Per every IBKR risk disclosure already reviewed — no guarantees exist"},
    {"id": "p2-nointerfere", "text": "Comfortable letting the bot run without second-guessing or manual overrides", "note": ""},
]

PHASE_3_STEPS = [
    "Fund with a small, disposable amount — £500-£1,000 minimum for whole-share position sizing to work sensibly",
    "Change IB_PORT from 4002 (paper) to 4001 (live) in .env only after every item above is checked",
    "After the first live entry, manually confirm the protective stop is actually resting (broker.open_orders()) — don't assume it",
    "Keep the same monitoring cadence that worked in paper: Autopilot page, Trade History, regular log checks",
    "Scale up capital only after sustained real profitability — never on a calendar",
]


def _read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_checked() -> dict[str, bool]:
    """Manually-checked items only — {item_id: True}. Auto items aren't stored here."""
    return _read_state()


def set_checked(item_id: str, checked: bool) -> None:
    state = _read_state()
    if checked:
        state[item_id] = True
    else:
        state.pop(item_id, None)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def compute_effective_status(closed_trade_count: int, autopilot_running: bool) -> dict[str, bool]:
    """The full, current true/false for every item — auto ones computed fresh, manual ones read from disk."""
    status = dict(load_checked())  # manual items start here

    filters_wired = _check_filters_wired()
    validation_days = get_validation_days(autopilot_running)
    uptime_ok = validation_days is not None and validation_days >= VALIDATION_TARGET_WEEKS * 7
    trades_ok = closed_trade_count >= TRADE_COUNT_TARGET

    status["wire_filters"] = filters_wired
    status["explore_rotation"] = _check_rotation_explored()
    status["explore_vol_sizing"] = _check_vol_sizing_explored()
    status["test_news_filter"] = _check_news_filter_tested()
    status["set_account_id"] = _check_account_id_set()

    status["p1-config"] = filters_wired
    status["p1-uptime"] = uptime_ok
    status["p1-trades"] = trades_ok
    # p1-nobugs, p1-tracks: left as manually set (or absent -> False)

    phase_1_ids = [item["id"] for item in PHASE_1_VALIDATION]
    status["p2-phase1"] = all(status.get(i, False) for i in phase_1_ids)
    # p2-lose, p2-nointerfere: left as manually set (or absent -> False)

    return status
