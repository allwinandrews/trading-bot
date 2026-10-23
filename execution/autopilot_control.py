"""Start/stop the autopilot trading loop as a background OS process, and
check whether it's currently running.

The dashboard can't just run run_paper_trader.py's infinite --loop inside
its own request-handling process — that would block the whole web page
forever. Instead this spawns it as a separate process and tracks it via a
small state file (PID, ticker, when it started) on disk, not in Streamlit's
session_state — so Start/Stop/status work correctly from any browser tab
or after a dashboard restart, not tied to one browser session.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
STATE_DIR = PROJECT_ROOT / "execution" / "live_data"

# "position" keeps the original, pre-multi-strategy file names so the
# instance already running under them isn't orphaned by this change. Any
# other strategy name gets its own suffixed state/log/trade-log files so
# multiple strategies can run — and be told apart — at once.
_DEFAULT_STRATEGY = "position"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows-only; 0 (no-op) elsewhere

_REGISTRY_FILE = STATE_DIR / "strategy_registry.json"


def _load_registry() -> dict:
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(_REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry_entry(strategy: str, config: dict) -> None:
    """Remember a strategy's launch config so the UI can list it (running or
    not) and offer to restart it with the same settings — otherwise a
    stopped strategy's configuration would be lost the moment its transient
    state file is deleted."""
    registry = _load_registry()
    registry[strategy] = config
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _REGISTRY_FILE.write_text(json.dumps(registry))


def list_strategies() -> dict[str, dict]:
    """Every strategy ever started via start_autopilot, with its saved
    config merged with its current live status — running instances not yet
    in the registry (shouldn't normally happen) are included too."""
    registry = _load_registry()
    if _DEFAULT_STRATEGY not in registry and get_status(_DEFAULT_STRATEGY)["running"]:
        registry[_DEFAULT_STRATEGY] = {}  # the pre-registry "position" instance, e.g. from before this change

    result = {}
    for name, config in registry.items():
        result[name] = {"config": config, "status": get_status(name)}
    return result


def _state_file(strategy: str) -> Path:
    if strategy == _DEFAULT_STRATEGY:
        return STATE_DIR / "autopilot_state.json"
    return STATE_DIR / f"autopilot_state_{strategy}.json"


def _log_file(strategy: str) -> Path:
    if strategy == _DEFAULT_STRATEGY:
        return PROJECT_ROOT / "autopilot.log"
    return PROJECT_ROOT / f"autopilot_{strategy}.log"


def trade_log_dir(strategy: str) -> Path:
    """Where this strategy's trades/equity/virtual-ledger are recorded.

    Kept separate per strategy (other than the original "position" one,
    which keeps using execution/live_data/ directly) so two strategies
    trading through the same real IBKR paper account don't mix results —
    see execution/ledger.py for why that matters for equity in particular.
    """
    if strategy == _DEFAULT_STRATEGY:
        return STATE_DIR
    return STATE_DIR / strategy


def _read_state(strategy: str) -> dict | None:
    path = _state_file(strategy)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(strategy: str, state: dict | None) -> None:
    path = _state_file(strategy)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if state is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(json.dumps(state))


def _pid_is_running(pid: int) -> bool:
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
    return str(pid) in result.stdout


def get_status(strategy: str = _DEFAULT_STRATEGY) -> dict:
    """{"running": False} or {"running": True, "pid", "ticker", "interval_seconds", "started_at"}."""
    state = _read_state(strategy)
    if state is None:
        return {"running": False}
    if not _pid_is_running(state["pid"]):
        _write_state(strategy, None)  # process died/was killed outside our control — clean up stale state
        return {"running": False}
    return {"running": True, **state}


def start_autopilot(
    ticker: str | None,
    interval_seconds: int,
    strategy: str = _DEFAULT_STRATEGY,
    short_window: int | None = None,
    long_window: int | None = None,
    stop_loss_pct: float | None = None,
    client_id: int | None = None,
    initial_capital: float | None = None,
    script: str = "run_paper_trader.py",
    extra_args: list[str] | None = None,
) -> None:
    """Raises RuntimeError if this strategy's autopilot is already running (stop it first).

    `short_window`/`long_window`/`stop_loss_pct`/`client_id`/`initial_capital`
    are left as run_paper_trader.py's own defaults (50/200-day, 2%, client 1,
    $100k) when omitted — only strategies other than the default need to
    override these to trade a different style side by side.

    `ticker=None` and `script="run_portfolio_trader.py"` launches the
    portfolio bot instead — it screens a universe rather than trading one
    named ticker, so no `--ticker` flag is passed at all; `extra_args` lets
    a caller pass that script's own extra flags (e.g. --max-positions)
    without this function needing to know every script's whole CLI.
    """
    if get_status(strategy)["running"]:
        raise RuntimeError(f"Autopilot ({strategy}) is already running — stop it first.")

    cmd = [str(PYTHON), script, "--loop", "--interval-seconds", str(interval_seconds)]
    if ticker is not None:
        cmd += ["--ticker", ticker]
    if short_window is not None:
        cmd += ["--short-window", str(short_window)]
    if long_window is not None:
        cmd += ["--long-window", str(long_window)]
    if stop_loss_pct is not None:
        cmd += ["--stop-loss-pct", str(stop_loss_pct)]
    if client_id is not None:
        cmd += ["--client-id", str(client_id)]
    if initial_capital is not None:
        cmd += ["--initial-capital", str(initial_capital)]
    if strategy != _DEFAULT_STRATEGY:
        cmd += ["--log-dir", str(trade_log_dir(strategy))]
    if extra_args:
        cmd += extra_args

    _save_registry_entry(strategy, {
        "ticker": ticker, "interval_seconds": interval_seconds,
        "short_window": short_window, "long_window": long_window,
        "stop_loss_pct": stop_loss_pct, "client_id": client_id,
        "initial_capital": initial_capital, "script": script, "extra_args": extra_args,
    })

    log_handle = open(_log_file(strategy), "a")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle, stderr=subprocess.STDOUT,
        creationflags=_CREATE_NO_WINDOW,
    )
    _write_state(strategy, {
        "pid": proc.pid, "ticker": ticker or "PORTFOLIO", "interval_seconds": interval_seconds,
        "started_at": datetime.now().isoformat(),
    })


def stop_autopilot(strategy: str = _DEFAULT_STRATEGY) -> bool:
    """Returns False if it wasn't running, True if it was and is now stopped."""
    status = get_status(strategy)
    if not status["running"]:
        return False
    subprocess.run(["taskkill", "/PID", str(status["pid"]), "/F"], capture_output=True)
    _write_state(strategy, None)
    return True


def read_recent_log(strategy: str = _DEFAULT_STRATEGY, lines: int = 40) -> str:
    path = _log_file(strategy)
    if not path.exists():
        return "(no log yet)"
    text = path.read_text(errors="ignore")
    return "\n".join(text.splitlines()[-lines:]) or "(log is empty)"
