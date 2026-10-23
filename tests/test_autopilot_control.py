import json
from unittest.mock import MagicMock, patch

import execution.autopilot_control as ac


def _patch_files(monkeypatch, tmp_path):
    """Point the module's per-strategy file lookups at a temp dir regardless
    of which strategy name is passed, so tests don't touch real state."""
    state_file = tmp_path / "state.json"
    log_file = tmp_path / "autopilot.log"
    monkeypatch.setattr(ac, "_state_file", lambda strategy: state_file)
    monkeypatch.setattr(ac, "_log_file", lambda strategy: log_file)
    monkeypatch.setattr(ac, "STATE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_REGISTRY_FILE", tmp_path / "strategy_registry.json")
    return state_file, log_file


def test_get_status_when_never_started(tmp_path, monkeypatch):
    _patch_files(monkeypatch, tmp_path)
    assert ac.get_status() == {"running": False}


@patch("execution.autopilot_control._pid_is_running", return_value=True)
def test_get_status_when_running(mock_pid_check, tmp_path, monkeypatch):
    state_file, _ = _patch_files(monkeypatch, tmp_path)
    state_file.write_text(json.dumps({
        "pid": 1234, "ticker": "AAPL", "interval_seconds": 300, "started_at": "2026-01-01T00:00:00",
    }))

    status = ac.get_status()
    assert status == {
        "running": True, "pid": 1234, "ticker": "AAPL",
        "interval_seconds": 300, "started_at": "2026-01-01T00:00:00",
    }


@patch("execution.autopilot_control._pid_is_running", return_value=False)
def test_get_status_cleans_up_stale_state(mock_pid_check, tmp_path, monkeypatch):
    # Process died outside our control (crashed, manually killed) — status
    # should reflect reality and clean up the leftover state file.
    state_file, _ = _patch_files(monkeypatch, tmp_path)
    state_file.write_text(json.dumps({"pid": 1234, "ticker": "AAPL", "interval_seconds": 300, "started_at": "x"}))

    assert ac.get_status() == {"running": False}
    assert not state_file.exists()


@patch("execution.autopilot_control.subprocess.Popen")
def test_start_autopilot_writes_state(mock_popen, tmp_path, monkeypatch):
    state_file, _ = _patch_files(monkeypatch, tmp_path)

    mock_proc = MagicMock()
    mock_proc.pid = 5678
    mock_popen.return_value = mock_proc

    ac.start_autopilot("AAPL", 300)

    state = json.loads(state_file.read_text())
    assert state["pid"] == 5678
    assert state["ticker"] == "AAPL"
    assert state["interval_seconds"] == 300


@patch("execution.autopilot_control.subprocess.Popen")
def test_start_autopilot_passes_strategy_overrides_as_cli_flags(mock_popen, tmp_path, monkeypatch):
    # A second, concurrently-running strategy (e.g. swing trading) needs its
    # own windows/stop/client id/capital/log-dir so it doesn't collide with
    # or get mixed into another strategy sharing the same real IBKR account.
    _patch_files(monkeypatch, tmp_path)

    mock_proc = MagicMock()
    mock_proc.pid = 5678
    mock_popen.return_value = mock_proc

    ac.start_autopilot(
        "MSFT", 60, strategy="swing",
        short_window=10, long_window=30, stop_loss_pct=0.03,
        client_id=3, initial_capital=50_000.0,
    )

    cmd = mock_popen.call_args[0][0]
    assert "--short-window" in cmd and "10" in cmd
    assert "--long-window" in cmd and "30" in cmd
    assert "--stop-loss-pct" in cmd and "0.03" in cmd
    assert "--client-id" in cmd and "3" in cmd
    assert "--initial-capital" in cmd and "50000.0" in cmd
    assert "--log-dir" in cmd


@patch("execution.autopilot_control._pid_is_running", return_value=True)
@patch("execution.autopilot_control.subprocess.Popen")
def test_start_autopilot_refuses_if_already_running(mock_popen, mock_pid_check, tmp_path, monkeypatch):
    state_file, _ = _patch_files(monkeypatch, tmp_path)
    state_file.write_text(json.dumps({"pid": 1234, "ticker": "AAPL", "interval_seconds": 300, "started_at": "x"}))

    try:
        ac.start_autopilot("MSFT", 60)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    mock_popen.assert_not_called()


@patch("execution.autopilot_control.subprocess.run")
@patch("execution.autopilot_control._pid_is_running", return_value=True)
def test_stop_autopilot_when_running(mock_pid_check, mock_run, tmp_path, monkeypatch):
    state_file, _ = _patch_files(monkeypatch, tmp_path)
    state_file.write_text(json.dumps({"pid": 1234, "ticker": "AAPL", "interval_seconds": 300, "started_at": "x"}))

    result = ac.stop_autopilot()

    assert result is True
    assert not state_file.exists()
    mock_run.assert_called_once()


def test_stop_autopilot_when_not_running(tmp_path, monkeypatch):
    _patch_files(monkeypatch, tmp_path)
    assert ac.stop_autopilot() is False


def test_read_recent_log_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "_log_file", lambda strategy: tmp_path / "nope.log")
    assert ac.read_recent_log() == "(no log yet)"


def test_read_recent_log_tails_last_lines(tmp_path, monkeypatch):
    log_file = tmp_path / "autopilot.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(100)))
    monkeypatch.setattr(ac, "_log_file", lambda strategy: log_file)

    result = ac.read_recent_log(lines=5)
    assert result == "\n".join(f"line {i}" for i in range(95, 100))


def test_trade_log_dir_default_strategy_is_shared_live_data_dir():
    assert ac.trade_log_dir("position") == ac.STATE_DIR


def test_trade_log_dir_other_strategy_gets_own_subdirectory():
    assert ac.trade_log_dir("swing") == ac.STATE_DIR / "swing"


@patch("execution.autopilot_control.subprocess.Popen")
def test_start_autopilot_saves_registry_entry(mock_popen, tmp_path, monkeypatch):
    _patch_files(monkeypatch, tmp_path)
    mock_popen.return_value = MagicMock(pid=1)

    ac.start_autopilot("PLTR", 120, strategy="swing_pltr", short_window=10, long_window=30, client_id=4)

    registry = ac._load_registry()
    assert registry["swing_pltr"]["ticker"] == "PLTR"
    assert registry["swing_pltr"]["short_window"] == 10
    assert registry["swing_pltr"]["client_id"] == 4


@patch("execution.autopilot_control.subprocess.Popen")
def test_start_autopilot_with_no_ticker_omits_ticker_flag(mock_popen, tmp_path, monkeypatch):
    # The portfolio bot has no single ticker (it screens a universe) — see
    # run_portfolio_trader.py. ticker=None must not add a bare "--ticker"
    # flag with nothing after it.
    _patch_files(monkeypatch, tmp_path)
    mock_popen.return_value = MagicMock(pid=1)

    ac.start_autopilot(
        None, 300, strategy="portfolio", client_id=5,
        script="run_portfolio_trader.py", extra_args=["--max-positions", "4"],
    )

    cmd = mock_popen.call_args[0][0]
    assert "--ticker" not in cmd
    assert "run_portfolio_trader.py" in cmd
    assert "--max-positions" in cmd and "4" in cmd

    registry = ac._load_registry()
    assert registry["portfolio"]["ticker"] is None
    assert registry["portfolio"]["script"] == "run_portfolio_trader.py"


@patch("execution.autopilot_control.subprocess.Popen")
def test_list_strategies_merges_registry_with_live_status(mock_popen, tmp_path, monkeypatch):
    state_files = {}

    def state_file_for(strategy):
        return state_files.setdefault(strategy, tmp_path / f"state_{strategy}.json")

    monkeypatch.setattr(ac, "_state_file", state_file_for)
    monkeypatch.setattr(ac, "_log_file", lambda strategy: tmp_path / f"log_{strategy}.log")
    monkeypatch.setattr(ac, "STATE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_REGISTRY_FILE", tmp_path / "strategy_registry.json")

    mock_popen.return_value = MagicMock(pid=42)
    ac.start_autopilot("MSFT", 300, strategy="swing_msft", client_id=3)

    with patch("execution.autopilot_control._pid_is_running", return_value=True):
        strategies = ac.list_strategies()

    assert "swing_msft" in strategies
    assert strategies["swing_msft"]["config"]["ticker"] == "MSFT"
    assert strategies["swing_msft"]["status"]["running"] is True
