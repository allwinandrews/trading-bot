import json

import execution.readiness_tracker as rt


def test_load_checked_when_never_set(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "STATE_FILE", tmp_path / "state.json")
    assert rt.load_checked() == {}


def test_set_checked_true_persists(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(rt, "STATE_FILE", state_file)
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)

    rt.set_checked("p1-nobugs", True)

    assert json.loads(state_file.read_text()) == {"p1-nobugs": True}
    assert rt.load_checked() == {"p1-nobugs": True}


def test_set_checked_false_removes_it(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(rt, "STATE_FILE", state_file)
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)
    state_file.write_text(json.dumps({"p1-nobugs": True, "p2-lose": True}))

    rt.set_checked("p1-nobugs", False)

    assert rt.load_checked() == {"p2-lose": True}


def test_checklist_items_have_unique_ids():
    all_items = rt.PENDING_ITEMS + rt.PHASE_1_VALIDATION + rt.PHASE_2_GO_LIVE
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids))


def test_account_id_auto_detected_from_env(monkeypatch):
    monkeypatch.delenv("IB_ACCOUNT_ID", raising=False)
    assert rt._check_account_id_set() is False
    monkeypatch.setenv("IB_ACCOUNT_ID", "U12345678")
    assert rt._check_account_id_set() is True


def test_rotation_explored_checks_for_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "PROJECT_ROOT", tmp_path)
    assert rt._check_rotation_explored() is False
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "momentum_rotation.py").write_text("# placeholder")
    assert rt._check_rotation_explored() is True


def test_validation_days_none_until_autopilot_seen_running(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)
    monkeypatch.setattr(rt, "VALIDATION_START_FILE", tmp_path / "vstart.json")

    assert rt.get_validation_days(autopilot_running=False) is None
    days = rt.get_validation_days(autopilot_running=True)
    assert days is not None and days < 0.01  # just started


def test_validation_days_persists_first_seen_time(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)
    monkeypatch.setattr(rt, "VALIDATION_START_FILE", tmp_path / "vstart.json")

    first = rt.get_validation_days(autopilot_running=True)
    # Calling again (even with running=False) should not reset the start time.
    second = rt.get_validation_days(autopilot_running=False)
    assert second is not None
    assert second >= first


def test_compute_effective_status_trade_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)
    monkeypatch.setattr(rt, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(rt, "VALIDATION_START_FILE", tmp_path / "vstart.json")

    below = rt.compute_effective_status(closed_trade_count=5, autopilot_running=False)
    assert below["p1-trades"] is False

    above = rt.compute_effective_status(closed_trade_count=20, autopilot_running=False)
    assert above["p1-trades"] is True


def test_compute_effective_status_derives_phase2_from_phase1(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(rt, "STATE_DIR", tmp_path)
    monkeypatch.setattr(rt, "STATE_FILE", state_file)
    monkeypatch.setattr(rt, "VALIDATION_START_FILE", tmp_path / "vstart.json")

    # Not enough trades and no manual bug/tracking confirmation -> phase 2 gate stays closed.
    status = rt.compute_effective_status(closed_trade_count=1, autopilot_running=False)
    assert status["p2-phase1"] is False

    # Manually confirm the two subjective phase-1 items, but trades still short.
    rt.set_checked("p1-nobugs", True)
    rt.set_checked("p1-tracks", True)
    status = rt.compute_effective_status(closed_trade_count=1, autopilot_running=False)
    assert status["p2-phase1"] is False  # trades + uptime + config still not met
