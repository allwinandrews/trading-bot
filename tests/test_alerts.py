import execution.alerts as alerts


def test_load_alerts_empty_when_none_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    assert alerts.load_alerts() == []


def test_record_and_load_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alerts, "STATE_DIR", tmp_path)

    alerts.record_alert("warning", "AAPL: invalid price data", source="run_once")

    loaded = alerts.load_alerts()
    assert len(loaded) == 1
    assert loaded[0]["severity"] == "warning"
    assert loaded[0]["message"] == "AAPL: invalid price data"
    assert loaded[0]["acknowledged"] is False


def test_load_alerts_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alerts, "STATE_DIR", tmp_path)

    alerts.record_alert("warning", "first")
    alerts.record_alert("error", "second")

    loaded = alerts.load_alerts()
    assert [a["message"] for a in loaded] == ["second", "first"]


def test_unacknowledged_only_filters_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alerts, "STATE_DIR", tmp_path)

    alerts.record_alert("warning", "first")
    alerts.record_alert("error", "second")
    first_id = alerts.load_alerts()[-1]["id"]  # the "first" one, oldest
    alerts.acknowledge_alert(first_id)

    unacked = alerts.load_alerts(unacknowledged_only=True)
    assert len(unacked) == 1
    assert unacked[0]["message"] == "second"


def test_acknowledge_all_clears_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alerts, "STATE_DIR", tmp_path)

    alerts.record_alert("warning", "first")
    alerts.record_alert("error", "second")
    alerts.acknowledge_all()

    assert alerts.load_alerts(unacknowledged_only=True) == []
    assert len(alerts.load_alerts()) == 2  # still there, just acknowledged


def test_alerts_file_bounded_to_max(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alerts, "STATE_DIR", tmp_path)
    monkeypatch.setattr(alerts, "MAX_ALERTS", 5)

    for i in range(10):
        alerts.record_alert("warning", f"alert {i}")

    loaded = alerts.load_alerts()
    assert len(loaded) == 5
    assert loaded[0]["message"] == "alert 9"  # newest kept
