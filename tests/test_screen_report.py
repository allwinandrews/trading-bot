from execution import screen_report


def test_load_returns_none_before_any_cycle_has_run(tmp_path):
    assert screen_report.load_screen_report(tmp_path) is None


def test_save_then_load_round_trips_and_stamps_a_timestamp(tmp_path):
    screen_report.save_screen_report(tmp_path, {"regime_ok": True, "candidates": [{"ticker": "KO"}]})

    loaded = screen_report.load_screen_report(tmp_path)

    assert loaded["regime_ok"] is True
    assert loaded["candidates"] == [{"ticker": "KO"}]
    assert "timestamp" in loaded


def test_each_save_replaces_the_previous_cycle(tmp_path):
    screen_report.save_screen_report(tmp_path, {"regime_ok": True})
    screen_report.save_screen_report(tmp_path, {"regime_ok": False})

    assert screen_report.load_screen_report(tmp_path)["regime_ok"] is False


def test_corrupt_report_reads_as_none_instead_of_crashing_the_page(tmp_path):
    (tmp_path / screen_report.REPORT_FILENAME).write_text("not json")

    assert screen_report.load_screen_report(tmp_path) is None
