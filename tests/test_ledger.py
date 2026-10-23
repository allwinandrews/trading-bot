from execution import ledger


def test_get_cash_seeds_from_initial_capital_on_first_use(tmp_path):
    assert ledger.get_cash(tmp_path, initial_capital=100_000.0) == 100_000.0
    # Second read shouldn't re-seed even if a different initial_capital is passed.
    assert ledger.get_cash(tmp_path, initial_capital=999.0) == 100_000.0


def test_apply_buy_deducts_cost(tmp_path):
    ledger.apply_buy(tmp_path, initial_capital=100_000.0, shares=100, price=50.0)
    assert ledger.get_cash(tmp_path, initial_capital=100_000.0) == 95_000.0


def test_apply_sell_adds_proceeds(tmp_path):
    ledger.set_cash(tmp_path, 95_000.0)
    new_cash = ledger.apply_sell(tmp_path, initial_capital=100_000.0, shares=100, price=55.0)
    assert new_cash == 100_500.0


def test_compute_equity_marks_position_to_market(tmp_path):
    ledger.set_cash(tmp_path, 95_000.0)
    equity = ledger.compute_equity(tmp_path, initial_capital=100_000.0, shares_held=100, last_price=52.0)
    assert equity == 95_000.0 + 100 * 52.0


def test_compute_equity_when_flat_equals_cash(tmp_path):
    equity = ledger.compute_equity(tmp_path, initial_capital=100_000.0, shares_held=0, last_price=52.0)
    assert equity == 100_000.0


def test_corrupt_ledger_file_reseeds_instead_of_crashing(tmp_path):
    path = tmp_path / ledger.LEDGER_FILENAME
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    assert ledger.get_cash(tmp_path, initial_capital=100_000.0) == 100_000.0


def test_compute_equity_multi_sums_several_positions(tmp_path):
    ledger.set_cash(tmp_path, 200.0)
    equity = ledger.compute_equity_multi(
        tmp_path, initial_capital=500.0,
        positions={"AAPL": (2, 330.0), "RIOT": (10, 21.5)},
    )
    assert equity == 200.0 + 2 * 330.0 + 10 * 21.5


def test_compute_equity_multi_when_flat_equals_cash(tmp_path):
    ledger.set_cash(tmp_path, 500.0)
    equity = ledger.compute_equity_multi(tmp_path, initial_capital=500.0, positions={})
    assert equity == 500.0
