# trading-bot

A systematic trading project: backtest a strategy on historical data, paper-trade it against live
data, and (only once proven) run it live through a broker API — with stop-loss, position sizing,
and a daily loss circuit breaker built in from the start. Includes an AI-written trade journal that
explains why each trade fired.

**Status:** step 1 done — backtester + moving-average strategy + risk layer, tested and working.

## Project structure

```
data/          downloaded historical price data (gitignored)
strategies/    strategy logic (signal generation only — no execution)
backtester/    simulates a strategy against historical data, tracks equity/drawdown
risk/          stop-loss, position sizing, daily loss circuit breaker
execution/     broker API wrapper — paper mode and live mode
ai/            trade journal / sentiment layer (Anthropic API)
notebooks/     exploration and result plots
```

## Build order

1. ✅ Backtester + a simple strategy (moving-average crossover) on free historical data, with
   stop-loss and fixed-fractional position sizing baked in — no API keys needed.
2. ✅ Comparison against buy-and-hold (built into `run_backtest.py`'s output).
3. ✅ AI trade journal on top of backtest results (`--ai-journal`, needs `ANTHROPIC_API_KEY`).
4. Paper trading against live data (Interactive Brokers).
5. Streamlit dashboard.
6. Live trading, small size, only after sustained paper-trading validation.

## Running a backtest

```bash
.venv\Scripts\activate
python run_backtest.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
python run_backtest.py --ticker RELIANCE.NS --short-window 10 --long-window 30 --stop-loss-pct 0.03
python run_backtest.py --ticker AAPL --ai-journal   # + a Claude-written review of the trade log
```

Prints a strategy-vs-buy-and-hold metrics comparison and the most recent trades, and saves an
equity curve chart to `results/` (gitignored — regenerate anytime). `--ai-journal` additionally
asks Claude to review the trade log in plain English and saves it to `results/<ticker>_trade_journal.md`
— needs `ANTHROPIC_API_KEY` set in `.env`; without it, the flag just prints a skip notice rather
than failing the run.

## Running tests

```bash
pytest
```

Covers the strategy's signal logic, position sizing, stop-loss pricing, and the backtest
engine's trade simulation (signal exits, stop-loss exits, buy-and-hold) against small synthetic
price series — no network access needed.

## Design notes / known simplifications (v1)

- Long-only, one position at a time, daily bars — no shorting or pyramiding yet.
- A signal computed from day *t*'s close is acted on at day *t+1*'s open, to avoid lookahead bias.
- Stop-losses are checked against the day's Low and assumed to fill exactly at the stop price —
  real fills can slip past the stop on a fast-moving/gapping market.
- The daily loss circuit breaker (`risk/stops.py`) exists but isn't wired into this daily-bar
  backtester (each bar already *is* a day, so it isn't meaningful yet) — it's there for the
  paper/live trading loop, which will run at finer granularity.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then fill in real keys — never commit .env
```
