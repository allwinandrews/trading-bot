# trading-bot

A systematic trading project: backtest a strategy on historical data, paper-trade it against live
data, and (only once proven) run it live through a broker API — with stop-loss, position sizing,
and a daily loss circuit breaker built in from the start. Includes an AI-written trade journal that
explains why each trade fired.

**Status:** backtester, risk layer, AI trade journal, and dashboard all working. Paper trading
(execution/) is wired up — validating it against a live IBKR paper account next, before any
real money is risked.

## Project structure

```
data/               downloaded historical price data (gitignored) + loader.py, news.py (yfinance headlines)
strategies/         strategy logic (signal generation, no execution) + trendiness.py (which tickers fit it)
backtester/         engine.py + portfolio_engine.py (simulation), metrics.py (scoring), runner.py (orchestration)
risk/               stop-loss, position sizing, daily loss circuit breaker
execution/          IBKR broker wrapper (broker.py) + live signal loop (paper_trader.py) + trade_log.py
ai/                 trade journal / sentiment layer (Gemini API)
notebooks/          exploration and result plots
tests/              pytest suite — synthetic data or mocked network, no real network dependency
run_backtest.py            CLI: backtest one ticker
run_robustness_check.py    CLI: backtest across many tickers/periods (repeated, not shared capital)
run_portfolio_backtest.py  CLI: backtest many tickers with ONE shared capital pool (real diversification)
run_trend_screen.py        CLI: rank tickers by how well-suited they are to this strategy
run_paper_trader.py        CLI: paper/live trading via IBKR (one-shot or --loop)
dashboard.py         Streamlit dashboard (Backtest, Live Account, Trade History, Portfolio Backtest, News pages)
```

## Build order

1. ✅ Backtester + a simple strategy (moving-average crossover) on free historical data, with
   stop-loss and fixed-fractional position sizing baked in — no API keys needed.
2. ✅ Comparison against buy-and-hold (built into `run_backtest.py`'s output).
3. ✅ AI trade journal on top of backtest results (`--ai-journal`, needs `GEMINI_API_KEY`).
4. ✅ Streamlit dashboard (`streamlit run dashboard.py`) — interactive version of the same backtest.
5. ✅ Paper trading against live data (Interactive Brokers) — code is wired up (`execution/`),
   **not yet validated against a live paper account**.
6. Live trading, small size, only after sustained paper-trading validation.

## Running a backtest

```bash
.venv\Scripts\activate
python run_backtest.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
python run_backtest.py --ticker RELIANCE.NS --short-window 10 --long-window 30 --stop-loss-pct 0.03
python run_backtest.py --ticker AAPL --ai-journal   # + a Gemini-written review of the trade log
```

Prints a strategy-vs-buy-and-hold metrics comparison and the most recent trades, and saves an
equity curve chart to `results/` (gitignored — regenerate anytime). `--ai-journal` additionally
asks Gemini to review the trade log in plain English and saves it to `results/<ticker>_trade_journal.md`
— needs `GEMINI_API_KEY` set in `.env` (get a free one at https://aistudio.google.com/app/apikey);
without it, the flag just prints a skip notice rather than failing the run.

## Does it actually have an edge?

```bash
python run_robustness_check.py
python run_robustness_check.py --short-window 50 --long-window 200 --start 2021-06-01 --end 2023-01-01
```

A strategy that looks good on one ticker over one time window hasn't proven anything. This runs
the same backtest across a diverse basket of tickers (AAPL, MSFT, JNJ, XOM, SPY by default) and
reports how often it actually beats buy-and-hold, plus drawdown comparison — not just a single
favorable number. Current honest finding: this strategy **loses to buy-and-hold on raw return in
strong bull periods**, but **consistently has a smaller max drawdown** (real capital protection
during downturns) and can beat buy-and-hold outright during bear/choppy periods. It's a
risk-reduction tool, not (yet, at these settings) a return-maximizing one — know which one you're
actually running before trusting it with real money.

**Diversification, tested for real** (`run_portfolio_backtest.py` / `backtester/portfolio_engine.py`,
not N independent single-ticker runs — one shared capital pool): spreading the same money across
multiple tickers at once beats concentrating it in one, on every risk metric (smaller drawdown,
better Sharpe) — but still doesn't fully close the gap to an equally-diversified buy-and-hold.

**Not every ticker fits this strategy, and now there's a way to check before trading one**
(`run_trend_screen.py` / `strategies/trendiness.py`, Kaufman's Efficiency Ratio): ranks tickers by
how cleanly they trend vs. how much they chop back and forth. Filtering out low-scoring tickers
(`run_portfolio_backtest.py --min-efficiency-ratio 0.05`) measurably improved the 2015-2024 basket
— Sharpe 0.75 → 0.87, win rate 13.7% → 23.8%, avg trade return 4.35% → 14.36%, fewer/higher-quality
trades (153 → 42). Chosen for cause (XOM's trades were visibly whipsawed), not fitted after the fact.

**A market regime filter closes most of the remaining Sharpe gap — at a real cost, not a free win**
(`--regime-filter` / `strategies/regime.py`): only takes/holds trades while SPY is itself above its
own 200-day average, sitting out (and exiting) during broad-market chop. On the trendiness-filtered
basket: Sharpe 0.87 → **0.99** (buy-and-hold: 1.0 — the closest match all session) and drawdown
-12.83% → **-9.31%**, but total return drops (86.64% → 65.76%), average trade quality drops
(14.36% → 5.85%), and trade count roughly doubles (42 → 85) from whipsawing around the regime
boundary. A genuine risk-adjusted improvement, not a strictly-better setting — know which tradeoff
you actually want before turning it on.

## Running the paper trader

```bash
python run_paper_trader.py --ticker AAPL
python run_paper_trader.py --ticker AAPL --loop --interval-seconds 300   # autopilot mode
```

Connects to TWS/IB Gateway (paper or live — only `IB_PORT` in `.env` differs, see
`.env.example` for the four port numbers), checks today's signal for the ticker, and
enters/exits/holds accordingly. Requires TWS or IB Gateway to be running and logged in.

By default runs once and exits — schedule it yourself (cron, Task Scheduler) for once-per-day
checks. `--loop` keeps it running continuously ("autopilot") until Ctrl+C, re-checking every
`--interval-seconds`; since this is a daily-bar strategy the trading *signal* only meaningfully
changes about once a day, so looping more often mainly keeps the dashboard's Live Account and
Trade History pages fresh, not "trades more often." Start with the paper account; only point it
at a live-port connection after sustained paper-trading validation.

**If autopilot is already running and you want to run a one-off manual check at the same time**,
pass `--client-id 3` (or any ID other than 1 and 2) — IBKR rejects/times out two simultaneous
connections sharing the same ID. Default CLI usage is client_id=1, the dashboard's Live Account
page uses client_id=2; pick a different one for anything else you run alongside them.

## Running the dashboard

```bash
streamlit run dashboard.py
```

Same backtest engine as the CLI, with sliders for strategy/risk parameters, an interactive
equity curve, a downloadable trade log, and an optional AI trade journal checkbox. Also includes
a **Live Account** page (sidebar nav) showing your connected TWS/IB Gateway account's equity,
positions, and open orders — click Refresh to connect and pull the current state.

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
- Default MA windows are **50/200** (not the more common tutorial default of 20/50). Chosen from
  evidence, not guesswork: `run_robustness_check.py` across AAPL/MSFT/JNJ/XOM/SPY (2015-2024)
  showed 50/200 roughly triples average return and improves Sharpe vs 20/50, with fewer trades
  (less commission drag) and the same strong drawdown protection. Neither setting beats plain
  buy-and-hold in a strong bull period — see "does it actually have an edge" below.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then fill in real keys — never commit .env
```
