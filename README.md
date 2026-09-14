# trading-bot

A systematic trading project: backtest a strategy on historical data, paper-trade it against live
data, and (only once proven) run it live through a broker API — with stop-loss, position sizing,
and a daily loss circuit breaker built in from the start. Includes an AI-written trade journal that
explains why each trade fired.

**Status:** early scaffold — building the backtester first.

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

1. Backtester + a simple strategy (moving-average crossover) on free historical data — no
   API keys needed.
2. Comparison against buy-and-hold.
3. Risk layer: stop-loss, position sizing, circuit breaker — tunable in the backtest.
4. AI trade journal on top of backtest results.
5. Paper trading against live data (Interactive Brokers).
6. Streamlit dashboard.
7. Live trading, small size, only after sustained paper-trading validation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then fill in real keys — never commit .env
```
