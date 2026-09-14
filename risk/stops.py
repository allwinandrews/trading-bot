"""Stop-loss and portfolio-level circuit breaker logic.

Kept as pure functions with no dependency on the backtester or broker, so the
exact same logic runs in backtest, paper, and live modes later — one
implementation, not three copies to keep in sync.
"""
from __future__ import annotations


def fixed_stop_price(entry_price: float, stop_loss_pct: float) -> float:
    """Stop price for a long position, `stop_loss_pct` below entry (e.g. 0.02 = 2%)."""
    if not (0 < stop_loss_pct < 1):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    return entry_price * (1 - stop_loss_pct)


def trailing_stop_price(highest_price_since_entry: float, stop_loss_pct: float) -> float:
    """Trailing stop for a long position: stop_loss_pct below the highest price seen so far."""
    if not (0 < stop_loss_pct < 1):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    return highest_price_since_entry * (1 - stop_loss_pct)


class DailyLossCircuitBreaker:
    """Halts new trades for the day once cumulative loss exceeds a threshold.

    Used by the backtester and (later) the live/paper execution loop alike —
    a placeholder for "the bot stops trading itself when it's having a bad day."
    """

    def __init__(self, max_daily_loss_pct: float = 0.03):
        if not (0 < max_daily_loss_pct <= 1):
            raise ValueError("max_daily_loss_pct must be between 0 and 1")
        self.max_daily_loss_pct = max_daily_loss_pct
        self._day_start_equity: float | None = None
        self._tripped = False

    def start_day(self, equity: float) -> None:
        self._day_start_equity = equity
        self._tripped = False

    def update(self, equity: float) -> bool:
        """Call once per bar/tick with current equity. Returns True if trading should halt."""
        if self._day_start_equity is None:
            self.start_day(equity)
            return False

        drawdown_pct = (self._day_start_equity - equity) / self._day_start_equity
        if drawdown_pct >= self.max_daily_loss_pct:
            self._tripped = True
        return self._tripped

    @property
    def tripped(self) -> bool:
        return self._tripped
