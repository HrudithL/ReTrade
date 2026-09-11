"""Type definitions for backtest results."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import pandas as pd

from src.strategy.params import StrategyParams


@dataclass
class Trade:
    """Represents a single trade in the backtest.

    Attributes:
        date: Trading date
        instrument: Instrument identifier
        entry_time: Entry timestamp
        entry_price: Entry price (effective after costs)
        side: Trade side ("long" or "short")
        stop: Stop loss price
        tp: Take profit price
        exit_time: Exit timestamp
        exit_price: Exit price (effective after costs)
        R: Net R multiple (after costs)
        reason: Exit reason ("TP", "SL", "EOD")
        position_size: Position size in units
        gross_R: Gross R multiple (before costs)
        net_R: Net R multiple (after costs, same as R)
    """

    date: date
    instrument: str
    entry_time: pd.Timestamp
    entry_price: float
    side: Literal["long", "short"]
    stop: float
    tp: float
    exit_time: pd.Timestamp
    exit_price: float
    R: float
    reason: str
    position_size: float = 0.0
    gross_R: float = 0.0
    net_R: float = 0.0

    def to_dict(self) -> dict:
        """Convert trade to dictionary for DataFrame creation."""
        return {
            "date": self.date,
            "instrument": self.instrument,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "side": self.side,
            "stop": self.stop,
            "tp": self.tp,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "R": self.R,
            "reason": self.reason,
            "position_size": self.position_size,
            "gross_R": self.gross_R,
            "net_R": self.net_R,
        }


@dataclass
class BacktestResult:
    """Complete backtest result.

    Attributes:
        trades: List of Trade objects
        equity_curve: DataFrame with columns: date, equity_currency, equity_R
        metrics: Dictionary of computed metrics
        params: StrategyParams used for the backtest
    """

    trades: list[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.DataFrame] = None
    metrics: dict = field(default_factory=dict)
    params: Optional[StrategyParams] = None

    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "date",
                    "instrument",
                    "entry_time",
                    "entry_price",
                    "side",
                    "stop",
                    "tp",
                    "exit_time",
                    "exit_price",
                    "R",
                    "reason",
                    "position_size",
                    "gross_R",
                    "net_R",
                ]
            )
        return pd.DataFrame([trade.to_dict() for trade in self.trades])

