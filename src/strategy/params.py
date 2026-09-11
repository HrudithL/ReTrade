"""Strategy parameters dataclass for Opening Range Breakout strategy."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StrategyParams:
    """Parameters for Opening Range Breakout strategy.

    Attributes:
        or_start_time: Opening range start time in HH:MM format (default: "09:30")
        or_end_time: Opening range end time in HH:MM format (default: "09:35")
        rr: Risk-reward ratio (default: 2.0)
        atr_period: ATR period for rolling mean calculation (default: 20)
        atr_mult: ATR multiplier threshold (default: 1.2)
        use_mid_stop: If True, use OR mid as stop; else use opposite boundary (default: False)
        tp_sl_conflict_policy: Policy when both TP and SL hit same bar: "sl_first" or "tp_first" (default: "sl_first")
        risk_per_trade: Risk per trade as fraction of equity (default: 0.0025 = 0.25%)
        max_daily_risk: Maximum daily risk as fraction of equity (default: 0.01 = 1%)
        slippage_model: Slippage model identifier (default: "oanda_default")
        fee_model: Fee model identifier (default: "oanda_default")
    """

    or_start_time: str = "09:30"
    or_end_time: str = "09:35"
    rr: float = 2.0
    atr_period: int = 20
    atr_mult: float = 1.2
    use_mid_stop: bool = False
    tp_sl_conflict_policy: Literal["sl_first", "tp_first"] = "sl_first"
    risk_per_trade: float = 0.0025
    max_daily_risk: float = 0.01
    slippage_model: str = "oanda_default"
    fee_model: str = "oanda_default"

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all parameters."""
        # Validate time format
        try:
            hour, minute = self.or_start_time.split(":")
            hour_int = int(hour)
            minute_int = int(minute)
            if not (0 <= hour_int < 24 and 0 <= minute_int < 60):
                raise ValueError(f"Invalid time format: {self.or_start_time}")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"or_start_time must be in HH:MM format, got {self.or_start_time}") from e

        try:
            hour, minute = self.or_end_time.split(":")
            hour_int = int(hour)
            minute_int = int(minute)
            if not (0 <= hour_int < 24 and 0 <= minute_int < 60):
                raise ValueError(f"Invalid time format: {self.or_end_time}")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"or_end_time must be in HH:MM format, got {self.or_end_time}") from e

        # Validate numeric parameters
        if not isinstance(self.rr, (int, float)) or self.rr <= 0:
            raise ValueError(f"rr must be a positive number, got {self.rr}")

        if not isinstance(self.atr_period, int) or self.atr_period <= 0:
            raise ValueError(f"atr_period must be a positive integer, got {self.atr_period}")

        if not isinstance(self.atr_mult, (int, float)) or self.atr_mult <= 0:
            raise ValueError(f"atr_mult must be a positive number, got {self.atr_mult}")

        if not isinstance(self.risk_per_trade, (int, float)) or self.risk_per_trade <= 0:
            raise ValueError(f"risk_per_trade must be a positive number, got {self.risk_per_trade}")

        if not isinstance(self.max_daily_risk, (int, float)) or self.max_daily_risk <= 0:
            raise ValueError(f"max_daily_risk must be a positive number, got {self.max_daily_risk}")

        if self.risk_per_trade > self.max_daily_risk:
            raise ValueError(
                f"risk_per_trade ({self.risk_per_trade}) cannot exceed max_daily_risk ({self.max_daily_risk})"
            )

        if self.tp_sl_conflict_policy not in ("sl_first", "tp_first"):
            raise ValueError(
                f"tp_sl_conflict_policy must be 'sl_first' or 'tp_first', got {self.tp_sl_conflict_policy}"
            )

