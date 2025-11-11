"""Backtesting engine for Opening Range Breakout strategy."""

from datetime import date
from typing import Dict, Literal, Optional

import pandas as pd

from src.backtest.metrics import compute_metrics
from src.config.settings import Settings
from src.data.fetch import resample_to_daily
from src.indicators.atr import get_daily_atr_for_date
from src.strategy.filters import atr_filter
from src.strategy.opening_range import (
    bias,
    build_opening_range,
    entry_signals,
    profit_target,
    stop_price,
)
from src.utils.calendar import build_daily_groups, get_or_window, is_valid_trading_day
from src.utils.math import apply_commission, apply_slippage


class BacktestEngine:
    """Backtesting engine for ORB strategy."""

    def __init__(self, params: Dict):
        """Initialize backtest engine.

        Args:
            params: Backtest parameters including:
                - instrument: str
                - granularity: str
                - use_mid_stop: bool
                - rr: float
                - atr_tf: str
                - atr_mult: float
                - atr_period: int (default 20)
                - fill_policy: 'tp_first' or 'sl_first' (default 'sl_first')
        """
        self.params = params
        self.instrument = params["instrument"]
        self.use_mid_stop = params.get("use_mid_stop", False)
        self.rr = params.get("rr", 2.0)
        self.atr_mult = params.get("atr_mult", 1.2)
        self.atr_period = params.get("atr_period", 20)
        self.fill_policy = params.get("fill_policy", "sl_first")  # 'tp_first' or 'sl_first'

    def run(
        self,
        df_all: pd.DataFrame,
        df_daily_atr: Optional[pd.DataFrame] = None,
        df_intraday_fallback: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run backtest.

        Args:
            df_all: Full DataFrame with main timeframe data (time_ny column required)
            df_daily_atr: Optional DataFrame with daily candles for ATR calculation
            df_intraday_fallback: Optional intraday data to resample if daily unavailable

        Returns:
            TradeLog DataFrame with columns:
            date, instrument, entry_time, entry_price, side, stop, tp, exit_time, exit_price, R, reason
        """
        if "time_ny" not in df_all.columns:
            raise ValueError("DataFrame must have 'time_ny' column")

        # Prepare daily ATR data
        df_daily_for_atr = None
        if df_daily_atr is not None and len(df_daily_atr) > 0:
            # Use provided daily candles
            df_daily_for_atr = df_daily_atr.copy()
            if "time_ny" not in df_daily_for_atr.columns:
                raise ValueError("Daily ATR DataFrame must have 'time_ny' column")
        elif df_intraday_fallback is not None and len(df_intraday_fallback) > 0:
            # Fallback: resample intraday to daily
            print("Resampling intraday data to daily for ATR calculation...")
            df_daily_for_atr = resample_to_daily(df_intraday_fallback, time_col="time_ny")

        # Build daily groups
        daily_groups = build_daily_groups(df_all)

        trades = []
        skipped_days = []

        for session_date, df_day in sorted(daily_groups.items()):
            # Check if valid trading day
            if not is_valid_trading_day(df_day):
                continue

            # Get OR window
            df_or = get_or_window(df_day, or_start="09:30", or_end="09:35")
            if len(df_or) == 0:
                continue

            # Build OR
            try:
                or_data = build_opening_range(df_or)
            except ValueError:
                continue

            # Determine bias
            bias_direction = bias(or_data["or_close"], or_data["or_mid"])
            if bias_direction == "none":
                continue

            # ATR filter
            if df_daily_for_atr is not None and len(df_daily_for_atr) > 0:
                try:
                    # Get date as timestamp for ATR lookup
                    date_ts = pd.Timestamp(session_date)
                    today_atr, threshold_atr = get_daily_atr_for_date(
                        df_daily_for_atr,
                        date_ts,
                        period=self.atr_period,
                        date_col="time_ny",
                    )

                    if threshold_atr == 0.0 or today_atr == 0.0:
                        # Insufficient ATR data, skip day
                        skipped_days.append(
                            {
                                "date": session_date,
                                "reason": "Insufficient ATR data",
                            }
                        )
                        continue

                    if not atr_filter(today_atr, threshold_atr, self.atr_mult):
                        # ATR filter failed
                        continue
                except Exception as e:
                    # Error in ATR calculation, skip day
                    skipped_days.append(
                        {
                            "date": session_date,
                            "reason": f"ATR calculation error: {e}",
                        }
                    )
                    continue

            # Get bars after 9:35
            df_after = df_day[df_day["time_ny"] > or_data["or_end"]].copy()
            if len(df_after) == 0:
                continue

            # Find entry signal
            entry_signal = entry_signals(
                df_after,
                or_data["or_high"],
                or_data["or_low"],
                bias_direction,
            )

            if entry_signal is None:
                continue

            # Calculate stop and TP
            stop = stop_price(
                bias_direction,
                or_data["or_high"],
                or_data["or_low"],
                or_data["or_mid"],
                self.use_mid_stop,
            )

            entry_price = entry_signal["entry_price"]
            tp = profit_target(entry_price, stop, bias_direction, self.rr)

            # Apply slippage
            slippage_bps = Settings.get_slippage_bps(self.instrument)
            entry_fill = apply_slippage(entry_price, bias_direction, slippage_bps)

            # Find entry bar index
            entry_index = entry_signal["entry_index"]
            entry_bar = df_after.loc[entry_index]

            # Simulate trade from entry bar onwards
            df_sim = df_after[df_after.index >= entry_index].copy()

            exit_time = None
            exit_price = None
            exit_reason = None
            R = 0.0

            for idx, bar in df_sim.iterrows():
                # Check for stop loss
                if bias_direction == "long":
                    stop_hit = bar["low"] <= stop
                else:
                    stop_hit = bar["high"] >= stop

                # Check for take profit
                if bias_direction == "long":
                    tp_hit = bar["high"] >= tp
                else:
                    tp_hit = bar["low"] <= tp

                # Handle both TP and SL on same bar
                if stop_hit and tp_hit:
                    if self.fill_policy == "tp_first":
                        exit_price = tp
                        exit_reason = "TP"
                        R = self.rr
                    else:  # sl_first
                        exit_price = stop
                        exit_reason = "SL"
                        R = -1.0
                    exit_time = bar["time_ny"]
                    break

                elif stop_hit:
                    exit_price = stop
                    exit_reason = "SL"
                    R = -1.0
                    exit_time = bar["time_ny"]
                    break

                elif tp_hit:
                    exit_price = tp
                    exit_reason = "TP"
                    R = self.rr
                    exit_time = bar["time_ny"]
                    break

            # If no exit found, exit at end of day
            if exit_time is None:
                last_bar = df_sim.iloc[-1]
                exit_time = last_bar["time_ny"]
                if bias_direction == "long":
                    exit_price = last_bar["close"]
                else:
                    exit_price = last_bar["close"]
                exit_reason = "EOD"
                # Calculate R for EOD exit
                risk = abs(entry_fill - stop)
                if risk > 0:
                    pnl = (exit_price - entry_fill) if bias_direction == "long" else (entry_fill - exit_price)
                    R = pnl / risk
                else:
                    R = 0.0

            # Apply commission
            commission_bps = Settings.get_commission_bps(self.instrument)
            commission = apply_commission(entry_fill, bias_direction, commission_bps)
            # Adjust R for commission (small impact)
            if abs(entry_fill - stop) > 0:
                R -= commission / abs(entry_fill - stop)

            trades.append(
                {
                    "date": session_date,
                    "instrument": self.instrument,
                    "entry_time": entry_signal["entry_time"],
                    "entry_price": entry_fill,
                    "side": bias_direction,
                    "stop": stop,
                    "tp": tp,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "R": R,
                    "reason": exit_reason,
                }
            )

        # Log skipped days if any
        if skipped_days:
            print(f"\nSkipped {len(skipped_days)} days due to ATR data issues:")
            for skip in skipped_days[:5]:  # Show first 5
                print(f"  {skip['date']}: {skip['reason']}")
            if len(skipped_days) > 5:
                print(f"  ... and {len(skipped_days) - 5} more")

        if not trades:
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
                ]
            )

        trades_df = pd.DataFrame(trades)
        return trades_df

