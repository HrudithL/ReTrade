"""Backtesting engine for Opening Range Breakout strategy."""

from datetime import date
from typing import Dict, List, Literal, Optional

import pandas as pd

from src.backtest.metrics import compute_metrics
from src.backtest.types import BacktestResult, Trade
from src.config.logging import get_logger
from src.config.settings import Settings
from src.data.fetch import resample_to_daily
from src.execution.costs import apply_costs, get_cost_config
from src.indicators.atr import get_daily_atr_for_date
from src.risk.position_sizing import enforce_daily_risk_limit, get_pip_value, position_size_from_risk
from src.strategy.filters import atr_filter
from src.strategy.opening_range import (
    bias,
    build_opening_range,
    entry_signals,
    profit_target,
    stop_price,
)
from src.strategy.params import StrategyParams
from src.utils.calendar import build_daily_groups, get_or_window, is_valid_trading_day

logger = get_logger(__name__)


class BacktestEngine:
    """Backtesting engine for ORB strategy."""

    def __init__(
        self,
        params: StrategyParams,
        instrument: str,
        starting_equity: float = 100000.0,
    ):
        """Initialize backtest engine.

        Args:
            params: StrategyParams instance
            instrument: Instrument identifier (e.g., "US500_USD")
            starting_equity: Starting equity in currency (default: 100000.0)
        """
        self.params = params
        self.instrument = instrument
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.current_equity_R = 0.0  # Cumulative R

        # Get cost config
        self.cost_config = get_cost_config(instrument, params.slippage_model)
        self.pip_value = get_pip_value(instrument)

    def run(
        self,
        df_all: pd.DataFrame,
        df_daily_atr: Optional[pd.DataFrame] = None,
        df_intraday_fallback: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            df_all: Full DataFrame with main timeframe data (time_ny column required)
            df_daily_atr: Optional DataFrame with daily candles for ATR calculation
            df_intraday_fallback: Optional intraday data to resample if daily unavailable
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        if "time_ny" not in df_all.columns:
            raise ValueError("DataFrame must have 'time_ny' column")

        # Filter by date range if provided
        if start_date or end_date:
            df_all = df_all.copy()
            if start_date:
                start_ts = pd.Timestamp(start_date, tz=df_all["time_ny"].dt.tz)
                df_all = df_all[df_all["time_ny"] >= start_ts]
            if end_date:
                end_ts = pd.Timestamp(end_date, tz=df_all["time_ny"].dt.tz) + pd.Timedelta(days=1)
                df_all = df_all[df_all["time_ny"] < end_ts]

        # Prepare daily ATR data
        df_daily_for_atr = None
        if df_daily_atr is not None and len(df_daily_atr) > 0:
            # Use provided daily candles
            df_daily_for_atr = df_daily_atr.copy()
            if "time_ny" not in df_daily_for_atr.columns:
                raise ValueError("Daily ATR DataFrame must have 'time_ny' column")
        elif df_intraday_fallback is not None and len(df_intraday_fallback) > 0:
            # Fallback: resample intraday to daily
            logger.info("Resampling intraday data to daily for ATR calculation...")
            df_daily_for_atr = resample_to_daily(df_intraday_fallback, time_col="time_ny")

        # Build daily groups
        daily_groups = build_daily_groups(df_all)

        # Reset equity for this run
        self.current_equity = self.starting_equity
        self.current_equity_R = 0.0

        trades: List[Trade] = []
        skipped_days: List[Dict[str, str]] = []
        equity_curve_data: List[Dict[str, float]] = []

        # Track daily risk
        current_day: Optional[date] = None
        daily_risk_amounts: List[float] = []

        for session_date, df_day in sorted(daily_groups.items()):
            # Reset daily risk tracking if new day
            if current_day != session_date:
                current_day = session_date
                daily_risk_amounts = []

            # Check if valid trading day
            if not is_valid_trading_day(df_day):
                continue

            # Get OR window
            df_or = get_or_window(df_day, or_start=self.params.or_start_time, or_end=self.params.or_end_time)
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
                        period=self.params.atr_period,
                        date_col="time_ny",
                    )

                    if threshold_atr == 0.0 or today_atr == 0.0:
                        # Insufficient ATR data, skip day
                        skipped_days.append(
                            {
                                "date": str(session_date),
                                "reason": "Insufficient ATR data",
                            }
                        )
                        logger.debug(f"Skipping {session_date}: Insufficient ATR data")
                        continue

                    if not atr_filter(today_atr, threshold_atr, self.params.atr_mult):
                        # ATR filter failed
                        logger.debug(f"Skipping {session_date}: ATR filter failed")
                        continue
                except Exception as e:
                    # Error in ATR calculation, skip day
                    skipped_days.append(
                        {
                            "date": str(session_date),
                            "reason": f"ATR calculation error: {e}",
                        }
                    )
                    logger.warning(f"ATR calculation error for {session_date}: {e}")
                    continue

            # Get bars after OR window
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
                self.params.use_mid_stop,
            )

            entry_price_base = entry_signal["entry_price"]
            tp = profit_target(entry_price_base, stop, bias_direction, self.params.rr)

            # Calculate position size based on risk
            position_size = position_size_from_risk(
                equity=self.current_equity,
                risk_per_trade=self.params.risk_per_trade,
                entry_price=entry_price_base,
                stop_price=stop,
                pip_value=self.pip_value,
            )

            if position_size == 0:
                logger.debug(f"Skipping trade on {session_date}: Position size is zero")
                continue

            # Calculate risk amount for this trade
            risk_amount = abs(entry_price_base - stop) * position_size

            # Check daily risk limit
            if not enforce_daily_risk_limit(daily_risk_amounts, self.current_equity, self.params.max_daily_risk):
                logger.debug(f"Skipping trade on {session_date}: Daily risk limit would be exceeded")
                continue

            # Apply costs to get effective entry price
            # For entry, we use the base entry price and apply costs
            effective_entry, _, _ = apply_costs(entry_price_base, entry_price_base, bias_direction, self.cost_config)

            # Find entry bar index
            entry_index = entry_signal["entry_index"]
            entry_bar = df_after.loc[entry_index]

            # Simulate trade from entry bar onwards
            df_sim = df_after[df_after.index >= entry_index].copy()

            exit_time: Optional[pd.Timestamp] = None
            exit_price_base: Optional[float] = None
            exit_reason: Optional[str] = None
            gross_R = 0.0

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
                    if self.params.tp_sl_conflict_policy == "tp_first":
                        exit_price_base = tp
                        exit_reason = "TP"
                        gross_R = self.params.rr
                    else:  # sl_first
                        exit_price_base = stop
                        exit_reason = "SL"
                        gross_R = -1.0
                    exit_time = bar["time_ny"]
                    break

                elif stop_hit:
                    exit_price_base = stop
                    exit_reason = "SL"
                    gross_R = -1.0
                    exit_time = bar["time_ny"]
                    break

                elif tp_hit:
                    exit_price_base = tp
                    exit_reason = "TP"
                    gross_R = self.params.rr
                    exit_time = bar["time_ny"]
                    break

            # If no exit found, exit at end of day
            if exit_time is None or exit_price_base is None:
                last_bar = df_sim.iloc[-1]
                exit_time = last_bar["time_ny"]
                exit_price_base = last_bar["close"]
                exit_reason = "EOD"
                # Calculate gross R for EOD exit
                risk = abs(effective_entry - stop)
                if risk > 0:
                    pnl = (exit_price_base - effective_entry) if bias_direction == "long" else (effective_entry - exit_price_base)
                    gross_R = pnl / risk
                else:
                    gross_R = 0.0

            # Safety check
            if exit_price_base is None:
                raise ValueError("exit_price_base is None - this should not happen")

            # Apply costs to exit
            exit_side: Literal["long", "short"] = "short" if bias_direction == "long" else "long"
            _, effective_exit, total_cost = apply_costs(entry_price_base, exit_price_base, exit_side, self.cost_config)

            # Calculate net R
            risk = abs(effective_entry - stop)
            if risk > 0:
                # Gross PnL in price units
                if bias_direction == "long":
                    gross_pnl_price = exit_price_base - entry_price_base
                else:
                    gross_pnl_price = entry_price_base - exit_price_base

                # Net PnL after costs
                net_pnl_price = gross_pnl_price - total_cost

                # Net R
                net_R = net_pnl_price / risk
            else:
                net_R = 0.0
                net_pnl_price = 0.0

            # Update equity
            pnl_currency = net_pnl_price * position_size
            self.current_equity += pnl_currency
            self.current_equity_R += net_R

            # Record daily risk
            daily_risk_amounts.append(risk_amount)

            # Create trade object
            trade = Trade(
                date=session_date,
                instrument=self.instrument,
                entry_time=entry_signal["entry_time"],
                entry_price=effective_entry,
                side=bias_direction,
                stop=stop,
                tp=tp,
                exit_time=exit_time,
                exit_price=effective_exit,
                R=net_R,
                reason=exit_reason or "UNKNOWN",
                position_size=position_size,
                gross_R=gross_R,
                net_R=net_R,
            )
            trades.append(trade)

            # Update equity curve
            equity_curve_data.append(
                {
                    "date": session_date,
                    "equity_currency": self.current_equity,
                    "equity_R": self.current_equity_R,
                }
            )

        # Log skipped days if any
        if skipped_days:
            logger.info(f"Skipped {len(skipped_days)} days due to ATR data issues")
            for skip in skipped_days[:5]:  # Show first 5
                logger.debug(f"  {skip['date']}: {skip['reason']}")
            if len(skipped_days) > 5:
                logger.debug(f"  ... and {len(skipped_days) - 5} more")

        # Create equity curve DataFrame
        equity_curve = None
        if equity_curve_data:
            equity_curve = pd.DataFrame(equity_curve_data)
            equity_curve["date"] = pd.to_datetime(equity_curve["date"])

        # Compute metrics
        if trades:
            trades_df = pd.DataFrame([trade.to_dict() for trade in trades])
            metrics = compute_metrics(trades_df)
        else:
            metrics = compute_metrics(pd.DataFrame())

        # Create result
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            params=self.params,
        )

        return result
