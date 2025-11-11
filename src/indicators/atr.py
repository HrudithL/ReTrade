"""Average True Range (ATR) indicator calculation."""

from typing import Tuple

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    """Calculate True Range.

    Args:
        df: DataFrame with OHLC data
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series of True Range values
    """
    high = df[high_col]
    low = df[low_col]
    prev_close = df[close_col].shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series of ATR values
    """
    tr = true_range(df, high_col, low_col, close_col)
    atr_values = tr.rolling(window=period, min_periods=1).mean()
    return atr_values


def daily_atr_from_daily_bars(
    df_daily: pd.DataFrame,
    period: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Calculate ATR from daily bars.

    This is the primary method for ATR calculation in the backtester.
    Daily ATR is computed directly from daily OHLC candles.

    Args:
        df_daily: DataFrame with daily OHLC bars
        period: ATR period (default 20)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series of daily ATR values
    """
    return atr(df_daily, period=period, high_col=high_col, low_col=low_col, close_col=close_col)


def get_daily_atr_for_date(
    df_daily: pd.DataFrame,
    target_date: pd.Timestamp,
    period: int = 20,
    date_col: str = "time_ny",
) -> Tuple[float, float]:
    """Get today's daily ATR and 20-day average ATR threshold for a specific date.

    Args:
        df_daily: DataFrame with daily bars (must have date_col)
        target_date: Target date to get ATR for
        period: ATR period for threshold calculation (default 20)
        date_col: Name of date column

    Returns:
        Tuple of (today_atr, threshold_atr)
        Returns (0.0, 0.0) if insufficient data
    """
    if date_col not in df_daily.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    # Filter data up to and including target date
    if isinstance(target_date, pd.Timestamp):
        target_date_only = target_date.date()
    else:
        target_date_only = target_date

    historical = df_daily[df_daily[date_col].dt.date <= target_date_only].copy()

    if len(historical) < period:
        # Insufficient data
        return (0.0, 0.0)

    # Calculate ATR series
    atr_series = daily_atr_from_daily_bars(historical, period=period)

    # Get threshold (average of last 'period' ATR values)
    if len(atr_series) >= period:
        threshold = atr_series.tail(period).mean()
    else:
        threshold = atr_series.mean()

    # Get today's ATR
    today_data = historical[historical[date_col].dt.date == target_date_only]
    if len(today_data) == 0:
        return (0.0, 0.0)

    # Calculate ATR up to today
    today_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.0

    return (today_atr, threshold)

