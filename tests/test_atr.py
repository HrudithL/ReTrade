"""Tests for ATR calculation."""

import pandas as pd
import pytest

from src.data.fetch import resample_to_daily
from src.indicators.atr import atr, daily_atr_from_daily_bars, get_daily_atr_for_date, true_range
from src.utils.timezones import to_ny


def test_true_range():
    """Test True Range calculation."""
    df = pd.DataFrame(
        {
            "high": [100.5, 101.0, 101.5],
            "low": [99.5, 100.0, 100.5],
            "close": [100.0, 100.5, 101.0],
        }
    )

    tr = true_range(df)
    assert len(tr) == 3
    assert tr.iloc[0] == 1.0  # high - low for first bar
    assert tr.iloc[1] >= 0.5  # Should consider prev close


def test_atr():
    """Test ATR calculation."""
    # Create data with known volatility
    df = pd.DataFrame(
        {
            "high": [100.5, 101.0, 101.5, 102.0, 102.5] * 5,
            "low": [99.5, 100.0, 100.5, 101.0, 101.5] * 5,
            "close": [100.0, 100.5, 101.0, 101.5, 102.0] * 5,
        }
    )

    atr_values = atr(df, period=14)
    assert len(atr_values) == len(df)
    assert atr_values.iloc[-1] > 0  # Should have positive ATR
    assert atr_values.iloc[0] == atr_values.iloc[0]  # First value should be valid


def test_daily_atr_from_daily_bars():
    """Test ATR calculation from daily bars."""
    # Create daily bars
    dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
    df_daily = pd.DataFrame(
        {
            "time_ny": dates,
            "high": [100.5 + i * 0.1 for i in range(30)],
            "low": [99.5 + i * 0.1 for i in range(30)],
            "close": [100.0 + i * 0.1 for i in range(30)],
        }
    )

    atr_series = daily_atr_from_daily_bars(df_daily, period=20)
    assert len(atr_series) == len(df_daily)
    assert atr_series.iloc[-1] > 0


def test_resample_to_daily():
    """Test resampling intraday to daily."""
    # Create intraday data
    times = pd.date_range("2024-01-15 09:30:00", periods=100, freq="5min", tz="America/New_York")
    df_intraday = pd.DataFrame(
        {
            "time_ny": times,
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [1000] * 100,
        }
    )

    df_daily = resample_to_daily(df_intraday, time_col="time_ny")
    assert len(df_daily) > 0
    assert "time_ny" in df_daily.columns
    assert "open" in df_daily.columns
    assert "high" in df_daily.columns
    assert "low" in df_daily.columns
    assert "close" in df_daily.columns


def test_get_daily_atr_for_date():
    """Test getting ATR for a specific date."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
    df_daily = pd.DataFrame(
        {
            "time_ny": dates,
            "high": [100.5 + i * 0.1 for i in range(30)],
            "low": [99.5 + i * 0.1 for i in range(30)],
            "close": [100.0 + i * 0.1 for i in range(30)],
        }
    )

    target_date = pd.Timestamp("2024-01-25", tz="America/New_York")
    today_atr, threshold = get_daily_atr_for_date(df_daily, target_date, period=20)

    assert today_atr > 0
    assert threshold > 0

