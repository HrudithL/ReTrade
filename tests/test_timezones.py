"""Tests for timezone utilities."""

from datetime import date

import pandas as pd
import pytest

from src.utils.timezones import get_ny_time, is_between, session_date, to_ny


def test_to_ny():
    """Test UTC to NY timezone conversion."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-15 14:30:00", periods=5, freq="5min", tz="UTC"),
            "close": [100, 101, 102, 103, 104],
        }
    )

    df_ny = to_ny(df)
    assert "time_ny" in df_ny.columns
    assert df_ny["time_ny"].dt.tz.zone == "America/New_York"


def test_session_date():
    """Test session date extraction."""
    df = pd.DataFrame(
        {
            "time_ny": pd.date_range("2024-01-15 09:30:00", periods=10, freq="5min", tz="America/New_York"),
        }
    )

    dates = session_date(df)
    assert all(dates == date(2024, 1, 15))


def test_is_between():
    """Test time range checking."""
    times = pd.Series(
        pd.date_range("2024-01-15 09:00:00", periods=10, freq="5min", tz="America/New_York")
    )

    mask = is_between(times, "09:30", "09:35", inclusive="both")
    assert mask.sum() == 2  # 09:30 and 09:35


def test_get_ny_time():
    """Test getting specific NY time as UTC."""
    utc_time = get_ny_time("2024-01-15", "09:30")
    assert utc_time.tz.zone == "UTC"
    # 09:30 NY in January is 14:30 UTC (EST, UTC-5)
    assert utc_time.hour == 14

