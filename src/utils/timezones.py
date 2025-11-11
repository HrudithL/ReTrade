"""Timezone utilities for handling America/New_York timezone."""

from typing import Optional

import pandas as pd
import pytz

NY_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC


def to_ny(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Convert UTC timestamps to America/New_York timezone.

    Args:
        df: DataFrame with timezone-aware UTC timestamps
        time_col: Name of the time column

    Returns:
        DataFrame with added 'time_ny' column in NY timezone
    """
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    if df[time_col].dtype != "datetime64[ns, UTC]":
        if df[time_col].dtype == "datetime64[ns]":
            df[time_col] = df[time_col].dt.tz_localize(UTC_TZ)
        else:
            raise ValueError(f"Column '{time_col}' must be datetime64[ns] or datetime64[ns, UTC]")

    df["time_ny"] = df[time_col].dt.tz_convert(NY_TZ)
    return df


def session_date(df: pd.DataFrame, time_col: str = "time_ny") -> pd.Series:
    """Extract date component in NY timezone.

    Args:
        df: DataFrame with time_ny column
        time_col: Name of the NY time column

    Returns:
        Series with date (YYYY-MM-DD) in NY timezone
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    return df[time_col].dt.date


def is_between(
    time_series: pd.Series,
    start_time: str,
    end_time: str,
    inclusive: str = "both",
) -> pd.Series:
    """Check if times are between start_time and end_time.

    Args:
        time_series: Series of datetime objects
        start_time: Start time in 'HH:MM' format
        end_time: End time in 'HH:MM' format
        inclusive: 'both', 'left', 'right', or 'neither'

    Returns:
        Boolean Series
    """
    start_hour, start_min = map(int, start_time.split(":"))
    end_hour, end_min = map(int, end_time.split(":"))

    time_of_day = time_series.dt.hour * 60 + time_series.dt.minute
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min

    if inclusive == "both":
        return (time_of_day >= start_minutes) & (time_of_day <= end_minutes)
    elif inclusive == "left":
        return (time_of_day >= start_minutes) & (time_of_day < end_minutes)
    elif inclusive == "right":
        return (time_of_day > start_minutes) & (time_of_day <= end_minutes)
    else:  # neither
        return (time_of_day > start_minutes) & (time_of_day < end_minutes)


def get_ny_time(
    date: str,
    time_str: str,
    tz: Optional[pytz.BaseTzInfo] = None,
) -> pd.Timestamp:
    """Get a specific NY time as UTC timestamp.

    Args:
        date: Date in 'YYYY-MM-DD' format
        time_str: Time in 'HH:MM' format
        tz: Timezone (defaults to NY_TZ)

    Returns:
        Timestamp in UTC
    """
    if tz is None:
        tz = NY_TZ

    hour, minute = map(int, time_str.split(":"))
    dt_ny = pd.Timestamp(date, tz=tz).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return dt_ny.tz_convert(UTC_TZ)

