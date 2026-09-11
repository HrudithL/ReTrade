"""Timezone utilities for handling timezone conversions."""

import re
from typing import Optional, Union

import pandas as pd
import pytz

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC


def get_timezone(tz_str: str) -> Union[ZoneInfo, pytz.BaseTzInfo]:
    """Get a timezone object from a timezone string.

    Tries zoneinfo.ZoneInfo first, falls back to pytz if needed.

    Args:
        tz_str: Timezone string (e.g., 'America/New_York', 'Europe/London')

    Returns:
        Timezone object (ZoneInfo or pytz timezone)

    Raises:
        ValueError: If the timezone string is invalid
    """
    tz_str = tz_str.strip()
    if not tz_str:
        raise ValueError("Timezone string cannot be empty")

    # Try zoneinfo first (preferred for Python 3.9+)
    try:
        return ZoneInfo(tz_str)
    except Exception:
        pass

    # Fall back to pytz
    try:
        return pytz.timezone(tz_str)
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(
            f"Invalid timezone: '{tz_str}'. "
            "Use IANA timezone names (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')"
        ) from None


def to_timezone(df: pd.DataFrame, tz_str: str, time_col: str = "time", output_col: str = "time_ny") -> pd.DataFrame:
    """Convert UTC timestamps to the specified timezone.

    Args:
        df: DataFrame with timezone-aware UTC timestamps
        tz_str: Timezone string (e.g., 'America/New_York', 'Europe/London')
        time_col: Name of the input time column
        output_col: Name of the output time column (default: 'time_ny' for backward compatibility)

    Returns:
        DataFrame with added time column in the specified timezone
    """
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise ValueError(f"Column '{time_col}' must be a datetime64 dtype (naive or UTC-aware)")

    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize(UTC_TZ)

    tz = get_timezone(tz_str)
    df[output_col] = df[time_col].dt.tz_convert(tz)
    return df


def to_ny(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Convert UTC timestamps to America/New_York timezone.

    Convenience wrapper for to_timezone() with NY timezone.

    Args:
        df: DataFrame with timezone-aware UTC timestamps
        time_col: Name of the time column

    Returns:
        DataFrame with added 'time_ny' column in NY timezone
    """
    return to_timezone(df, "America/New_York", time_col=time_col)


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


def _parse_time_string(time_str: str, param_name: str) -> tuple[int, int]:
    """Parse and validate a time string in HH:MM format.

    Args:
        time_str: Time string to parse
        param_name: Name of the parameter (for error messages)

    Returns:
        Tuple of (hour, minute) as integers

    Raises:
        ValueError: If the time string is malformed or out of range
    """
    # Strip whitespace
    time_str = time_str.strip()

    # Validate HH:MM pattern
    if not re.match(r"^\d{2}:\d{2}$", time_str):
        raise ValueError(
            f"{param_name} must be in 'HH:MM' format (e.g., '09:30'), got '{time_str}'"
        )

    # Parse with error handling
    try:
        hour, minute = map(int, time_str.split(":"))
    except ValueError as e:
        raise ValueError(
            f"{param_name} must be in 'HH:MM' format with numeric values, got '{time_str}'"
        ) from e

    # Validate ranges
    if not (0 <= hour <= 23):
        raise ValueError(
            f"{param_name} hour must be between 0 and 23, got '{time_str}' (hour: {hour})"
        )
    if not (0 <= minute <= 59):
        raise ValueError(
            f"{param_name} minute must be between 0 and 59, got '{time_str}' (minute: {minute})"
        )

    return hour, minute


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
    start_hour, start_min = _parse_time_string(start_time, "start_time")
    end_hour, end_min = _parse_time_string(end_time, "end_time")

    time_of_day = time_series.dt.hour * 60 + time_series.dt.minute
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min

    valid_inclusive = {"both", "left", "right", "neither"}
    if inclusive not in valid_inclusive:
        raise ValueError(f"inclusive must be one of {valid_inclusive}, got '{inclusive}'")

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
    # Validate time format
    try:
        hour, minute = map(int, time_str.split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"Invalid time values: {time_str}")
    except (ValueError, AttributeError) as e:
        raise ValueError(f"time_str must be in 'HH:MM' format, got '{time_str}'") from e

    if tz is None:
        tz = NY_TZ

    try:
        dt_ny = pd.Timestamp(date, tz=tz).replace(hour=hour, minute=minute, second=0, microsecond=0)
    except Exception as e:
        raise ValueError(f"Invalid date format: {date}") from e
    return dt_ny.tz_convert(UTC_TZ)

