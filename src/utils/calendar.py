"""Calendar utilities for building trading sessions."""

from datetime import date
from typing import Dict

import pandas as pd

from src.utils.timezones import session_date


def build_daily_groups(df: pd.DataFrame) -> Dict[date, pd.DataFrame]:
    """Group DataFrame by trading day in NY timezone.

    Args:
        df: DataFrame with 'time_ny' column

    Returns:
        Dictionary mapping date -> DataFrame for that day
    """
    if "time_ny" not in df.columns:
        raise ValueError("DataFrame must have 'time_ny' column")

    df = df.copy()
    df["session_date"] = session_date(df, "time_ny")

    daily_groups: Dict[date, pd.DataFrame] = {}
    for session_dt, group_df in df.groupby("session_date"):
        daily_groups[session_dt] = group_df.sort_values("time_ny").reset_index(drop=True)

    return daily_groups


def get_or_window(
    df_day: pd.DataFrame,
    or_start: str = "09:30",
    or_end: str = "09:35",
) -> pd.DataFrame:
    """Extract the Opening Range window (9:30-9:35) from a day's data.

    Args:
        df_day: DataFrame for a single day with 'time_ny' column
        or_start: Start time in 'HH:MM' format
        or_end: End time in 'HH:MM' format

    Returns:
        DataFrame filtered to OR window
    """
    from src.utils.timezones import is_between

    if "time_ny" not in df_day.columns:
        raise ValueError("DataFrame must have 'time_ny' column")

    mask = is_between(df_day["time_ny"], or_start, or_end, inclusive="both")
    return df_day[mask].copy()


def is_valid_trading_day(df_day: pd.DataFrame, or_start: str = "09:30", or_end: str = "09:35") -> bool:
    """Check if a day has valid OR window data.

    Args:
        df_day: DataFrame for a single day
        or_start: Start time in 'HH:MM' format
        or_end: End time in 'HH:MM' format

    Returns:
        True if OR window exists and has data
    """
    or_df = get_or_window(df_day, or_start, or_end)
    return len(or_df) > 0

