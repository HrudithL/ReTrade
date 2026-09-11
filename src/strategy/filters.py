"""Strategy filters for trade selection."""

from typing import Optional
import pandas as pd

def atr_filter(daily_atr: float, threshold_atr: float, multiplier: float = 1.2) -> bool:
    """ATR-based session filter.

    Args:
        daily_atr: Today's daily ATR value
        threshold_atr: 20-period average ATR
        multiplier: Multiplier threshold (default 1.2)

    Returns:
        True if daily_atr >= multiplier * threshold_atr
    """
    if threshold_atr == 0:
        return False  # No threshold data

    return daily_atr >= (multiplier * threshold_atr)


def volume_filter(df_day: pd.DataFrame, min_volume: Optional[float] = None) -> bool:
    """Volume filter (stubbed for future use).

    Args:
        df_day: DataFrame for the day
        min_volume: Minimum volume threshold

    Returns:
        True if filter passes
    """
    if min_volume is None:
        return True

    if df_day.empty:
        return False

    if "volume" not in df_day.columns:
        return False

    avg_volume = df_day["volume"].dropna().mean()
    if pd.isna(avg_volume):
        return False

    return avg_volume >= min_volume


def overnight_range_filter(
    prev_close: float,
    or_open: float,
    max_gap_pct: float = 0.02,
) -> bool:
    """Filter for excessive overnight gaps (stubbed).

    Args:
        prev_close: Previous day's close
        or_open: OR open price
        max_gap_pct: Maximum gap percentage

    Returns:
        True if filter passes
    """
    if prev_close == 0:
        return True

    gap_pct = abs(or_open - prev_close) / prev_close
    return gap_pct <= max_gap_pct

