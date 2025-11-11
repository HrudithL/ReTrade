"""Data fetching utilities with caching support."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import Settings
from src.data.oanda_client import OandaClient
from src.io.cache import CacheManager


def fetch_historical_data(
    instrument: str,
    granularity: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical data with optional caching.

    Args:
        instrument: OANDA instrument
        granularity: Candle granularity
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        use_cache: Whether to use cache

    Returns:
        DataFrame with candle data
    """
    cache = CacheManager(cache_dir=Settings.CACHE_DIR)

    if use_cache and Settings.ENABLE_CACHE:
        cached = cache.get(instrument, granularity, start, end)
        if cached is not None:
            return cached

    client = OandaClient()
    df = client.get_candles(instrument, granularity, start, end)

    if use_cache and Settings.ENABLE_CACHE:
        cache.save(instrument, granularity, start, end, df)

    return df


def fetch_daily_for_atr(
    instrument: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch daily candles specifically for ATR calculation.

    Args:
        instrument: OANDA instrument
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        use_cache: Whether to use cache

    Returns:
        DataFrame with daily candle data, or None if unavailable
    """
    try:
        return fetch_historical_data(instrument, "D", start, end, use_cache)
    except Exception as e:
        print(f"Warning: Could not fetch daily candles for ATR: {e}")
        return None


def resample_to_daily(df_intraday: pd.DataFrame, time_col: str = "time_ny") -> pd.DataFrame:
    """Resample intraday OHLC data to daily bars.

    Args:
        df_intraday: DataFrame with intraday OHLC data
        time_col: Name of the time column (should be timezone-aware)

    Returns:
        DataFrame with daily OHLC bars
    """
    if time_col not in df_intraday.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    df = df_intraday.copy()
    df = df.set_index(time_col)

    # Resample to daily
    daily = df.resample("D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Reset index and rename
    daily = daily.reset_index()
    daily = daily.rename(columns={time_col: "time_ny"})

    # Filter out days with no data
    daily = daily.dropna(subset=["open", "high", "low", "close"])

    return daily

