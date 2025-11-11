"""Caching utilities for historical data."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd


class CacheManager:
    """Manages local cache for historical data."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, instrument: str, granularity: str, start: str, end: str) -> str:
        """Generate cache key from parameters."""
        key_str = f"{instrument}_{granularity}_{start}_{end}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self,
        instrument: str,
        granularity: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Get cached data if available.

        Args:
            instrument: OANDA instrument
            granularity: Candle granularity
            start: Start date
            end: End date

        Returns:
            Cached DataFrame or None
        """
        cache_key = self._cache_key(instrument, granularity, start, end)
        cache_path = self._cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception:
            return None

    def save(
        self,
        instrument: str,
        granularity: str,
        start: str,
        end: str,
        df: pd.DataFrame,
    ) -> None:
        """Save data to cache.

        Args:
            instrument: OANDA instrument
            granularity: Candle granularity
            start: Start date
            end: End date
            df: DataFrame to cache
        """
        cache_key = self._cache_key(instrument, granularity, start, end)
        cache_path = self._cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

