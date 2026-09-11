"""Caching utilities for historical data."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


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
        key_str = json.dumps([instrument, granularity, start, end], sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.parquet"

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
            data = pd.read_parquet(cache_path)
            return data
        except FileNotFoundError:
            logger.warning(
                f"Cache file not found (may have been deleted): {cache_path}",
                exc_info=True,
            )
            return None
        except PermissionError as e:
            logger.error(
                f"Permission denied reading cache file: {cache_path}",
                exc_info=True,
            )
            return None
        except OSError as e:
            logger.error(
                f"OS error reading cache file: {cache_path}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error reading cache file: {cache_path}",
                exc_info=True,
            )
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
            df.to_parquet(cache_path, index=False)
        except PermissionError as e:
            logger.error(
                f"Permission denied writing cache file: {cache_path}",
                exc_info=True,
            )
        except OSError as e:
            logger.error(
                f"OS error writing cache file: {cache_path}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error writing cache file: {cache_path}",
                exc_info=True,
            )

