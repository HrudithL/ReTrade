"""OANDA v20 REST API client for fetching historical candles."""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from tqdm import tqdm

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class OandaClient:
    """Client for fetching historical candle data from OANDA."""

    # OANDA returns max 5000 candles per request
    MAX_CANDLES = 5000

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """Initialize OANDA client.

        Args:
            api_key: OANDA API key (defaults to Settings.OANDA_API_KEY)
            account_id: OANDA account ID (defaults to Settings.OANDA_ACCOUNT_ID)
            environment: 'practice' or 'live' (defaults to Settings.OANDA_ENV)
        """
        self.api_key = api_key or Settings.OANDA_API_KEY
        self.account_id = account_id or Settings.OANDA_ACCOUNT_ID
        self.environment = environment or Settings.OANDA_ENV

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA API key and account ID must be provided")

        self.api = API(access_token=self.api_key, environment=self.environment)

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        start: str,
        end: str,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical candles from OANDA.

        Args:
            instrument: OANDA instrument (e.g., 'US500_USD')
            granularity: Candle granularity (e.g., 'M5', 'M15', 'D')
            start: Start date in 'YYYY-MM-DD' format
            end: End date in 'YYYY-MM-DD' format (inclusive)
            show_progress: Show progress bar

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
            'time' is timezone-aware UTC datetime
        """
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

        # Validate granularity and compute minutes_per_candle
        granularity_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D": 1440,
        }
        if granularity not in granularity_minutes:
            supported = ", ".join(sorted(granularity_minutes.keys()))
            raise ValueError(
                f"Unsupported granularity '{granularity}'. "
                f"Supported granularities: {supported}"
            )
        minutes_per_candle = granularity_minutes[granularity]

        all_candles = []
        current_start = start_dt

        # Retry configuration
        max_retries = 5
        base_delay = 1.0  # Initial delay in seconds

        # Estimate total requests needed
        if show_progress:
            total_days = (end_dt - start_dt).days
            pbar = tqdm(total=total_days, desc=f"Fetching {instrument} {granularity}")

        while current_start < end_dt:
            # Calculate end for this batch (max MAX_CANDLES candles)
            batch_days = max(1, int(self.MAX_CANDLES * minutes_per_candle / 1440))
            batch_end = min(current_start + pd.Timedelta(days=batch_days), end_dt)

            params = {
                "from": current_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "to": batch_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "granularity": granularity,
                "price": "M",  # Mid prices
            }

            # Retry loop for this batch
            retries = 0
            batch_success = False
            no_more_data = False
            while retries < max_retries:
                try:
                    request = InstrumentsCandles(instrument=instrument, params=params)
                    response = self.api.request(request)

                    if "candles" not in response:
                        # No more data available, exit main loop
                        no_more_data = True
                        batch_success = True  # Mark as success to exit cleanly
                        break

                    for candle in response["candles"]:
                        if candle["complete"]:
                            all_candles.append(
                                {
                                    "time": pd.Timestamp(candle["time"], tz="UTC"),
                                    "open": float(candle["mid"]["o"]),
                                    "high": float(candle["mid"]["h"]),
                                    "low": float(candle["mid"]["l"]),
                                    "close": float(candle["mid"]["c"]),
                                    "volume": int(candle["volume"]),
                                }
                            )

                    if show_progress:
                        pbar.update(min(batch_days, (batch_end - current_start).days))

                    # Rate limiting: OANDA allows ~120 requests per minute
                    time.sleep(0.5)

                    current_start = batch_end
                    batch_success = True
                    break  # Exit retry loop on success

                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException,
                ) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) reached for batch "
                            f"{current_start.strftime('%Y-%m-%d')} to "
                            f"{batch_end.strftime('%Y-%m-%d')}. Error: {e}",
                            exc_info=True,
                        )
                        # Skip this batch and continue to next one
                        current_start = batch_end
                        break
                    delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(
                        f"Error fetching candles for batch "
                        f"{current_start.strftime('%Y-%m-%d')} to "
                        f"{batch_end.strftime('%Y-%m-%d')}: {e}. "
                        f"Retrying in {delay:.1f} seconds (attempt {retries}/{max_retries})",
                        exc_info=True,
                    )
                    time.sleep(delay)
                except Exception as e:
                    # Unexpected exceptions should be raised immediately
                    logger.error(
                        f"Unexpected error fetching candles for batch "
                        f"{current_start.strftime('%Y-%m-%d')} to "
                        f"{batch_end.strftime('%Y-%m-%d')}: {e}",
                        exc_info=True,
                    )
                    raise

            # If batch failed after all retries, we've already advanced current_start
            # Continue to next iteration
            if not batch_success:
                continue
            elif no_more_data:
                # No more data available, exit main loop
                break

        if show_progress:
            pbar.close()

        if not all_candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles)
        df = df.sort_values("time").reset_index(drop=True)
        df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)

        return df

