"""OANDA v20 REST API client for fetching historical candles."""

import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from tqdm import tqdm

from src.config.settings import Settings


class OandaClient:
    """Client for fetching historical candle data from OANDA."""

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

        all_candles = []
        current_start = start_dt

        # OANDA returns max 5000 candles per request
        # Estimate total requests needed
        if show_progress:
            total_days = (end_dt - start_dt).days
            pbar = tqdm(total=total_days, desc=f"Fetching {instrument} {granularity}")

        while current_start < end_dt:
            # Calculate end for this batch (max 5000 candles)
            # Approximate: M1 = 1440/day, M5 = 288/day, M15 = 96/day, D = 1/day
            granularity_minutes = {
                "M1": 1,
                "M5": 5,
                "M15": 15,
                "H1": 60,
                "H4": 240,
                "D": 1440,
            }
            minutes_per_candle = granularity_minutes.get(granularity, 5)
            max_candles = 5000
            batch_days = max(1, int(max_candles * minutes_per_candle / 1440))
            batch_end = min(current_start + pd.Timedelta(days=batch_days), end_dt)

            params = {
                "from": current_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "to": batch_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "granularity": granularity,
                "price": "M",  # Mid prices
            }

            try:
                request = InstrumentsCandles(instrument=instrument, params=params)
                response = self.api.request(request)

                if "candles" not in response:
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

            except Exception as e:
                print(f"Error fetching candles: {e}")
                # Retry with exponential backoff
                time.sleep(2)
                continue

        if show_progress:
            pbar.close()

        if not all_candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles)
        df = df.sort_values("time").reset_index(drop=True)
        df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)

        return df

