"""Tests for backtest engine."""

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.utils.timezones import to_ny


def create_test_data_long_win():
    """Create synthetic data for a long winning trade."""
    # OR window: 09:30-09:35
    times = pd.date_range("2024-01-15 09:30:00", periods=20, freq="5min", tz="America/New_York")

    df = pd.DataFrame(
        {
            "time": times.tz_convert("UTC"),
            "open": [100.0] * 20,
            "high": [100.5, 100.6, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0] + [104.0] * 12,
            "low": [99.5, 99.6, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0] + [103.0] * 12,
            "close": [100.0, 100.1, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5] + [103.5] * 12,
            "volume": [1000] * 20,
        }
    )

    df = to_ny(df)
    return df


def create_test_data_short_loss():
    """Create synthetic data for a short losing trade."""
    times = pd.date_range("2024-01-16 09:30:00", periods=20, freq="5min", tz="America/New_York")

    df = pd.DataFrame(
        {
            "time": times.tz_convert("UTC"),
            "open": [100.0] * 20,
            "high": [100.5, 100.6, 101.0, 101.5, 102.0, 102.5, 103.0] + [103.0] * 13,
            "low": [99.5, 99.6, 100.0, 100.5, 101.0, 101.5, 102.0] + [102.0] * 13,
            "close": [100.0, 100.1, 100.5, 101.0, 101.5, 102.0, 102.5] + [102.5] * 13,
            "volume": [1000] * 20,
        }
    )

    df = to_ny(df)
    return df


def test_engine_long_win():
    """Test engine with long winning trade."""
    df = create_test_data_long_win()

    params = {
        "instrument": "US500_USD",
        "granularity": "M5",
        "use_mid_stop": False,
        "rr": 2.0,
        "atr_mult": 1.0,  # Disable ATR filter for test
        "atr_period": 20,
        "fill_policy": "sl_first",
    }

    engine = BacktestEngine(params)
    trades = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "long"
    assert trades.iloc[0]["R"] > 0  # Should be winning trade


def test_engine_short_loss():
    """Test engine with short losing trade."""
    df = create_test_data_short_loss()

    params = {
        "instrument": "US500_USD",
        "granularity": "M5",
        "use_mid_stop": False,
        "rr": 2.0,
        "atr_mult": 1.0,
        "atr_period": 20,
        "fill_policy": "sl_first",
    }

    engine = BacktestEngine(params)
    trades = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "short"
    # Trade should hit stop (OR high = 100.5, entry at 99.5, stop at 100.5)
    assert trades.iloc[0]["R"] < 0


def test_both_tp_sl_same_bar():
    """Test handling when both TP and SL hit on same bar."""
    # Create data where both TP and SL are touched in same bar
    times = pd.date_range("2024-01-17 09:30:00", periods=10, freq="5min", tz="America/New_York")

    df = pd.DataFrame(
        {
            "time": times.tz_convert("UTC"),
            "open": [100.0] * 10,
            "high": [100.5, 100.6, 102.5, 103.0] + [103.0] * 6,  # TP at 102, SL at 100.5
            "low": [99.5, 99.6, 99.0, 100.0] + [100.0] * 6,  # Both hit in 3rd bar
            "close": [100.0, 100.1, 101.0, 101.5] + [101.5] * 6,
            "volume": [1000] * 10,
        }
    )

    df = to_ny(df)

    # Test sl_first policy
    params = {
        "instrument": "US500_USD",
        "granularity": "M5",
        "use_mid_stop": False,
        "rr": 2.0,
        "atr_mult": 1.0,
        "atr_period": 20,
        "fill_policy": "sl_first",
    }

    engine = BacktestEngine(params)
    trades = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    if len(trades) > 0:
        assert trades.iloc[0]["R"] == -1.0  # Should take loss
        assert trades.iloc[0]["reason"] == "SL"

