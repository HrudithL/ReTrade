"""Tests for backtest engine."""

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.strategy.params import StrategyParams
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
            # OR window (09:30-09:35): OR high=100.6, OR low=99.5, OR mid=100.05
            # After 09:35: first bar (09:40) breaks below OR low (99.5) to trigger short entry
            # Then price rallies to hit stop at OR high (100.6)
            "high": [100.5, 100.6, 100.7, 100.8, 100.9, 101.0] + [101.0] * 14,  # Hits stop at 100.6
            "low": [99.5, 99.6, 99.4, 99.3, 99.2, 99.1] + [99.1] * 14,  # 09:40 bar breaks below OR low
            # OR close (09:35) = 99.8, which is below OR mid (100.05) -> short bias
            "close": [100.0, 99.8, 99.5, 100.0, 100.3, 100.5] + [100.5] * 14,
            "volume": [1000] * 20,
        }
    )

    df = to_ny(df)
    return df


def test_engine_long_win():
    """Test engine with long winning trade."""
    df = create_test_data_long_win()

    params = StrategyParams(
        use_mid_stop=False,
        rr=2.0,
        atr_mult=1.0,  # Disable ATR filter for test
        atr_period=20,
        tp_sl_conflict_policy="sl_first",
    )

    engine = BacktestEngine(params, "US500_USD")
    result = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    assert len(result.trades) == 1
    assert result.trades[0].side == "long"
    assert result.trades[0].R > 0  # Should be winning trade


def test_engine_short_loss():
    """Test engine with short losing trade."""
    df = create_test_data_short_loss()

    params = StrategyParams(
        use_mid_stop=False,
        rr=2.0,
        atr_mult=1.0,
        atr_period=20,
        tp_sl_conflict_policy="sl_first",
    )

    engine = BacktestEngine(params, "US500_USD")
    result = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    assert len(result.trades) == 1
    assert result.trades[0].side == "short"
    # Trade should hit stop (short entry ~100.0, stop at OR high 100.5)
    assert result.trades[0].R < 0

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
    params = StrategyParams(
        use_mid_stop=False,
        rr=2.0,
        atr_mult=1.0,
        atr_period=20,
        tp_sl_conflict_policy="sl_first",
    )

    engine = BacktestEngine(params, "US500_USD")
    result = engine.run(df, df_daily_atr=None, df_intraday_fallback=None)

    assert len(result.trades) == 1, "Should generate exactly one trade"
    assert result.trades[0].reason == "SL", "Should exit via stop loss"
    assert result.trades[0].gross_R == -1.0, "Should take loss with sl_first policy"
