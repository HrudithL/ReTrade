"""Tests for Opening Range strategy logic."""

import pandas as pd
import pytest

from src.strategy.opening_range import (
    bias,
    build_opening_range,
    entry_signals,
    profit_target,
    stop_price,
)


def test_build_opening_range():
    """Test OR construction from synthetic bars."""
    df_or = pd.DataFrame(
        {
            "time_ny": pd.date_range("2024-01-15 09:30:00", periods=2, freq="5min", tz="America/New_York"),
            "open": [100.0, 100.1],
            "high": [100.5, 101.0],
            "low": [99.5, 100.0],
            "close": [100.0, 100.5],
        }
    )

    or_data = build_opening_range(df_or)
    assert or_data["or_open"] == 100.0  # First open
    assert or_data["or_high"] == 101.0
    assert or_data["or_low"] == 99.5
    assert or_data["or_mid"] == 100.25
    assert or_data["or_close"] == 100.5  # Last close


def test_bias():
    """Test bias determination."""
    assert bias(100.5, 100.0) == "long"
    assert bias(99.5, 100.0) == "short"
    assert bias(100.0, 100.0) == "none"


def test_entry_signals_long():
    """Test long entry signal detection."""
    df_after = pd.DataFrame(
        {
            "time_ny": pd.date_range("2024-01-15 09:35:00", periods=5, freq="5min", tz="America/New_York"),
            "high": [100.0, 100.5, 101.5, 102.0, 103.0],
            "low": [99.0, 99.5, 100.5, 101.0, 102.0],
        }
    )

    signal = entry_signals(df_after, or_high=101.0, or_low=99.0, bias_direction="long")
    assert signal is not None
    assert signal["entry_price"] == 101.0
    assert signal["entry_time"].hour == 9
    assert signal["entry_time"].minute == 40  # First break at 09:40


def test_entry_signals_short():
    """Test short entry signal detection."""
    df_after = pd.DataFrame(
        {
            "time_ny": pd.date_range("2024-01-15 09:35:00", periods=5, freq="5min", tz="America/New_York"),
            "high": [101.0, 100.5, 99.5, 99.0, 98.0],
            "low": [100.0, 99.5, 98.5, 98.0, 97.0],
        }
    )

    signal = entry_signals(df_after, or_high=101.0, or_low=99.0, bias_direction="short")
    assert signal is not None
    assert signal["entry_price"] == 99.0


def test_stop_price():
    """Test stop price calculation."""
    or_high = 101.0
    or_low = 99.0
    or_mid = 100.0

    # Long with opposite boundary
    assert stop_price("long", or_high, or_low, or_mid, use_mid_stop=False) == or_low

    # Long with mid stop
    assert stop_price("long", or_high, or_low, or_mid, use_mid_stop=True) == or_mid

    # Short with opposite boundary
    assert stop_price("short", or_high, or_low, or_mid, use_mid_stop=False) == or_high

    # Short with mid stop
    assert stop_price("short", or_high, or_low, or_mid, use_mid_stop=True) == or_mid


def test_profit_target():
    """Test take profit calculation."""
    # Long trade: entry 100, stop 99, RR 2.0
    tp = profit_target(entry=100.0, stop=99.0, side="long", rr=2.0)
    assert tp == 102.0  # 100 + (100-99)*2

    # Short trade: entry 100, stop 101, RR 2.0
    tp = profit_target(entry=100.0, stop=101.0, side="short", rr=2.0)
    assert tp == 98.0  # 100 - (101-100)*2

