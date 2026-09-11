"""Tests for extended metrics functions."""

import pandas as pd
import pytest

from src.backtest.metrics import breakdown_by_day_of_week, breakdown_by_month, equity_curve_stats


def test_equity_curve_stats():
    """Test equity curve statistics calculation."""
    # Create a simple equity curve
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    equity_values = 100000.0 * (1.0 + pd.Series(range(len(dates))) * 0.001)  # Growing equity
    equity_series = pd.Series(equity_values, index=dates)

    stats = equity_curve_stats(equity_series)

    assert "annualized_return" in stats
    assert "annualized_vol" in stats
    assert "sharpe" in stats
    assert "max_drawdown" in stats
    assert "max_drawdown_duration" in stats

    # All should be numeric
    assert isinstance(stats["annualized_return"], (int, float))
    assert isinstance(stats["annualized_vol"], (int, float))
    assert isinstance(stats["sharpe"], (int, float))
    assert isinstance(stats["max_drawdown"], (int, float))
    assert isinstance(stats["max_drawdown_duration"], (int, float))


def test_equity_curve_stats_empty():
    """Test equity curve stats with empty or insufficient data."""
    # Empty series
    empty_series = pd.Series([], dtype=float)
    stats = equity_curve_stats(empty_series)
    assert stats["annualized_return"] == 0.0
    assert stats["annualized_vol"] == 0.0

    # Single value
    single_series = pd.Series([100000.0], index=[pd.Timestamp("2023-01-01")])
    stats = equity_curve_stats(single_series)
    assert stats["annualized_return"] == 0.0


def test_breakdown_by_day_of_week():
    """Test breakdown by day of week."""
    trades_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]),
            "R": [1.0, -1.0, 2.0, -0.5, 1.5],
        }
    )

    breakdown = breakdown_by_day_of_week(trades_df)

    assert "day_of_week" in breakdown.columns
    assert "count" in breakdown.columns
    assert "hit_rate" in breakdown.columns
    assert "avg_R" in breakdown.columns

    # Should have 5 days (Monday-Friday)
    assert len(breakdown) == 5


def test_breakdown_by_day_of_week_empty():
    """Test breakdown by day of week with empty data."""
    empty_df = pd.DataFrame(columns=["date", "R"])
    breakdown = breakdown_by_day_of_week(empty_df)
    assert len(breakdown) == 0
    assert list(breakdown.columns) == ["day_of_week", "count", "hit_rate", "avg_R"]


def test_breakdown_by_month():
    """Test breakdown by month."""
    trades_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15", "2023-05-15"]
            ),
            "R": [1.0, -1.0, 2.0, -0.5, 1.5],
        }
    )

    breakdown = breakdown_by_month(trades_df)

    assert "month" in breakdown.columns
    assert "count" in breakdown.columns
    assert "hit_rate" in breakdown.columns
    assert "avg_R" in breakdown.columns

    # Should have 5 months
    assert len(breakdown) == 5


def test_breakdown_by_month_empty():
    """Test breakdown by month with empty data."""
    empty_df = pd.DataFrame(columns=["date", "R"])
    breakdown = breakdown_by_month(empty_df)
    assert len(breakdown) == 0
    assert list(breakdown.columns) == ["month", "count", "hit_rate", "avg_R"]


def test_breakdown_with_entry_time():
    """Test breakdown using entry_time column instead of date."""
    trades_df = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "R": [1.0, -1.0, 2.0],
        }
    )

    breakdown = breakdown_by_day_of_week(trades_df)
    assert len(breakdown) > 0

    breakdown_month = breakdown_by_month(trades_df)
    assert len(breakdown_month) > 0

