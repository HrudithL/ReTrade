"""Tests for position sizing and risk management."""

import pytest

from src.risk.position_sizing import (
    enforce_daily_risk_limit,
    get_pip_value,
    position_size_from_risk,
)


def test_get_pip_value():
    """Test pip value lookup."""
    assert get_pip_value("US500_USD") == 1.0
    assert get_pip_value("EUR_USD") == 0.0001
    assert get_pip_value("USD_JPY") == 0.01
    assert get_pip_value("UNKNOWN") == 1.0  # Default


def test_position_size_from_risk():
    """Test position sizing calculation."""
    equity = 100000.0
    risk_per_trade = 0.0025  # 0.25%
    entry_price = 4000.0
    stop_price = 3990.0  # 10 point stop
    pip_value = 1.0

    # Risk amount = 100000 * 0.0025 = 250
    # Stop distance = 10 points
    # Position size = 250 / 10 = 25 units
    size = position_size_from_risk(equity, risk_per_trade, entry_price, stop_price, pip_value)
    assert size == 25.0

    # Test with smaller stop (larger position)
    stop_price2 = 3995.0  # 5 point stop
    size2 = position_size_from_risk(equity, risk_per_trade, entry_price, stop_price2, pip_value)
    assert size2 == 50.0

    # Test with zero stop distance
    size3 = position_size_from_risk(equity, risk_per_trade, entry_price, entry_price, pip_value)
    assert size3 == 0.0


def test_position_size_from_risk_forex():
    """Test position sizing for forex."""
    equity = 100000.0
    risk_per_trade = 0.0025
    entry_price = 1.1000
    stop_price = 1.0990  # 10 pips stop
    pip_value = 0.0001

    size = position_size_from_risk(equity, risk_per_trade, entry_price, stop_price, pip_value)
    # Risk = 250, stop = 0.0010, size = 250 / 0.0010 = 250000 units
    assert size > 0


def test_enforce_daily_risk_limit():
    """Test daily risk limit enforcement."""
    equity = 100000.0
    max_daily_risk = 0.01  # 1%

    # No trades yet - should allow
    assert enforce_daily_risk_limit([], equity, max_daily_risk) is True

    # One trade at 0.5% risk - should allow
    assert enforce_daily_risk_limit([500.0], equity, max_daily_risk) is True

    # Multiple trades totaling 0.8% - should allow
    assert enforce_daily_risk_limit([300.0, 500.0], equity, max_daily_risk) is True

    # Trades totaling exactly 1% - should not allow (strict <)
    assert enforce_daily_risk_limit([1000.0], equity, max_daily_risk) is False

    # Trades totaling more than 1% - should not allow
    assert enforce_daily_risk_limit([600.0, 500.0], equity, max_daily_risk) is False


def test_position_size_validation():
    """Test position sizing validation."""
    with pytest.raises(ValueError, match="risk_per_trade must be positive"):
        position_size_from_risk(100000, -0.001, 4000, 3990, 1.0)

    with pytest.raises(ValueError, match="entry_price must be positive"):
        position_size_from_risk(100000, 0.0025, -4000, 3990, 1.0)

    with pytest.raises(ValueError, match="stop_price must be positive"):
        position_size_from_risk(100000, 0.0025, 4000, -3990, 1.0)


def test_daily_risk_limit_validation():
    """Test daily risk limit validation."""
    with pytest.raises(ValueError, match="max_daily_risk must be positive"):
        enforce_daily_risk_limit([], 100000, -0.01)

    with pytest.raises(ValueError, match="equity must be positive"):
        enforce_daily_risk_limit([], -100000, 0.01)

