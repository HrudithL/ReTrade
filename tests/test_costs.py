"""Tests for cost and slippage modeling."""

import pytest

from src.execution.costs import InstrumentCostConfig, apply_costs, get_cost_config


def test_instrument_cost_config():
    """Test InstrumentCostConfig creation and validation."""
    config = InstrumentCostConfig(spread=1.0, commission_per_million=0.0, slippage_per_trade=0.0001)
    assert config.spread == 1.0
    assert config.commission_per_million == 0.0
    assert config.slippage_per_trade == 0.0001

    # Test validation
    with pytest.raises(ValueError, match="spread must be non-negative"):
        InstrumentCostConfig(spread=-1.0)

    with pytest.raises(ValueError, match="commission_per_million must be non-negative"):
        InstrumentCostConfig(commission_per_million=-1.0)

    with pytest.raises(ValueError, match="slippage_per_trade must be non-negative"):
        InstrumentCostConfig(slippage_per_trade=-1.0)


def test_get_cost_config():
    """Test cost config retrieval."""
    config = get_cost_config("US500_USD", "oanda_default")
    assert isinstance(config, InstrumentCostConfig)
    assert config.spread > 0

    with pytest.raises(ValueError, match="Unknown cost model"):
        get_cost_config("US500_USD", "unknown_model")


def test_apply_costs_long():
    """Test cost application for long trades."""
    entry_price = 4000.0
    exit_price = 4010.0
    side = "long"
    config = InstrumentCostConfig(spread=1.0, commission_per_million=0.0, slippage_per_trade=0.0001)

    effective_entry, effective_exit, total_cost = apply_costs(entry_price, exit_price, side, config)

    # Long entry: pays spread/2 + slippage
    expected_entry = entry_price + 0.5 + (entry_price * 0.0001)
    assert abs(effective_entry - expected_entry) < 0.01

    # Long exit: receives less (spread/2 - slippage)
    expected_exit = exit_price - 0.5 - (exit_price * 0.0001)
    assert abs(effective_exit - expected_exit) < 0.01

    # Total cost should be positive
    assert total_cost > 0


def test_apply_costs_short():
    """Test cost application for short trades."""
    entry_price = 4000.0
    exit_price = 3990.0
    side = "short"
    config = InstrumentCostConfig(spread=1.0, commission_per_million=0.0, slippage_per_trade=0.0001)

    effective_entry, effective_exit, total_cost = apply_costs(entry_price, exit_price, side, config)

    # Short entry: receives less (entry - spread/2 - slippage)
    expected_entry = entry_price - 0.5 - (entry_price * 0.0001)
    assert abs(effective_entry - expected_entry) < 0.01

    # Short exit: pays more (exit + spread/2 + slippage)
    expected_exit = exit_price + 0.5 + (exit_price * 0.0001)
    assert abs(effective_exit - expected_exit) < 0.01

    # Total cost should be positive
    assert total_cost > 0


def test_apply_costs_with_commission():
    """Test cost application with commission."""
    entry_price = 4000.0
    exit_price = 4010.0
    side = "long"
    config = InstrumentCostConfig(spread=1.0, commission_per_million=10.0, slippage_per_trade=0.0001)

    effective_entry, effective_exit, total_cost = apply_costs(entry_price, exit_price, side, config)

    # Commission should be included in total cost
    # For 1 unit: commission = 10.0 / 1e6 = 0.00001 (very small)
    assert total_cost > 0

