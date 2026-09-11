"""Cost and slippage modeling for backtesting."""

from dataclasses import dataclass
from typing import Literal, Tuple

from src.config.settings import Settings
from src.utils.math import apply_commission, apply_slippage


@dataclass
class InstrumentCostConfig:
    """Cost configuration for an instrument.

    Attributes:
        spread: Spread in price units (half-spread applied to each side)
        commission_per_million: Commission per million units traded (in currency)
        slippage_per_trade: Slippage per trade in price units
    """

    spread: float = 0.0
    commission_per_million: float = 0.0
    slippage_per_trade: float = 0.0

    def __post_init__(self) -> None:
        """Validate cost config."""
        if self.spread < 0:
            raise ValueError(f"spread must be non-negative, got {self.spread}")
        if self.commission_per_million < 0:
            raise ValueError(
                f"commission_per_million must be non-negative, got {self.commission_per_million}"
            )
        if self.slippage_per_trade < 0:
            raise ValueError(
                f"slippage_per_trade must be non-negative, got {self.slippage_per_trade}"
            )


def get_cost_config(instrument: str, model: str = "oanda_default") -> InstrumentCostConfig:
    """Get cost configuration for an instrument based on model identifier.

    Args:
        instrument: OANDA instrument identifier
        model: Cost model identifier (default: "oanda_default")

    Returns:
        InstrumentCostConfig instance

    Raises:
        ValueError: If model is not recognized
    """
    if model == "oanda_default":
        # Use Settings values converted to price units
        # Slippage is in basis points, convert to price units
        slippage_bps = Settings.get_slippage_bps(instrument)
        commission_bps = Settings.get_commission_bps(instrument)

        # For now, we'll use a simple model where:
        # - Spread is typically embedded in OANDA pricing (we use mid prices)
        # - Slippage is applied as basis points (converted in apply_costs)
        # - Commission is typically 0 for OANDA (included in spread)

        # Estimate spread from typical OANDA spreads
        # For index CFDs: ~0.5-2 points spread
        # For forex: ~1-3 pips spread
        if "USD" in instrument and instrument.startswith(("US", "NAS", "SPX")):
            # Index CFD
            spread = 1.0  # 1 point typical spread
        elif "_USD" in instrument or "USD_" in instrument:
            # Forex pair
            spread = 0.0002  # 2 pips typical spread
        else:
            spread = 1.0  # Default

        return InstrumentCostConfig(
            spread=spread,
            commission_per_million=0.0,  # OANDA typically includes in spread
            slippage_per_trade=slippage_bps / 10000.0,  # Convert bps to fraction
        )
    else:
        raise ValueError(f"Unknown cost model: {model}")


def apply_costs(
    entry_price: float,
    exit_price: float,
    side: Literal["long", "short"],
    cost_config: InstrumentCostConfig,
) -> Tuple[float, float, float]:
    """Apply costs (spread, slippage, commission) to entry and exit prices.

    Args:
        entry_price: Base entry price
        exit_price: Base exit price
        side: Trade side ("long" or "short")
        cost_config: Cost configuration

    Returns:
        Tuple of (effective_entry, effective_exit, total_cost)
        where total_cost is in currency units per unit traded
    """
    # Apply spread (half on each side)
    half_spread = cost_config.spread / 2.0

    # Apply slippage
    slippage_entry = cost_config.slippage_per_trade * entry_price
    slippage_exit = cost_config.slippage_per_trade * exit_price

    # Effective entry: long pays more (spread + slippage), short receives less
    if side == "long":
        effective_entry = entry_price + half_spread + slippage_entry
    else:  # short
        effective_entry = entry_price - half_spread - slippage_entry

    # Effective exit: long receives less (sells), short pays more (buys to cover)
    if side == "long":
        effective_exit = exit_price - half_spread - slippage_exit
    else:  # short
        effective_exit = exit_price + half_spread + slippage_exit

    # Calculate total cost in price units
    if side == "long":
        total_cost_price_units = (effective_entry - entry_price) + (exit_price - effective_exit)
    else:  # short
        total_cost_price_units = (entry_price - effective_entry) + (effective_exit - exit_price)

    # Commission (typically 0 for OANDA, but included for completeness)
    # Commission is per million units, so for 1 unit it's commission_per_million / 1e6
    commission = cost_config.commission_per_million / 1_000_000.0

    # Total cost includes commission
    total_cost = total_cost_price_units + commission

    return (effective_entry, effective_exit, total_cost)

