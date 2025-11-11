"""Mathematical utilities for trading calculations."""

from typing import Literal


def apply_slippage(price: float, side: Literal["long", "short"], slippage_bps: float) -> float:
    """Apply slippage to a fill price.

    Args:
        price: Base price
        side: 'long' or 'short'
        slippage_bps: Slippage in basis points (1 bp = 0.01%)

    Returns:
        Adjusted price
    """
    slippage_pct = slippage_bps / 10000.0
    if side == "long":
        return price * (1 + slippage_pct)  # Buy at higher price
    else:
        return price * (1 - slippage_pct)  # Sell at lower price


def apply_commission(price: float, side: Literal["long", "short"], commission_bps: float) -> float:
    """Apply commission to a fill price.

    Args:
        price: Base price
        side: 'long' or 'short'
        commission_bps: Commission in basis points

    Returns:
        Commission amount (not adjusted price)
    """
    commission_pct = commission_bps / 10000.0
    return price * commission_pct

