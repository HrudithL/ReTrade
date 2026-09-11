"""Position sizing and risk management utilities."""

from typing import Dict, List, Optional

# Pip/point values for common OANDA instruments
# For index CFDs: 1 point = 1 USD per unit
# For forex: 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs
INSTRUMENT_PIP_VALUES: Dict[str, float] = {
    "US500_USD": 1.0,  # S&P 500 CFD: 1 point = 1 USD
    "NAS100_USD": 1.0,  # NASDAQ 100 CFD: 1 point = 1 USD
    "SPX500_USD": 1.0,  # S&P 500 CFD: 1 point = 1 USD
    "EUR_USD": 0.0001,  # Forex: 1 pip = 0.0001
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,  # JPY pairs: 1 pip = 0.01
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "default": 1.0,  # Default for index CFDs
}


def get_pip_value(instrument: str) -> float:
    """Get pip/point value for an instrument.

    Args:
        instrument: OANDA instrument identifier (e.g., "US500_USD")

    Returns:
        Pip/point value in price units
    """
    return INSTRUMENT_PIP_VALUES.get(instrument, INSTRUMENT_PIP_VALUES["default"])


def position_size_from_risk(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    pip_value: float,
) -> float:
    """Calculate position size in units based on risk per trade.

    Args:
        equity: Current account equity
        risk_per_trade: Risk per trade as fraction of equity (e.g., 0.0025 = 0.25%)
        entry_price: Entry price
        stop_price: Stop loss price
        pip_value: Pip/point value for the instrument

    Returns:
        Position size in units (rounded to reasonable precision)

    Raises:
        ValueError: If risk_per_trade <= 0, entry_price <= 0, or stop_price <= 0
    """
    if risk_per_trade <= 0:
        raise ValueError(f"risk_per_trade must be positive, got {risk_per_trade}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if stop_price <= 0:
        raise ValueError(f"stop_price must be positive, got {stop_price}")

    # Calculate risk amount in currency
    risk_amount = equity * risk_per_trade

    # Calculate stop distance in price units
    stop_distance = abs(entry_price - stop_price)

    if stop_distance == 0:
        # Cannot calculate position size if stop is at entry
        return 0.0

    # Convert stop distance to pips/points
    stop_distance_pips = stop_distance / pip_value

    # Calculate units: risk_amount / stop_distance_pips
    # Since pip_value converts price units to currency per unit per pip
    units = risk_amount / (stop_distance_pips * pip_value)

    # Round to reasonable precision (avoid fractional units for most instruments)
    # For index CFDs, typically whole units; for forex, can be fractional
    if pip_value >= 1.0:  # Index CFDs
        units = round(units)
    else:  # Forex
        units = round(units, 2)

    return max(0.0, units)


def enforce_daily_risk_limit(
    trades_for_day: List[float],
    equity: float,
    max_daily_risk: float,
) -> bool:
    """Check if adding another trade would exceed daily risk limit.

    Args:
        trades_for_day: List of risk amounts (in currency) for trades already taken today
        equity: Current account equity
        max_daily_risk: Maximum daily risk as fraction of equity (e.g., 0.01 = 1%)

    Returns:
        True if another trade is allowed, False if daily limit would be exceeded

    Raises:
        ValueError: If max_daily_risk <= 0 or equity <= 0
    """
    if max_daily_risk <= 0:
        raise ValueError(f"max_daily_risk must be positive, got {max_daily_risk}")
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")

    total_risk_today = sum(trades_for_day)
    max_risk_allowed = equity * max_daily_risk

    return total_risk_today < max_risk_allowed

