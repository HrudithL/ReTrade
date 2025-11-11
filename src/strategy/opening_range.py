"""Opening Range Breakout strategy logic."""

from typing import Dict, Literal, Optional

import pandas as pd


def build_opening_range(df_or: pd.DataFrame) -> Dict[str, float]:
    """Build Opening Range from 9:30-9:35 candle(s).

    Aggregates multiple candles (e.g., M1) into a single synthetic 5-minute candle:
    - open = first(09:30)
    - high = max(high)
    - low = min(low)
    - close = last(09:35)

    Args:
        df_or: DataFrame containing the OR window (9:30-9:35)

    Returns:
        Dictionary with or_high, or_low, or_mid, or_close, or_open, or_start, or_end
    """
    if len(df_or) == 0:
        raise ValueError("OR window is empty")

    # Aggregate the OR window into a single synthetic candle
    or_open = df_or["open"].iloc[0]  # First candle open (09:30)
    or_high = df_or["high"].max()
    or_low = df_or["low"].min()
    or_close = df_or["close"].iloc[-1]  # Last candle close (09:35)
    or_mid = (or_high + or_low) / 2.0
    or_start = df_or["time_ny"].iloc[0]
    or_end = df_or["time_ny"].iloc[-1]

    return {
        "or_open": or_open,
        "or_high": or_high,
        "or_low": or_low,
        "or_mid": or_mid,
        "or_close": or_close,
        "or_start": or_start,
        "or_end": or_end,
    }


def bias(or_close: float, or_mid: float) -> Literal["long", "short", "none"]:
    """Determine trading bias from OR close vs mid.

    Args:
        or_close: Close price of OR candle
        or_mid: Mid price of OR

    Returns:
        'long', 'short', or 'none' (if equal, skip day)
    """
    if or_close > or_mid:
        return "long"
    elif or_close < or_mid:
        return "short"
    else:
        return "none"


def entry_signals(
    df_after_or: pd.DataFrame,
    or_high: float,
    or_low: float,
    bias_direction: Literal["long", "short"],
) -> Optional[pd.Series]:
    """Find first breakout signal after OR window.

    Args:
        df_after_or: DataFrame with bars after 9:35
        or_high: OR high level
        or_low: OR low level
        bias_direction: 'long' or 'short'

    Returns:
        Series with first entry signal (index, entry_price, entry_time) or None
    """
    if bias_direction == "none":
        return None

    if bias_direction == "long":
        # First bar where high breaks above OR_high
        breakouts = df_after_or[df_after_or["high"] > or_high]
        if len(breakouts) > 0:
            first_break = breakouts.iloc[0]
            return pd.Series(
                {
                    "entry_price": or_high,  # Entry at OR_high
                    "entry_time": first_break["time_ny"],
                    "entry_index": first_break.name,
                }
            )
    else:  # short
        # First bar where low breaks below OR_low
        breakouts = df_after_or[df_after_or["low"] < or_low]
        if len(breakouts) > 0:
            first_break = breakouts.iloc[0]
            return pd.Series(
                {
                    "entry_price": or_low,  # Entry at OR_low
                    "entry_time": first_break["time_ny"],
                    "entry_index": first_break.name,
                }
            )

    return None


def stop_price(
    side: Literal["long", "short"],
    or_high: float,
    or_low: float,
    or_mid: float,
    use_mid_stop: bool = False,
) -> float:
    """Calculate stop loss price.

    Args:
        side: 'long' or 'short'
        or_high: OR high
        or_low: OR low
        or_mid: OR mid
        use_mid_stop: If True, use OR_mid as stop; else use opposite boundary

    Returns:
        Stop price
    """
    if use_mid_stop:
        return or_mid

    if side == "long":
        return or_low  # Stop below OR
    else:
        return or_high  # Stop above OR


def profit_target(entry: float, stop: float, side: Literal["long", "short"], rr: float) -> float:
    """Calculate take profit price.

    Args:
        entry: Entry price
        stop: Stop loss price
        side: 'long' or 'short'
        rr: Risk-reward ratio

    Returns:
        Take profit price
    """
    risk = abs(entry - stop)

    if side == "long":
        return entry + (risk * rr)
    else:
        return entry - (risk * rr)

