"""Backtest metrics calculation."""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Compute backtest performance metrics.

    Args:
        trades_df: TradeLog DataFrame with 'R' column

    Returns:
        Dictionary of metrics
    """
    if len(trades_df) == 0:
        return {
            "win_rate": 0.0,
            "avg_R": 0.0,
            "total_R": 0.0,
            "expectancy": 0.0,
            "max_drawdown_R": 0.0,
            "longest_loser": 0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "num_trades": 0,
        }

    R = trades_df["R"].values

    # Win rate
    wins = (R > 0).sum()
    win_rate = wins / len(R) if len(R) > 0 else 0.0

    # Average R
    avg_R = R.mean()

    # Total R
    total_R = R.sum()

    # Expectancy
    expectancy = avg_R

    # Max drawdown in R
    cumulative = np.cumsum(R)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown_R = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Longest losing streak
    losing_streak = 0
    max_losing_streak = 0
    for r in R:
        if r < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    # Profit factor
    gross_profit = R[R > 0].sum() if (R > 0).any() else 0.0
    gross_loss = abs(R[R < 0].sum()) if (R < 0).any() else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    # Sharpe ratio (using daily R as returns)
    if len(R) > 1:
        sharpe = stats.skew(R) if np.std(R) > 0 else 0.0
        # More standard Sharpe: mean / std * sqrt(252) for daily
        sharpe = (np.mean(R) / np.std(R)) * np.sqrt(252) if np.std(R) > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "expectancy": expectancy,
        "max_drawdown_R": max_drawdown_R,
        "longest_loser": max_losing_streak,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "num_trades": len(trades_df),
    }

