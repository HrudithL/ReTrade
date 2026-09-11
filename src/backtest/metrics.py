"""Backtest metrics calculation."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(trades_df: pd.DataFrame, trading_days: Optional[int] = None) -> Dict[str, float]:
    """Compute backtest performance metrics.

    Args:
        trades_df: TradeLog DataFrame with 'R' column. If annualization is desired,
            the DataFrame should contain timestamp columns ('date', 'entry_time', or
            'exit_time') to infer trading frequency, or trading_days should be provided.
        trading_days: Optional number of trading days per year for Sharpe annualization.
            If None, attempts to infer from timestamps. If inference fails, returns
            non-annualized Sharpe ratio.

    Returns:
        Dictionary of metrics. Sharpe ratio is annualized only if trading_days is
        provided or can be inferred from timestamps.
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

    if "R" not in trades_df.columns:
        raise ValueError("trades_df must contain an 'R' column")

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

    # Sharpe ratio (using sample standard deviation)
    if len(R) > 1:
        std_R = np.std(R, ddof=1)  # Sample standard deviation
        if std_R > 0:
            sharpe = np.mean(R) / std_R
            
            # Annualize if trading_days is provided or can be inferred
            annual_factor = None
            if trading_days is not None:
                annual_factor = np.sqrt(trading_days)
            else:
                # Try to infer trading frequency from timestamps
                timestamp_col = None
                for col in ["entry_time", "exit_time", "date"]:
                    if col in trades_df.columns:
                        timestamp_col = col
                        break
                
                if timestamp_col is not None:
                    try:
                        timestamps = pd.to_datetime(trades_df[timestamp_col])
                        if len(timestamps) > 1:
                            # Estimate trading days per year from time span
                            time_span_days = (timestamps.max() - timestamps.min()).days
                            if time_span_days > 0:
                                num_trades = len(trades_df)
                                # Estimate trading days per year
                                estimated_trading_days = (num_trades / time_span_days) * 365.25
                                annual_factor = np.sqrt(estimated_trading_days)
                    except (ValueError, TypeError):
                        # If timestamp parsing fails, don't annualize
                        pass
            
            if annual_factor is not None:
                sharpe *= annual_factor
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Payoff ratio
    winning_trades = R[R > 0]
    losing_trades = R[R < 0]
    avg_win_R = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss_R = losing_trades.mean() if len(losing_trades) > 0 else 0.0
    payoff_ratio = avg_win_R / abs(avg_loss_R) if avg_loss_R != 0 else (avg_win_R if avg_win_R > 0 else 0.0)

    # Hit rate (alias for win_rate)
    hit_rate = win_rate

    return {
        "win_rate": win_rate,
        "hit_rate": hit_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "expectancy": expectancy,
        "max_drawdown_R": max_drawdown_R,
        "longest_loser": max_losing_streak,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "num_trades": len(trades_df),
        "payoff_ratio": payoff_ratio,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
    }


def equity_curve_stats(equity_series: pd.Series, trading_days_per_year: int = 252) -> Dict[str, float]:
    """Calculate statistics from an equity curve.

    Args:
        equity_series: Series of equity values over time (index should be dates)
        trading_days_per_year: Number of trading days per year (default: 252)

    Returns:
        Dictionary with:
        - annualized_return: Annualized return
        - annualized_vol: Annualized volatility
        - sharpe: Sharpe ratio (annualized)
        - max_drawdown: Maximum drawdown (as fraction)
        - max_drawdown_duration: Maximum drawdown duration in days
    """
    if len(equity_series) < 2:
        return {
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0.0,
        }

    # Calculate returns
    returns = equity_series.pct_change().dropna()

    if len(returns) == 0:
        return {
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0.0,
        }

    # Annualized return
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    num_periods = len(equity_series) - 1
    periods_per_year = trading_days_per_year / max(1, num_periods) if num_periods > 0 else 1.0
    annualized_return = (1.0 + total_return) ** (1.0 / periods_per_year) - 1.0 if periods_per_year > 0 else 0.0

    # Annualized volatility
    if len(returns) > 1:
        std_daily = returns.std()
        annualized_vol = std_daily * np.sqrt(trading_days_per_year)
    else:
        annualized_vol = 0.0

    # Sharpe ratio
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    # Maximum drawdown
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Maximum drawdown duration
    max_drawdown_duration = 0.0
    if len(drawdown) > 0:
        in_drawdown = drawdown < -0.001  # Consider in drawdown if > 0.1%
        if in_drawdown.any():
            # Find longest consecutive period in drawdown
            groups = (in_drawdown != in_drawdown.shift()).cumsum()
            drawdown_periods = in_drawdown.groupby(groups).sum()
            if len(drawdown_periods) > 0:
                max_drawdown_duration = float(drawdown_periods.max())

    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
    }


def breakdown_by_day_of_week(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Break down trades by day of week.

    Args:
        trades_df: DataFrame with 'date' or 'entry_time' column and 'R' column

    Returns:
        DataFrame with columns: day_of_week, count, hit_rate, avg_R
    """
    if len(trades_df) == 0:
        return pd.DataFrame(columns=["day_of_week", "count", "hit_rate", "avg_R"])

    # Get date column
    date_col = None
    for col in ["date", "entry_time", "exit_time"]:
        if col in trades_df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("trades_df must contain 'date', 'entry_time', or 'exit_time' column")

    df = trades_df.copy()
    dates = pd.to_datetime(df[date_col])
    df["day_of_week"] = dates.dt.day_name()

    # Group by day of week
    grouped = df.groupby("day_of_week").agg(
        {
            "R": ["count", lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.0, "mean"],
        }
    )
    grouped.columns = ["count", "hit_rate", "avg_R"]
    grouped = grouped.reset_index()

    # Order by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    grouped["day_of_week"] = pd.Categorical(grouped["day_of_week"], categories=day_order, ordered=True)
    grouped = grouped.sort_values("day_of_week").reset_index(drop=True)

    return grouped


def breakdown_by_month(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Break down trades by month.

    Args:
        trades_df: DataFrame with 'date' or 'entry_time' column and 'R' column

    Returns:
        DataFrame with columns: month, count, hit_rate, avg_R
    """
    if len(trades_df) == 0:
        return pd.DataFrame(columns=["month", "count", "hit_rate", "avg_R"])

    # Get date column
    date_col = None
    for col in ["date", "entry_time", "exit_time"]:
        if col in trades_df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("trades_df must contain 'date', 'entry_time', or 'exit_time' column")

    df = trades_df.copy()
    dates = pd.to_datetime(df[date_col])
    df["month"] = dates.dt.month_name()

    # Group by month
    grouped = df.groupby("month").agg(
        {
            "R": ["count", lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.0, "mean"],
        }
    )
    grouped.columns = ["count", "hit_rate", "avg_R"]
    grouped = grouped.reset_index()

    # Order by month
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    grouped["month"] = pd.Categorical(grouped["month"], categories=month_order, ordered=True)
    grouped = grouped.sort_values("month").reset_index(drop=True)

    return grouped

