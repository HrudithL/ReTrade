#!/usr/bin/env python3
"""CLI script for running Opening Range Breakout backtests."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.backtest.engine import BacktestEngine
from src.backtest.grid_search import run_grid_search
from src.backtest.metrics import breakdown_by_day_of_week, breakdown_by_month, compute_metrics, equity_curve_stats
from src.config.logging import setup_logging
from src.config.settings import Settings
from src.data.fetch import fetch_daily_for_atr, fetch_historical_data
from src.strategy.params import StrategyParams
from src.utils.timezones import get_timezone, to_timezone


def print_metrics_table(metrics: dict, title: str = "BACKTEST RESULTS") -> None:
    """Print a compact metrics table."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 60)
    print(f"{'Number of Trades':<30} {metrics.get('num_trades', 0):>15}")
    print(f"{'Win Rate':<30} {metrics.get('win_rate', 0.0):>15.2%}")
    print(f"{'Hit Rate':<30} {metrics.get('hit_rate', 0.0):>15.2%}")
    print(f"{'Average R':<30} {metrics.get('avg_R', 0.0):>15.3f}")
    print(f"{'Total R':<30} {metrics.get('total_R', 0.0):>15.3f}")
    print(f"{'Payoff Ratio':<30} {metrics.get('payoff_ratio', 0.0):>15.3f}")
    print(f"{'Expectancy':<30} {metrics.get('expectancy', 0.0):>15.3f}")
    print(f"{'Max Drawdown (R)':<30} {metrics.get('max_drawdown_R', 0.0):>15.3f}")
    print(f"{'Longest Losing Streak':<30} {metrics.get('longest_loser', 0):>15}")
    print(f"{'Profit Factor':<30} {metrics.get('profit_factor', 0.0):>15.3f}")
    print(f"{'Sharpe Ratio':<30} {metrics.get('sharpe', 0.0):>15.3f}")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest Opening Range Breakout Strategy")
    parser.add_argument("--instrument", type=str, required=True, help="OANDA instrument (e.g., US500_USD)")
    parser.add_argument("--granularity", type=str, default="M5", help="Candle granularity (default: M5)")

    # Date range arguments (backward compatible)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) - deprecated, use --train_start or --test_start")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) - deprecated, use --train_end or --test_end")

    # Train/test split arguments
    parser.add_argument("--train_start", type=str, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train_end", type=str, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test_start", type=str, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test_end", type=str, help="Test end date (YYYY-MM-DD)")

    # Strategy parameters
    parser.add_argument(
        "--use_mid_stop",
        action="store_true",
        help="Use OR mid as stop instead of opposite boundary (default: False)",
    )
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-reward ratio (default: 2.0)")
    parser.add_argument("--atr_mult", type=float, default=1.2, help="ATR multiplier (default: 1.2)")
    parser.add_argument("--atr_period", type=int, default=20, help="ATR period (default: 20)")
    parser.add_argument(
        "--or_start_time",
        type=str,
        default="09:30",
        help="Opening range start time in HH:MM format (default: 09:30)",
    )
    parser.add_argument(
        "--or_end_time",
        type=str,
        default="09:35",
        help="Opening range end time in HH:MM format (default: 09:35)",
    )
    parser.add_argument(
        "--tp_sl_conflict_policy",
        type=str,
        choices=["sl_first", "tp_first"],
        default="sl_first",
        help="Policy when both TP and SL hit same bar (default: sl_first)",
    )
    parser.add_argument(
        "--risk_per_trade",
        type=float,
        default=0.0025,
        help="Risk per trade as fraction of equity (default: 0.0025 = 0.25%%)",
    )
    parser.add_argument(
        "--max_daily_risk",
        type=float,
        default=0.01,
        help="Maximum daily risk as fraction of equity (default: 0.01 = 1%%)",
    )

    # Grid search
    parser.add_argument("--grid_search", action="store_true", help="Enable hyperparameter grid search on training period")
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["sharpe", "total_R"],
        default="sharpe",
        help="Metric to use for selecting best params in grid search (default: sharpe)",
    )

    # Other options
    parser.add_argument(
        "--tz",
        type=str,
        default="America/New_York",
        help="Timezone for data conversion (IANA timezone name, e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable cache")
    parser.add_argument("--starting_equity", type=float, default=100000.0, help="Starting equity (default: 100000.0)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Validate timezone
    try:
        tz = get_timezone(args.tz)
        tz_name = args.tz
    except ValueError as e:
        print(f"Error: Invalid timezone '{args.tz}': {e}")
        return

    # Validate settings
    Settings.validate()

    # Determine date ranges
    use_train_test = args.train_start and args.train_end
    use_legacy = args.start and args.end

    if use_train_test:
        train_start = args.train_start
        train_end = args.train_end
        test_start = args.test_start
        test_end = args.test_end
        # For data fetching, use the full range
        data_start = min(train_start, test_start) if test_start else train_start
        data_end = max(train_end, test_end) if test_end else train_end
    elif use_legacy:
        # Backward compatibility
        data_start = args.start
        data_end = args.end
        train_start = None
        train_end = None
        test_start = None
        test_end = None
    else:
        print("Error: Must provide either (--start, --end) or (--train_start, --train_end)")
        return

    print(f"Backtesting {args.instrument}")
    if use_train_test:
        print(f"Training period: {train_start} to {train_end}")
        if test_start and test_end:
            print(f"Test period: {test_start} to {test_end}")
    else:
        print(f"Period: {data_start} to {data_end}")
    print(f"Granularity: {args.granularity}, ATR uses daily candles")
    print(f"RR: {args.rr}, ATR Mult: {args.atr_mult}, Use Mid Stop: {args.use_mid_stop}")
    print(f"Timezone: {tz_name}")
    print()

    # Fetch main data
    print("Fetching main timeframe data...")
    df_main = fetch_historical_data(
        args.instrument,
        args.granularity,
        data_start,
        data_end,
        use_cache=not args.no_cache,
    )

    if len(df_main) == 0:
        print("Error: No data fetched")
        return

    # Convert to specified timezone
    print(f"Converting timestamps to {tz_name} timezone...")
    df_main = to_timezone(df_main, tz_name)

    # Fetch daily candles for ATR calculation
    print("Fetching daily candles for ATR calculation...")
    df_daily = fetch_daily_for_atr(
        args.instrument,
        data_start,
        data_end,
        use_cache=not args.no_cache,
    )

    # Prepare fallback data if daily unavailable
    df_intraday_fallback = None
    if df_daily is None or len(df_daily) == 0:
        print("Daily candles unavailable, will resample intraday data if needed...")
        df_intraday_fallback = df_main.copy()
    else:
        # Convert daily to specified timezone
        df_daily = to_timezone(df_daily, tz_name)

    # Create StrategyParams
    strategy_params = StrategyParams(
        or_start_time=args.or_start_time,
        or_end_time=args.or_end_time,
        rr=args.rr,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        use_mid_stop=args.use_mid_stop,
        tp_sl_conflict_policy=args.tp_sl_conflict_policy,
        risk_per_trade=args.risk_per_trade,
        max_daily_risk=args.max_daily_risk,
    )

    # Engine factory for grid search
    def engine_factory(params: StrategyParams, instrument: str) -> BacktestEngine:
        return BacktestEngine(params, instrument, starting_equity=args.starting_equity)

    # Run grid search if enabled
    if args.grid_search and use_train_test:
        print("\nRunning hyperparameter grid search on training period...")
        param_grid = {
            "rr": [1.5, 2.0, 2.5],
            "atr_mult": [1.0, 1.2, 1.5],
            "use_mid_stop": [True, False],
        }

        grid_results = run_grid_search(
            engine_factory=engine_factory,
            train_start=train_start,
            train_end=train_end,
            param_grid=param_grid,
            instrument=args.instrument,
            df_all=df_main,
            df_daily_atr=df_daily,
            df_intraday_fallback=df_intraday_fallback,
            selection_metric=args.selection_metric,
        )

        # Use best params for test period
        strategy_params = grid_results["best_params"]
        print(f"\nBest parameters selected: RR={strategy_params.rr}, ATR Mult={strategy_params.atr_mult}, Use Mid Stop={strategy_params.use_mid_stop}")

    # Run backtest
    print("\nRunning backtest...")
    engine = engine_factory(strategy_params, args.instrument)

    if use_train_test and test_start and test_end:
        # Run on test period
        result = engine.run(
            df_all=df_main,
            df_daily_atr=df_daily,
            df_intraday_fallback=df_intraday_fallback,
            start_date=test_start,
            end_date=test_end,
        )
        period_name = "Test"
    elif use_train_test:
        # Run on training period
        result = engine.run(
            df_all=df_main,
            df_daily_atr=df_daily,
            df_intraday_fallback=df_intraday_fallback,
            start_date=train_start,
            end_date=train_end,
        )
        period_name = "Training"
    else:
        # Legacy mode
        result = engine.run(
            df_all=df_main,
            df_daily_atr=df_daily,
            df_intraday_fallback=df_intraday_fallback,
        )
        period_name = "Backtest"

    if len(result.trades) == 0:
        print("No trades generated")
        return

    # Get trades DataFrame
    trades_df = result.to_trades_dataframe()

    # Compute metrics
    metrics = result.metrics

    # Print results
    print_metrics_table(metrics, title=f"{period_name.upper()} RESULTS")

    # Print equity curve stats if available
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        equity_series = result.equity_curve.set_index("date")["equity_currency"]
        equity_stats = equity_curve_stats(equity_series)
        print("\n" + "=" * 60)
        print("EQUITY CURVE STATISTICS")
        print("=" * 60)
        print(f"{'Annualized Return':<30} {equity_stats.get('annualized_return', 0.0):>15.2%}")
        print(f"{'Annualized Volatility':<30} {equity_stats.get('annualized_vol', 0.0):>15.2%}")
        print(f"{'Sharpe Ratio':<30} {equity_stats.get('sharpe', 0.0):>15.3f}")
        print(f"{'Max Drawdown':<30} {equity_stats.get('max_drawdown', 0.0):>15.2%}")
        print(f"{'Max Drawdown Duration (days)':<30} {equity_stats.get('max_drawdown_duration', 0.0):>15.0f}")
        print("=" * 60)

    # Print breakdowns
    day_breakdown = breakdown_by_day_of_week(trades_df)
    if len(day_breakdown) > 0:
        print("\nBreakdown by Day of Week:")
        print(day_breakdown.to_string(index=False))

    month_breakdown = breakdown_by_month(trades_df)
    if len(month_breakdown) > 0:
        print("\nBreakdown by Month:")
        print(month_breakdown.to_string(index=False))

    # Save trades
    runs_dir = Path("./runs")
    runs_dir.mkdir(exist_ok=True)

    if use_train_test and test_start and test_end:
        filename = f"{args.instrument}_{test_start}_{test_end}_{args.granularity}.csv"
    elif use_train_test:
        filename = f"{args.instrument}_{train_start}_{train_end}_{args.granularity}.csv"
    else:
        filename = f"{args.instrument}_{data_start}_{data_end}_{args.granularity}.csv"

    filepath = runs_dir / filename
    trades_df.to_csv(filepath, index=False)
    print(f"\nTrades saved to: {filepath}")

    # Save equity curve
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        equity_filename = filename.replace(".csv", "_equity.csv")
        equity_filepath = runs_dir / equity_filename
        result.equity_curve.to_csv(equity_filepath, index=False)
        print(f"Equity curve saved to: {equity_filepath}")


if __name__ == "__main__":
    main()
