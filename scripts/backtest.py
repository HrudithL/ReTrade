#!/usr/bin/env python3
"""CLI script for running Opening Range Breakout backtests."""

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.config.settings import Settings
from src.data.fetch import fetch_daily_for_atr, fetch_historical_data, resample_to_daily
from src.utils.timezones import to_ny


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest Opening Range Breakout Strategy")
    parser.add_argument("--instrument", type=str, required=True, help="OANDA instrument (e.g., US500_USD)")
    parser.add_argument("--granularity", type=str, default="M5", help="Candle granularity (default: M5)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--use_mid_stop",
        action="store_true",
        help="Use OR mid as stop instead of opposite boundary (default: False)",
    )
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-reward ratio (default: 2.0)")
    parser.add_argument("--atr_tf", type=str, default="M15", help="ATR timeframe hint (default: M15, uses daily)")
    parser.add_argument("--atr_mult", type=float, default=1.2, help="ATR multiplier (default: 1.2)")
    parser.add_argument("--tz", type=str, default="America/New_York", help="Timezone (default: America/New_York)")
    parser.add_argument("--no_cache", action="store_true", help="Disable cache")

    args = parser.parse_args()

    # Validate settings
    Settings.validate()

    print(f"Backtesting {args.instrument} from {args.start} to {args.end}")
    print(f"Granularity: {args.granularity}, ATR uses daily candles")
    print(f"RR: {args.rr}, ATR Mult: {args.atr_mult}, Use Mid Stop: {args.use_mid_stop}")
    print()

    # Fetch main data
    print("Fetching main timeframe data...")
    df_main = fetch_historical_data(
        args.instrument,
        args.granularity,
        args.start,
        args.end,
        use_cache=not args.no_cache,
    )

    if len(df_main) == 0:
        print("Error: No data fetched")
        return

    # Convert to NY timezone
    df_main = to_ny(df_main)

    # Fetch daily candles for ATR calculation
    print("Fetching daily candles for ATR calculation...")
    df_daily = fetch_daily_for_atr(
        args.instrument,
        args.start,
        args.end,
        use_cache=not args.no_cache,
    )

    # Prepare fallback data if daily unavailable
    df_intraday_fallback = None
    if df_daily is None or len(df_daily) == 0:
        print("Daily candles unavailable, will resample intraday data if needed...")
        # Use main data as fallback
        df_intraday_fallback = df_main.copy()
    else:
        # Convert daily to NY timezone
        df_daily = to_ny(df_daily)

    # Build backtest parameters
    params = {
        "instrument": args.instrument,
        "granularity": args.granularity,
        "use_mid_stop": args.use_mid_stop,
        "rr": args.rr,
        "atr_tf": args.atr_tf,
        "atr_mult": args.atr_mult,
        "atr_period": 20,
        "fill_policy": "sl_first",
    }

    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(params)
    trades_df = engine.run(df_main, df_daily_atr=df_daily, df_intraday_fallback=df_intraday_fallback)

    if len(trades_df) == 0:
        print("No trades generated")
        return

    # Compute metrics
    metrics = compute_metrics(trades_df)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average R: {metrics['avg_R']:.3f}")
    print(f"Total R: {metrics['total_R']:.3f}")
    print(f"Expectancy: {metrics['expectancy']:.3f}")
    print(f"Max Drawdown (R): {metrics['max_drawdown_R']:.3f}")
    print(f"Longest Losing Streak: {metrics['longest_loser']}")
    print(f"Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
    print("=" * 60)

    # Save trades
    runs_dir = Path("./runs")
    runs_dir.mkdir(exist_ok=True)

    filename = f"{args.instrument}_{args.start}_{args.end}_{args.granularity}.csv"
    filepath = runs_dir / filename

    trades_df.to_csv(filepath, index=False)
    print(f"\nTrades saved to: {filepath}")


if __name__ == "__main__":
    main()

