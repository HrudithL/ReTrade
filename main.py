"""
Main entry point for the OANDA Trading System
"""
import sys
import argparse
from datetime import datetime

from backtest_engine import run_backtest
from paper_trading import PaperTradingSystem

def main():
    parser = argparse.ArgumentParser(description='OANDA Trading System')
    parser.add_argument('mode', choices=['backtest', 'paper'], 
                       help='Trading mode: backtest or paper')
    parser.add_argument('--instrument', default='EUR_USD', 
                       help='Trading instrument (default: EUR_USD)')
    parser.add_argument('--timeframe', default='H1', 
                       help='Timeframe for backtesting (default: H1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OANDA Trading System")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Instrument: {args.instrument}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if args.mode == 'backtest':
        print("Running backtest...")
        try:
            results = run_backtest()
            if results:
                print("\nBacktest completed successfully!")
            else:
                print("\nBacktest failed!")
        except Exception as e:
            print(f"Backtest error: {e}")
    
    elif args.mode == 'paper':
        print("Starting paper trading...")
        try:
            trading_system = PaperTradingSystem(instrument=args.instrument)
            trading_system.run_trading_loop()
        except KeyboardInterrupt:
            print("\nPaper trading stopped by user")
        except Exception as e:
            print(f"Paper trading error: {e}")

if __name__ == "__main__":
    main()
