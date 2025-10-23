"""
Backtesting engine using Backtrader
"""
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from oanda_client import OANDAClient
from ema_strategy import EMAStrategy
from config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE, INITIAL_CAPITAL,
    DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME
)

class BacktestEngine:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.cerebro = bt.Cerebro()
        self.initial_capital = initial_capital
        self.oanda_client = OANDAClient()
        
    def add_data(self, instrument=DEFAULT_INSTRUMENT, timeframe=DEFAULT_TIMEFRAME):
        """Add historical data to the backtest"""
        # Fetch historical data from OANDA
        print(f"Fetching historical data for {instrument}...")
        
        # Convert timeframe to OANDA format
        oanda_timeframe = self._convert_timeframe(timeframe)
        
        # Fetch data
        df = self.oanda_client.get_historical_data(
            instrument=instrument,
            granularity=oanda_timeframe,
            from_time=BACKTEST_START_DATE + "T00:00:00Z",
            to_time=BACKTEST_END_DATE + "T23:59:59Z"
        )
        
        if df.empty:
            print("No data fetched. Please check your API key and dates.")
            return False
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Create Backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed)
        return True
    
    def _convert_timeframe(self, timeframe):
        """Convert Backtrader timeframe to OANDA format"""
        timeframe_map = {
            'M1': 'M1',
            'M5': 'M5',
            'M15': 'M15',
            'M30': 'M30',
            'H1': 'H1',
            'H4': 'H4',
            'D': 'D',
            'W': 'W',
            'M': 'M'
        }
        return timeframe_map.get(timeframe, 'H1')
    
    def add_strategy(self, strategy_class=EMAStrategy, **kwargs):
        """Add trading strategy to the backtest"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_analyzers(self):
        """Add performance analyzers"""
        # Add Sharpe Ratio analyzer
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        # Add DrawDown analyzer
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # Add Trade analyzer
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Add Returns analyzer
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Add TimeReturn analyzer
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    def run_backtest(self):
        """Run the backtest"""
        # Set initial capital
        self.cerebro.broker.setcash(self.initial_capital)
        
        # Set commission (0.1% per trade)
        self.cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        self.add_analyzers()
        
        print(f"Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}")
        
        # Run the backtest
        results = self.cerebro.run()
        
        print(f"Final Portfolio Value: {self.cerebro.broker.getvalue():.2f}")
        
        return results
    
    def plot_results(self, save_plot=True):
        """Plot backtest results"""
        try:
            # Plot the results
            self.cerebro.plot(style='candlestick', barup='green', bardown='red')
            
            if save_plot:
                plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
                print("Plot saved as 'backtest_results.png'")
            
            plt.show()
        except Exception as e:
            print(f"Error plotting results: {e}")
    
    def print_analysis(self, results):
        """Print detailed analysis of backtest results"""
        if not results:
            print("No results to analyze")
            return
        
        strategy = results[0]
        
        print("\n" + "="*50)
        print("BACKTEST ANALYSIS")
        print("="*50)
        
        # Portfolio performance
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Sharpe Ratio
        sharpe = strategy.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe:
            print(f"Sharpe Ratio: {sharpe['sharperatio']:.2f}")
        
        # DrawDown
        drawdown = strategy.analyzers.drawdown.get_analysis()
        if 'max' in drawdown:
            print(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
            print(f"Max Drawdown Duration: {drawdown['max']['len']} periods")
        
        # Trade Analysis
        trades = strategy.analyzers.trades.get_analysis()
        if 'total' in trades:
            total_trades = trades['total']['total']
            won_trades = trades['won']['total']
            lost_trades = trades['lost']['total']
            
            print(f"\nTrade Statistics:")
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {won_trades}")
            print(f"Losing Trades: {lost_trades}")
            
            if total_trades > 0:
                win_rate = (won_trades / total_trades) * 100
                print(f"Win Rate: {win_rate:.2f}%")
                
                if 'won' in trades and 'pnl' in trades['won']:
                    avg_win = trades['won']['pnl']['average']
                    print(f"Average Win: ${avg_win:.2f}")
                
                if 'lost' in trades and 'pnl' in trades['lost']:
                    avg_loss = trades['lost']['pnl']['average']
                    print(f"Average Loss: ${avg_loss:.2f}")
        
        # Returns
        returns = strategy.analyzers.returns.get_analysis()
        if 'rtot' in returns:
            print(f"Total Return: {returns['rtot']:.2f}%")
        if 'rnorm' in returns:
            print(f"Normalized Return: {returns['rnorm']:.2f}%")
        
        print("="*50)

def run_backtest():
    """Main function to run backtest"""
    # Create backtest engine
    engine = BacktestEngine()
    
    # Add data
    if not engine.add_data():
        return
    
    # Add strategy
    engine.add_strategy(EMAStrategy)
    
    # Run backtest
    results = engine.run_backtest()
    
    # Print analysis
    engine.print_analysis(results)
    
    # Plot results
    engine.plot_results()
    
    return results

if __name__ == "__main__":
    run_backtest()
