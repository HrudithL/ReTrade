"""
Paper Trading System for OANDA Demo Account
"""
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

from oanda_client import OANDAClient
from config import (
    DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME, FAST_EMA_PERIOD, SLOW_EMA_PERIOD,
    DEFAULT_STOP_LOSS_PIPS, DEFAULT_TAKE_PROFIT_PIPS, MAX_RISK_PER_TRADE, DAILY_LOSS_CAP
)

class PaperTradingSystem:
    def __init__(self, instrument=DEFAULT_INSTRUMENT):
        self.instrument = instrument
        self.oanda_client = OANDAClient()
        self.running = False
        
        # Strategy parameters
        self.fast_ema_period = FAST_EMA_PERIOD
        self.slow_ema_period = SLOW_EMA_PERIOD
        self.stop_loss_pips = DEFAULT_STOP_LOSS_PIPS
        self.take_profit_pips = DEFAULT_TAKE_PROFIT_PIPS
        self.max_risk_per_trade = MAX_RISK_PER_TRADE
        self.daily_loss_cap = DAILY_LOSS_CAP
        
        # Data storage
        self.price_data = pd.DataFrame()
        self.indicators = {}
        
        # Position tracking
        self.current_position = None
        self.daily_start_balance = None
        self.daily_loss_limit = None
        self.current_date = None
        
        # Trade history
        self.trade_history = []
        
    def log(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def fetch_latest_data(self, count=100):
        """Fetch latest price data"""
        try:
            df = self.oanda_client.get_historical_data(
                instrument=self.instrument,
                count=count,
                granularity="H1"
            )
            
            if not df.empty:
                self.price_data = df
                self.calculate_indicators()
                return True
            return False
            
        except Exception as e:
            self.log(f"Error fetching data: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate EMA indicators"""
        if len(self.price_data) < self.slow_ema_period:
            return
        
        # Calculate EMAs
        self.indicators['fast_ema'] = self.price_data['close'].ewm(span=self.fast_ema_period).mean()
        self.indicators['slow_ema'] = self.price_data['close'].ewm(span=self.slow_ema_period).mean()
        
        # Calculate crossover signals
        self.indicators['crossover'] = self.indicators['fast_ema'] - self.indicators['slow_ema']
        self.indicators['crossover_signal'] = np.where(
            (self.indicators['crossover'] > 0) & (self.indicators['crossover'].shift(1) <= 0), 1,  # Bullish crossover
            np.where(
                (self.indicators['crossover'] < 0) & (self.indicators['crossover'].shift(1) >= 0), -1,  # Bearish crossover
                0  # No signal
            )
        )
    
    def get_current_price(self):
        """Get current market price"""
        try:
            price_info = self.oanda_client.get_current_price(self.instrument)
            if price_info:
                return price_info['mid']
            return None
        except Exception as e:
            self.log(f"Error getting current price: {e}")
            return None
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            account_info = self.oanda_client.get_account_info()
            if account_info:
                return float(account_info['balance'])
            return None
        except Exception as e:
            self.log(f"Error getting account balance: {e}")
            return None
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk management"""
        account_balance = self.get_account_balance()
        if not account_balance:
            return 0
        
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate pip value (assuming 1 lot = 100,000 units)
        pip_value = 10  # For EUR/USD, 1 pip = $10 per lot
        pips_at_risk = abs(price - stop_loss_price) * 10000  # Convert to pips
        
        if pips_at_risk == 0:
            return 0
        
        # Calculate position size in lots
        position_size = risk_amount / (pips_at_risk * pip_value)
        
        # Convert to units (1 lot = 100,000 units)
        position_size_units = int(position_size * 100000)
        
        # Ensure minimum position size
        if abs(position_size_units) < 1000:
            position_size_units = 1000 if position_size_units > 0 else -1000
        
        return position_size_units
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit has been reached"""
        current_date = datetime.now().date()
        
        # Reset daily tracking on new day
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_balance = self.get_account_balance()
            if self.daily_start_balance:
                self.daily_loss_limit = self.daily_start_balance * (1 - self.daily_loss_cap)
        
        # Check if daily loss limit reached
        if self.daily_loss_limit:
            current_balance = self.get_account_balance()
            if current_balance and current_balance <= self.daily_loss_limit:
                self.log(f"DAILY LOSS LIMIT REACHED - Closing all positions")
                self.close_all_positions()
                return True
        
        return False
    
    def get_signal(self):
        """Get current trading signal"""
        if len(self.indicators) == 0 or len(self.price_data) < 2:
            return 0
        
        # Get the latest crossover signal
        latest_signal = self.indicators['crossover_signal'].iloc[-1]
        return latest_signal
    
    def place_order(self, signal, price):
        """Place order based on signal"""
        if signal == 0:
            return
        
        # Calculate stop loss and take profit
        if signal > 0:  # Buy signal
            stop_loss = price - (self.stop_loss_pips / 10000)
            take_profit = price + (self.take_profit_pips / 10000)
            units = self.calculate_position_size(price, stop_loss)
        else:  # Sell signal
            stop_loss = price + (self.stop_loss_pips / 10000)
            take_profit = price - (self.take_profit_pips / 10000)
            units = -self.calculate_position_size(price, stop_loss)
        
        if units == 0:
            self.log("Position size calculated as 0, skipping trade")
            return
        
        # Place order
        order_result = self.oanda_client.place_market_order(
            instrument=self.instrument,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order_result:
            self.log(f"Order placed successfully: {signal > 0 and 'BUY' or 'SELL'} "
                    f"{abs(units)} units at {price:.5f}")
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'signal': signal,
                'price': price,
                'units': units,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': order_result.get('orderFillTransaction', {}).get('id', 'N/A')
            })
        else:
            self.log("Failed to place order")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            result = self.oanda_client.close_position(self.instrument)
            if result:
                self.log("All positions closed")
            else:
                self.log("Failed to close positions")
        except Exception as e:
            self.log(f"Error closing positions: {e}")
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.oanda_client.get_positions()
            return positions
        except Exception as e:
            self.log(f"Error getting positions: {e}")
            return []
    
    def run_trading_loop(self, check_interval=300):  # 5 minutes
        """Main trading loop"""
        self.log("Starting paper trading system...")
        self.running = True
        
        while self.running:
            try:
                # Check daily loss limit
                if self.check_daily_loss_limit():
                    self.log("Daily loss limit reached. Stopping trading.")
                    break
                
                # Fetch latest data
                if not self.fetch_latest_data():
                    self.log("Failed to fetch data, retrying in 1 minute...")
                    time.sleep(60)
                    continue
                
                # Get current price
                current_price = self.get_current_price()
                if not current_price:
                    self.log("Failed to get current price, retrying in 1 minute...")
                    time.sleep(60)
                    continue
                
                # Get trading signal
                signal = self.get_signal()
                
                # Check if we have an open position
                positions = self.get_positions()
                has_position = len(positions) > 0
                
                # Execute trading logic
                if signal != 0 and not has_position:
                    # New signal and no position - place order
                    self.place_order(signal, current_price)
                elif has_position and signal != 0:
                    # We have a position and got a new signal - check if it's opposite
                    current_position = positions[0]
                    position_side = 1 if float(current_position['long']['units']) > 0 else -1
                    
                    if signal != position_side:
                        # Opposite signal - close position and place new order
                        self.log("Opposite signal received, closing position and placing new order")
                        self.close_all_positions()
                        time.sleep(2)  # Wait for position to close
                        self.place_order(signal, current_price)
                
                # Log current status
                balance = self.get_account_balance()
                self.log(f"Balance: ${balance:.2f}, Price: {current_price:.5f}, "
                        f"Signal: {signal}, Position: {'Yes' if has_position else 'No'}")
                
                # Wait before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.log("Trading stopped by user")
                break
            except Exception as e:
                self.log(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.running = False
        self.log("Paper trading system stopped")
    
    def stop_trading(self):
        """Stop the trading system"""
        self.running = False
    
    def print_trade_summary(self):
        """Print summary of all trades"""
        if not self.trade_history:
            self.log("No trades executed")
            return
        
        self.log(f"\nTrade Summary ({len(self.trade_history)} trades):")
        self.log("-" * 80)
        
        for i, trade in enumerate(self.trade_history, 1):
            self.log(f"Trade {i}: {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"{'BUY' if trade['signal'] > 0 else 'SELL'} {abs(trade['units'])} units "
                    f"at {trade['price']:.5f}")

def main():
    """Main function to run paper trading"""
    # Create paper trading system
    trading_system = PaperTradingSystem()
    
    try:
        # Run trading loop
        trading_system.run_trading_loop()
    except KeyboardInterrupt:
        trading_system.stop_trading()
    finally:
        # Print trade summary
        trading_system.print_trade_summary()

if __name__ == "__main__":
    main()
