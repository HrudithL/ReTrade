"""
EMA Crossover Strategy for Backtrader
"""
import backtrader as bt
import numpy as np
from config import FAST_EMA_PERIOD, SLOW_EMA_PERIOD, DEFAULT_STOP_LOSS_PIPS, DEFAULT_TAKE_PROFIT_PIPS

class EMAStrategy(bt.Strategy):
    params = (
        ('fast_ema', FAST_EMA_PERIOD),
        ('slow_ema', SLOW_EMA_PERIOD),
        ('stop_loss_pips', DEFAULT_STOP_LOSS_PIPS),
        ('take_profit_pips', DEFAULT_TAKE_PROFIT_PIPS),
        ('max_risk_per_trade', 0.02),  # 2% risk per trade
        ('daily_loss_cap', 0.05),  # 5% daily loss cap
    )
    
    def __init__(self):
        # Initialize EMAs
        self.fast_ema = bt.indicators.EMA(period=self.params.fast_ema)
        self.slow_ema = bt.indicators.EMA(period=self.params.slow_ema)
        
        # Crossover signals
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
        
        # Track daily P&L for risk management
        self.daily_start_balance = self.broker.getvalue()
        self.current_date = None
        self.daily_loss_limit = None
        
        # Track trade information
        self.order = None
        self.stop_loss_order = None
        self.take_profit_order = None
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.5f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
            else:
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.5f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT - Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit has been reached"""
        current_date = self.datas[0].datetime.date(0)
        
        # Reset daily tracking on new day
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_balance = self.broker.getvalue()
            self.daily_loss_limit = self.daily_start_balance * (1 - self.params.daily_loss_cap)
        
        # Check if daily loss limit reached
        current_balance = self.broker.getvalue()
        if current_balance <= self.daily_loss_limit:
            self.log(f'DAILY LOSS LIMIT REACHED - Closing all positions')
            self.close_all_positions()
            return True
        
        return False
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk management"""
        account_balance = self.broker.getvalue()
        risk_amount = account_balance * self.params.max_risk_per_trade
        
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
    
    def close_all_positions(self):
        """Close all open positions"""
        for data in self.datas:
            if self.getposition(data).size != 0:
                self.close(data)
    
    def next(self):
        """Main strategy logic"""
        # Check daily loss limit first
        if self.check_daily_loss_limit():
            return
        
        # Skip if we have pending orders
        if self.order:
            return
        
        # Get current position
        position = self.getposition()
        
        # No position - look for entry signals
        if not position.size:
            # Bullish crossover - buy signal
            if self.crossover > 0:
                price = self.data.close[0]
                stop_loss = price - (self.params.stop_loss_pips / 10000)
                take_profit = price + (self.params.take_profit_pips / 10000)
                
                # Calculate position size
                size = self.calculate_position_size(price, stop_loss)
                
                if size > 0:
                    self.log(f'BUY SIGNAL - Price: {price:.5f}, Size: {size}, '
                            f'Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}')
                    
                    # Place market order
                    self.order = self.buy(size=size)
                    
                    # Place stop loss and take profit orders
                    self.stop_loss_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_loss)
                    self.take_profit_order = self.sell(size=size, exectype=bt.Order.Limit, price=take_profit)
            
            # Bearish crossover - sell signal
            elif self.crossover < 0:
                price = self.data.close[0]
                stop_loss = price + (self.params.stop_loss_pips / 10000)
                take_profit = price - (self.params.take_profit_pips / 10000)
                
                # Calculate position size
                size = self.calculate_position_size(price, stop_loss)
                
                if size > 0:
                    self.log(f'SELL SIGNAL - Price: {price:.5f}, Size: {size}, '
                            f'Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}')
                    
                    # Place market order
                    self.order = self.sell(size=size)
                    
                    # Place stop loss and take profit orders
                    self.stop_loss_order = self.buy(size=size, exectype=bt.Order.Stop, price=stop_loss)
                    self.take_profit_order = self.buy(size=size, exectype=bt.Order.Limit, price=take_profit)
        
        # We have a position - check for exit signals
        else:
            # Exit on opposite crossover
            if (position.size > 0 and self.crossover < 0) or (position.size < 0 and self.crossover > 0):
                self.log(f'EXIT SIGNAL - Closing position')
                self.close()
                
                # Cancel pending orders
                if self.stop_loss_order:
                    self.cancel(self.stop_loss_order)
                if self.take_profit_order:
                    self.cancel(self.take_profit_order)
