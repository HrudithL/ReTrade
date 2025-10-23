"""
OANDA API client for data fetching and trading operations
"""
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
from datetime import datetime, timedelta
import time
from config import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL

class OANDAClient:
    def __init__(self):
        self.api = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")
        self.account_id = OANDA_ACCOUNT_ID
        
    def get_historical_data(self, instrument, count=500, granularity="H1", from_time=None, to_time=None):
        """
        Fetch historical candle data from OANDA
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            count: Number of candles to fetch
            granularity: Timeframe (H1, H4, D, etc.)
            from_time: Start time (ISO format)
            to_time: End time (ISO format)
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M"  # Mid prices
        }
        
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
            
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.api.request(r)
            
            data = []
            for candle in r.response['candles']:
                if candle['complete']:
                    data.append({
                        'datetime': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_account_info(self):
        """Get account information including balance"""
        try:
            r = accounts.AccountDetails(accountID=self.account_id)
            self.api.request(r)
            return r.response['account']
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return None
    
    def get_current_price(self, instrument):
        """Get current price for an instrument"""
        try:
            params = {"instruments": instrument}
            r = instruments.InstrumentsPricing(accountID=self.account_id, params=params)
            self.api.request(r)
            
            price_data = r.response['prices'][0]
            return {
                'bid': float(price_data['bids'][0]['price']),
                'ask': float(price_data['asks'][0]['price']),
                'mid': (float(price_data['bids'][0]['price']) + float(price_data['asks'][0]['price'])) / 2
            }
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def place_market_order(self, instrument, units, stop_loss=None, take_profit=None):
        """
        Place a market order
        
        Args:
            instrument: Currency pair
            units: Number of units (positive for buy, negative for sell)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response or None if failed
        """
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
        
        try:
            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            self.api.request(r)
            return r.response
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            self.api.request(r)
            return r.response['positions']
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    def close_position(self, instrument):
        """Close all positions for an instrument"""
        try:
            r = positions.PositionClose(accountID=self.account_id, instrument=instrument)
            self.api.request(r)
            return r.response
        except Exception as e:
            print(f"Error closing position: {e}")
            return None
