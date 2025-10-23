"""
Configuration file for OANDA trading system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OANDA API Configuration
OANDA_API_KEY = "3a73969eb501323abc852182d0a3704d-60c8535b2e3b4e09a661f18174077e0a"
OANDA_ACCOUNT_ID = "101-001-37149443-001"  # Demo account ID - replace with your actual account ID
OANDA_ENVIRONMENT = "practice"  # Use "live" for real trading

# API URLs
OANDA_BASE_URL = "https://api-fxpractice.oanda.com" if OANDA_ENVIRONMENT == "practice" else "https://api-fxtrade.oanda.com"
OANDA_STREAM_URL = "https://stream-fxpractice.oanda.com" if OANDA_ENVIRONMENT == "practice" else "https://stream-fxtrade.oanda.com"

# Trading Configuration
DEFAULT_INSTRUMENT = "EUR_USD"
DEFAULT_TIMEFRAME = "H1"  # 1 hour candles

# Risk Management
MAX_RISK_PER_TRADE = 0.02  # 2% of account balance
DAILY_LOSS_CAP = 0.05  # 5% of account balance
DEFAULT_STOP_LOSS_PIPS = 50
DEFAULT_TAKE_PROFIT_PIPS = 100

# EMA Strategy Parameters
FAST_EMA_PERIOD = 12
SLOW_EMA_PERIOD = 26

# Backtesting Configuration
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2024-01-01"
INITIAL_CAPITAL = 10000.0
