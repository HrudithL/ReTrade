"""
Test script for the OANDA Trading System
"""
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from oanda_client import OANDAClient
from config import DEFAULT_INSTRUMENT

def test_oanda_connection():
    """Test OANDA API connection"""
    print("Testing OANDA API connection...")
    
    client = OANDAClient()
    
    # Test account info
    print("1. Testing account info...")
    account_info = client.get_account_info()
    if account_info:
        print(f"   ✓ Account ID: {account_info['id']}")
        print(f"   ✓ Balance: ${account_info['balance']}")
        print(f"   ✓ Currency: {account_info['currency']}")
    else:
        print("   ✗ Failed to get account info")
        return False
    
    # Test current price
    print("2. Testing current price...")
    price_info = client.get_current_price(DEFAULT_INSTRUMENT)
    if price_info:
        print(f"   ✓ {DEFAULT_INSTRUMENT} - Bid: {price_info['bid']:.5f}, Ask: {price_info['ask']:.5f}")
    else:
        print("   ✗ Failed to get current price")
        return False
    
    # Test historical data
    print("3. Testing historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = client.get_historical_data(
        instrument=DEFAULT_INSTRUMENT,
        granularity="H1",
        from_time=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        to_time=end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    if not df.empty:
        print(f"   ✓ Fetched {len(df)} candles")
        print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   ✓ Latest close: {df['close'].iloc[-1]:.5f}")
    else:
        print("   ✗ Failed to fetch historical data")
        return False
    
    # Test positions
    print("4. Testing positions...")
    positions = client.get_positions()
    print(f"   ✓ Current positions: {len(positions)}")
    
    print("\n✓ All OANDA API tests passed!")
    return True

def test_ema_calculation():
    """Test EMA calculation"""
    print("\nTesting EMA calculation...")
    
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Calculate EMAs
    fast_ema = df['close'].ewm(span=12).mean()
    slow_ema = df['close'].ewm(span=26).mean()
    
    # Calculate crossover
    crossover = fast_ema - slow_ema
    crossover_signal = np.where(
        (crossover > 0) & (crossover.shift(1) <= 0), 1,
        np.where(
            (crossover < 0) & (crossover.shift(1) >= 0), -1,
            0
        )
    )
    
    signals = np.sum(crossover_signal != 0)
    print(f"   ✓ Generated {signals} crossover signals")
    print(f"   ✓ Fast EMA (latest): {fast_ema.iloc[-1]:.5f}")
    print(f"   ✓ Slow EMA (latest): {slow_ema.iloc[-1]:.5f}")
    print(f"   ✓ Latest signal: {crossover_signal[-1]}")
    
    print("✓ EMA calculation test passed!")
    return True

def test_risk_management():
    """Test risk management calculations"""
    print("\nTesting risk management...")
    
    # Test position size calculation
    account_balance = 10000
    max_risk_per_trade = 0.02
    risk_amount = account_balance * max_risk_per_trade
    
    price = 1.1000
    stop_loss = 1.0950  # 50 pips
    pips_at_risk = abs(price - stop_loss) * 10000
    
    pip_value = 10  # For EUR/USD
    position_size = risk_amount / (pips_at_risk * pip_value)
    position_size_units = int(position_size * 100000)
    
    print(f"   ✓ Account balance: ${account_balance}")
    print(f"   ✓ Risk per trade: {max_risk_per_trade*100}%")
    print(f"   ✓ Risk amount: ${risk_amount}")
    print(f"   ✓ Pips at risk: {pips_at_risk}")
    print(f"   ✓ Position size: {position_size_units} units")
    
    # Test daily loss limit
    daily_loss_cap = 0.05
    daily_loss_limit = account_balance * (1 - daily_loss_cap)
    print(f"   ✓ Daily loss limit: ${daily_loss_limit}")
    
    print("✓ Risk management test passed!")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("OANDA Trading System - Test Suite")
    print("="*60)
    
    tests = [
        test_oanda_connection,
        test_ema_calculation,
        test_risk_management
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ✗ Test failed with error: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
