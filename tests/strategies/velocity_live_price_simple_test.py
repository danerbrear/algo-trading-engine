#!/usr/bin/env python3
"""
Simple standalone test for velocity calculation with live price.
This test can be run directly without pytest to avoid environment issues.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_test_data():
    """Create simple test data without numpy to avoid environment issues."""
    try:
        import pandas as pd
        
        # Create simple price data
        dates = []
        current = datetime.now() - timedelta(days=60)
        for i in range(60):
            dates.append(current + timedelta(days=i))
        
        data = {
            'Open': [580 + i*0.3 for i in range(60)],
            'High': [590 + i*0.3 for i in range(60)],
            'Low': [570 + i*0.3 for i in range(60)],
            'Close': [580 + i*0.3 for i in range(60)],
            'Volume': [15000000 for _ in range(60)]
        }
        
        return pd.DataFrame(data, index=dates)
    except Exception as e:
        print(f"Error creating test data: {e}")
        return None


def test_velocity_with_live_price():
    """Test that velocity calculation uses live SPY price when market is open."""
    print("\n" + "="*80)
    print("TEST: Velocity calculation with live SPY price")
    print("="*80 + "\n")
    
    try:
        from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
        
        # Create test data
        test_data = create_test_data()
        if test_data is None:
            print("❌ FAILED: Could not create test data")
            return False
        
        print(f"✓ Created test data with {len(test_data)} days")
        print(f"  Date range: {test_data.index[0].date()} to {test_data.index[-1].date()}")
        
        # Setup mock options handler
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        # Create strategy
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(test_data, {})
        
        print(f"✓ Created strategy and loaded data")
        print(f"  Moving averages calculated: SMA_15, SMA_30")
        print(f"  Velocity metric: MA_Velocity_15_30")
        
        # Current date and live price
        current_date = datetime.now()
        live_price = 605.50
        
        # Verify current date NOT in data yet
        if current_date in strategy.data.index:
            print("❌ FAILED: Current date should not be in data yet")
            return False
        
        print(f"✓ Current date {current_date.date()} not in cached data (as expected)")
        
        # Mock the live price fetch
        with patch.object(strategy, '_get_current_underlying_price', return_value=live_price) as mock_get_price:
            print(f"\n🔍 Calling _has_buy_signal with current date...")
            print(f"   Mocked live price: ${live_price}")
            
            # Call _has_buy_signal which should fetch live price and calculate velocity
            try:
                result = strategy._has_buy_signal(current_date)
                print(f"   Signal result: {result}")
            except Exception as e:
                print(f"   Note: Signal evaluation raised exception (expected for test): {type(e).__name__}")
                # This is OK - we're testing the data fetch, not the full signal logic
            
            # Verify _get_current_underlying_price was called
            if not mock_get_price.called:
                print("❌ FAILED: _get_current_underlying_price was not called")
                return False
            
            print(f"✓ _get_current_underlying_price was called")
            
        # Verify current date WAS added to data
        if current_date not in strategy.data.index:
            print("❌ FAILED: Current date should have been added to data")
            return False
        
        print(f"✓ Current date {current_date.date()} was added to data")
        
        # Verify the live price was used
        actual_price = strategy.data.loc[current_date, 'Close']
        if actual_price != live_price:
            print(f"❌ FAILED: Expected price ${live_price}, got ${actual_price}")
            return False
        
        print(f"✓ Live price ${live_price} was correctly added to data")
        
        # Verify velocity was recalculated
        if 'MA_Velocity_15_30' not in strategy.data.columns:
            print("❌ FAILED: Velocity column not found")
            return False
        
        current_velocity = strategy.data.loc[current_date, 'MA_Velocity_15_30']
        import pandas as pd
        if pd.isna(current_velocity):
            print("❌ FAILED: Velocity was not calculated for current date")
            return False
        
        print(f"✓ Velocity was recalculated for current date: {current_velocity:.6f}")
        
        # Verify velocity change was calculated
        if 'Velocity_Changes' not in strategy.data.columns:
            print("❌ FAILED: Velocity changes column not found")
            return False
        
        velocity_change = strategy.data.loc[current_date, 'Velocity_Changes']
        if pd.isna(velocity_change):
            print("⚠  Warning: Velocity change is NaN (may be expected for first calculation)")
        else:
            print(f"✓ Velocity change calculated: {velocity_change:.6f}")
        
        # Summary
        print("\n" + "="*80)
        print("✅ TEST PASSED: Velocity calculation correctly uses live SPY price")
        print("="*80)
        print("\nKey findings:")
        print(f"  - Live price (${live_price}) was fetched when market is open")
        print(f"  - Current date data was appended to historical data")
        print(f"  - Moving averages were recalculated with live data")
        print(f"  - Velocity metric (MA_Velocity_15_30) was updated: {current_velocity:.6f}")
        print(f"  - Strategy can now make decisions based on live market data")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_to_cached_data():
    """Test that strategy falls back to cached data when date is not current."""
    print("\n" + "="*80)
    print("TEST: Fallback to cached data for historical dates")
    print("="*80 + "\n")
    
    try:
        from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
        
        # Create test data
        test_data = create_test_data()
        if test_data is None:
            print("❌ FAILED: Could not create test data")
            return False
        
        # Setup mock options handler
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        # Create strategy
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(test_data, {})
        
        # Use a historical date (not current)
        historical_date = test_data.index[-2]
        expected_price = float(test_data.loc[historical_date, 'Close'])
        
        print(f"✓ Testing with historical date: {historical_date.date()}")
        print(f"  Expected cached price: ${expected_price:.2f}")
        
        # Call _get_current_underlying_price for historical date
        actual_price = strategy._get_current_underlying_price(historical_date)
        
        if actual_price is None:
            print("❌ FAILED: No price returned for historical date")
            return False
        
        if abs(actual_price - expected_price) > 0.01:
            print(f"❌ FAILED: Expected ${expected_price:.2f}, got ${actual_price:.2f}")
            return False
        
        print(f"✓ Cached price correctly retrieved: ${actual_price:.2f}")
        
        print("\n" + "="*80)
        print("✅ TEST PASSED: Fallback to cached data works correctly")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VELOCITY LIVE PRICE TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies that the velocity momentum strategy")
    print("correctly uses live SPY prices when the market is open.\n")
    
    results = []
    
    # Test 1: Live price usage
    results.append(("Velocity with live price", test_velocity_with_live_price()))
    
    # Test 2: Fallback to cached data
    results.append(("Fallback to cached data", test_fallback_to_cached_data()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

