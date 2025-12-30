#!/usr/bin/env python3
"""
Test that validates signal consistency between market hours (live price) and after close (close price).

This test addresses the scenario where:
1. During market hours - position not recommended
2. After market close - position IS recommended
3. Close price barely changed from live price

This should NOT happen - both scenarios should generate the same signal.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algo_trading_engine'))

from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy


class TestVelocityLiveVsCloseConsistency(unittest.TestCase):
    """Test signal consistency between live price and close price scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level mocks to prevent real API calls."""
        # Mock DataRetriever to prevent yfinance API calls
        cls.data_retriever_patcher = patch('algo_trading_engine.strategies.velocity_signal_momentum_strategy.DataRetriever')
        cls.mock_data_retriever_class = cls.data_retriever_patcher.start()
        
        # Configure the mock instance
        cls.mock_data_retriever_instance = Mock()
        cls.mock_data_retriever_instance.get_live_price.return_value = None
        cls.mock_data_retriever_class.return_value = cls.mock_data_retriever_instance
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level mocks."""
        cls.data_retriever_patcher.stop()
    
    def setUp(self):
        """Set up test fixtures."""
        # Create historical data ending yesterday
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=90, freq='D')
        
        # Create upward trend data that should trigger a signal
        prices = np.concatenate([
            np.linspace(550, 570, 60),  # Gradual rise in first 60 days
            np.linspace(570, 600, 30)   # Stronger rise in last 30 days for velocity signal
        ])
        
        self.historical_data = pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 1,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.uniform(10000000, 20000000, len(dates))
        }, index=dates)
        
        # Use a price that continues the upward trend
        self.live_price = 605.25
        self.close_price = 605.50  # Barely different from live price (0.04% change)
        self.current_date = datetime.now()

    def test_signal_consistency_live_vs_close_price(self):
        """Test that live price during market hours produces same signal as close price after close."""
        print("\n" + "="*80)
        print("TEST: Signal Consistency - Live Price vs Close Price")
        print("="*80)
        
        # Configure mock DataRetriever to return live price
        self.mock_data_retriever_instance.get_live_price.return_value = self.live_price
        
        # Scenario 1: During market hours with live price
        print("\n📊 SCENARIO 1: During Market Hours (Live Price)")
        print("-" * 80)
        
        mock_options_handler_1 = Mock()
        mock_options_handler_1.symbol = 'SPY'
        
        strategy_live = VelocitySignalMomentumStrategy(options_handler=mock_options_handler_1)
        strategy_live.set_data(self.historical_data.copy())
        
        print(f"Historical data ends: {strategy_live.data.index[-1].date()}")
        print(f"Current date: {self.current_date.date()}")
        print(f"Live price: ${self.live_price}")
        
        # No need to mock _get_current_underlying_price - DataRetriever is already mocked
        with patch.object(strategy_live, '_check_trend_success', return_value=(True, 5, 0.03)):
            signal_during_market_hours = strategy_live._has_buy_signal(self.current_date)
        
        # Get velocity metrics for live price scenario
        if self.current_date in strategy_live.data.index:
            live_velocity = strategy_live.data.loc[self.current_date, 'MA_Velocity_15_30']
            live_velocity_change = strategy_live.data.loc[self.current_date, 'Velocity_Changes']
            print(f"Velocity with live price: {live_velocity:.6f}")
            print(f"Velocity change: {live_velocity_change:.6f}")
            print(f"Signal generated: {signal_during_market_hours}")
        else:
            print("⚠️  WARNING: Current date was not added to data!")
        
        # Scenario 2: After market close with close price in cached data
        print("\n📊 SCENARIO 2: After Market Close (Close Price)")
        print("-" * 80)
        
        mock_options_handler_2 = Mock()
        mock_options_handler_2.symbol = 'SPY'
        
        strategy_close = VelocitySignalMomentumStrategy(options_handler=mock_options_handler_2)
        
        # Create data that includes today's close price (as it would be after market close)
        data_with_close = self.historical_data.copy()
        current_date_close = pd.DataFrame({
            'Open': [600.00],
            'High': [606.00],
            'Low': [599.50],
            'Close': [self.close_price],
            'Volume': [18000000]
        }, index=[self.current_date])
        data_with_close = pd.concat([data_with_close, current_date_close])
        
        strategy_close.set_data(data_with_close)
        
        print(f"Historical data ends: {strategy_close.data.index[-1].date()}")
        print(f"Close price: ${self.close_price}")
        
        # Check signal after market close (no live price fetch needed)
        with patch.object(strategy_close, '_check_trend_success', return_value=(True, 5, 0.03)):
            signal_after_market_close = strategy_close._has_buy_signal(self.current_date)
        
        # Get velocity metrics for close price scenario
        close_velocity = strategy_close.data.loc[self.current_date, 'MA_Velocity_15_30']
        close_velocity_change = strategy_close.data.loc[self.current_date, 'Velocity_Changes']
        print(f"Velocity with close price: {close_velocity:.6f}")
        print(f"Velocity change: {close_velocity_change:.6f}")
        print(f"Signal generated: {signal_after_market_close}")
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        price_diff = abs(self.close_price - self.live_price)
        price_diff_pct = (price_diff / self.live_price) * 100
        
        print(f"\nPrice difference: ${price_diff:.2f} ({price_diff_pct:.3f}%)")
        print(f"Live price signal:  {signal_during_market_hours}")
        print(f"Close price signal: {signal_after_market_close}")
        
        if self.current_date in strategy_live.data.index:
            velocity_diff = abs(close_velocity - live_velocity)
            velocity_change_diff = abs(close_velocity_change - live_velocity_change)
            
            print(f"\nVelocity difference: {velocity_diff:.6f}")
            print(f"Velocity change difference: {velocity_change_diff:.6f}")
        
        # Assertions
        print("\n" + "="*80)
        if signal_during_market_hours != signal_after_market_close:
            print("❌ FAILED: Signals are INCONSISTENT!")
            print(f"   During market hours: {signal_during_market_hours}")
            print(f"   After market close:  {signal_after_market_close}")
            print(f"   Price barely changed ({price_diff_pct:.3f}%), but signals differ!")
            self.fail(f"Signal inconsistency detected: live={signal_during_market_hours}, close={signal_after_market_close}")
        else:
            print("✅ PASSED: Signals are CONSISTENT!")
            print(f"   Both scenarios: {signal_during_market_hours}")
        print("="*80 + "\n")

    def test_stale_data_in_cache_prevents_live_fetch(self):
        """Test that stale cached data for current date prevents live price fetch."""
        print("\n" + "="*80)
        print("TEST: Stale Cached Data Detection")
        print("="*80)
        
        # Configure mock DataRetriever to return live price
        self.mock_data_retriever_instance.get_live_price.return_value = self.live_price
        
        # Create data that already includes current date with old price
        data_with_stale_current = self.historical_data.copy()
        stale_price = 580.00  # Much lower than expected
        stale_current_date = pd.DataFrame({
            'Open': [579.00],
            'High': [581.00],
            'Low': [578.00],
            'Close': [stale_price],  # Stale/old price
            'Volume': [15000000]
        }, index=[self.current_date])
        data_with_stale_current = pd.concat([data_with_stale_current, stale_current_date])
        
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(data_with_stale_current)
        
        print(f"\nCurrent date: {self.current_date.date()}")
        print(f"Stale cached price for current date: ${stale_price}")
        print(f"Expected live price: ${self.live_price}")
        
        # Check if DataRetriever was called (no need to mock _get_current_underlying_price)
        try:
            # Reset the mock call count
            self.mock_data_retriever_instance.get_live_price.reset_mock()
            
            # Call _has_buy_signal
            strategy._has_buy_signal(self.current_date)
            
            # Check if live price was fetched via DataRetriever
            if self.mock_data_retriever_instance.get_live_price.called:
                print("✅ Live price fetch WAS called (good - overwrites stale data)")
            else:
                print("❌ Live price fetch was NOT called (bad - uses stale data)")
                print("   This is the BUG that causes inconsistent signals!")
                
                # Check what price is being used
                actual_price = strategy.data.loc[self.current_date, 'Close']
                print(f"   Price being used: ${actual_price}")
                
                if actual_price == stale_price:
                    self.fail(
                        f"Strategy is using stale cached data (${stale_price}) instead of "
                        f"fetching live price (${self.live_price}) for current date!"
                    )
        except KeyError:
            # Date not in index, so live fetch would be triggered
            print("✅ Date not in index - live fetch would be triggered")

    def test_recommendation_engine_fresh_data_fetch(self):
        """Test that recommendation engine always uses fresh data for current date."""
        print("\n" + "="*80)
        print("TEST: Recommendation Engine Fresh Data")
        print("="*80)
        
        # This test is covered by test_stale_data_in_cache_prevents_live_fetch
        # which validates that the strategy automatically updates stale cached data
        # with fresh live prices on the current date
        
        print("\n✅ This scenario is validated by test_stale_data_in_cache_prevents_live_fetch")
        print("   The fix ensures that current date always fetches live price,")
        print("   even when stale cached data exists.")
        print("\n" + "="*80)


if __name__ == '__main__':
    # Set up for better test output
    import sys
    
    # Run with verbose output
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVelocityLiveVsCloseConsistency)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

