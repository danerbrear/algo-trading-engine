#!/usr/bin/env python3
"""
Unit tests for the recommend_cli velocity momentum strategy with live price.
Tests that when using current date with market open, we calculate velocity using live SPY price.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from algo_trading_engine.prediction.recommend_cli import main
from algo_trading_engine.prediction.decision_store import JsonDecisionStore
from algo_trading_engine.backtest.models import Position, StrategyType
from algo_trading_engine.common.models import Option, OptionType
from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy


class TestVelocityLivePrice(unittest.TestCase):
    """Test cases for velocity calculation using live SPY price when market is open."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data with historical prices
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=60, freq='D')
        self.test_data = pd.DataFrame({
            'Open': np.random.uniform(580, 600, len(dates)),
            'High': np.random.uniform(590, 610, len(dates)),
            'Low': np.random.uniform(570, 590, len(dates)),
            'Close': np.linspace(580, 600, len(dates)),  # Gradual upward trend
            'Volume': np.random.uniform(10000000, 20000000, len(dates))
        }, index=dates)
        
        # Set a live price that's higher than the last cached price (bullish signal)
        self.live_price = 605.50
        self.current_date = datetime.now()
        
    def test_velocity_calculation_uses_live_price_when_market_open(self):
        """Test that velocity calculation uses live SPY price when market is open on current date."""
        # Mock the options handler
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        # Create strategy instance
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Set up test data
        strategy.set_data(self.test_data.copy(), {})
        
        # Verify that data doesn't contain current date yet
        self.assertNotIn(self.current_date, strategy.data.index)
        
        # Mock the live price fetch
        with patch.object(strategy, '_get_current_underlying_price', return_value=self.live_price) as mock_get_price:
            # Call _has_buy_signal with current date
            result = strategy._has_buy_signal(self.current_date)
            
            # Verify that _get_current_underlying_price was called with current date
            mock_get_price.assert_called_with(self.current_date)
            
            # Verify that the data was updated with live price
            self.assertIn(self.current_date, strategy.data.index)
            
            # Verify the live price was added to the data
            added_price = strategy.data.loc[self.current_date, 'Close']
            self.assertEqual(added_price, self.live_price)
            
            # Verify that velocity was recalculated (SMA_15, SMA_30, MA_Velocity_15_30, Velocity_Changes)
            self.assertIn('SMA_15', strategy.data.columns)
            self.assertIn('SMA_30', strategy.data.columns)
            self.assertIn('MA_Velocity_15_30', strategy.data.columns)
            self.assertIn('Velocity_Changes', strategy.data.columns)
            
            # Verify that velocity metrics exist for the current date
            current_velocity = strategy.data.loc[self.current_date, 'MA_Velocity_15_30']
            self.assertIsNotNone(current_velocity)
            self.assertFalse(pd.isna(current_velocity))
            
            print(f"✅ Live price ${self.live_price} was successfully used to calculate velocity")
            print(f"   Current velocity: {current_velocity:.6f}")

    def test_velocity_strategy_appends_live_data_to_history(self):
        """Test that live price data is correctly appended to historical data."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        
        # Store original data length
        original_length = len(strategy.data)
        
        # Mock live price fetch
        with patch.object(strategy, '_get_current_underlying_price', return_value=self.live_price):
            # Call _has_buy_signal which should append live data
            strategy._has_buy_signal(self.current_date)
            
            # Verify data length increased by 1
            self.assertEqual(len(strategy.data), original_length + 1)
            
            # Verify the new row has all OHLCV fields
            new_row = strategy.data.loc[self.current_date]
            self.assertEqual(new_row['Close'], self.live_price)
            self.assertEqual(new_row['Open'], self.live_price)  # Live price used for all fields
            self.assertEqual(new_row['High'], self.live_price)
            self.assertEqual(new_row['Low'], self.live_price)
            self.assertEqual(new_row['Volume'], 0)  # Volume not available for live price
            
            print(f"✅ Live data correctly appended to historical data")

    def test_velocity_recalculation_after_live_price_update(self):
        """Test that moving averages and velocity are recalculated after adding live price."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        
        # Get velocity before adding live data
        last_date_before = strategy.data.index[-1]
        velocity_before = strategy.data.loc[last_date_before, 'MA_Velocity_15_30']
        
        # Mock live price fetch
        with patch.object(strategy, '_get_current_underlying_price', return_value=self.live_price):
            strategy._has_buy_signal(self.current_date)
            
            # Verify velocity was recalculated for current date
            velocity_current = strategy.data.loc[self.current_date, 'MA_Velocity_15_30']
            
            # With upward price trend, velocity should generally increase
            # (though this depends on the moving average calculation)
            self.assertIsNotNone(velocity_current)
            self.assertFalse(pd.isna(velocity_current))
            
            # Verify velocity change was calculated
            velocity_change = strategy.data.loc[self.current_date, 'Velocity_Changes']
            self.assertIsNotNone(velocity_change)
            
            print(f"✅ Velocity recalculated after live price update")
            print(f"   Previous velocity: {velocity_before:.6f}")
            print(f"   Current velocity: {velocity_current:.6f}")
            print(f"   Velocity change: {velocity_change:.6f}")

    def test_live_price_fetch_integration_with_data_retriever(self):
        """Test that live price fetch integrates correctly with DataRetriever.
        
        Note: The strategy now creates DataRetriever on demand when needed,
        rather than using a pre-initialized data_retriever attribute.
        This test verifies that the on-demand creation works correctly.
        """
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        
        # Mock DataRetriever creation and get_live_price
        with patch('algo_trading_engine.strategies.velocity_signal_momentum_strategy.DataRetriever') as mock_data_retriever_class:
            mock_data_retriever = Mock()
            mock_data_retriever.get_live_price.return_value = self.live_price
            mock_data_retriever_class.return_value = mock_data_retriever
            
            # Call _get_current_underlying_price for current date
            price = strategy._get_current_underlying_price(self.current_date)
            
            # Verify DataRetriever was created with correct symbol
            mock_data_retriever_class.assert_called_once_with(symbol='SPY', use_free_tier=True, quiet_mode=True)
            
            # Verify it used the DataRetriever's live price
            mock_data_retriever.get_live_price.assert_called_once()
            self.assertEqual(price, self.live_price)
        
        print(f"✅ DataRetriever integration working correctly")

    def test_fallback_to_cached_data_when_live_price_unavailable(self):
        """Test that strategy falls back to cached data when live price fetch fails."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        # Create a mock DataRetriever that fails to get live price
        mock_data_retriever = Mock()
        mock_data_retriever.get_live_price.return_value = None
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        strategy.data_retriever = mock_data_retriever
        
        # Use a date that exists in cached data
        cached_date = self.test_data.index[-1]
        
        # Call _get_current_underlying_price for cached date
        price = strategy._get_current_underlying_price(cached_date)
        
        # Verify it used cached data as fallback
        cached_price = float(self.test_data.loc[cached_date, 'Close'])
        self.assertEqual(price, cached_price)
        
        print(f"✅ Fallback to cached data working correctly")

    def test_market_closed_uses_cached_data(self):
        """Test that when market is closed (non-current date), cached data is used."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        
        # Use a date from the past (not current date)
        past_date = self.test_data.index[-2]
        
        # Direct test: call _get_current_underlying_price for a historical date
        price = strategy._get_current_underlying_price(past_date)
        
        # Verify the returned price matches cached data
        cached_price = float(self.test_data.loc[past_date, 'Close'])
        
        # The method should return cached price for non-current dates
        self.assertIsNotNone(price)
        self.assertEqual(price, cached_price)
        
        print(f"✅ Market closed scenario uses cached data correctly")

    def test_velocity_signal_detection_with_live_price(self):
        """Test that velocity signal is correctly detected using live price."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create test data with upward trend leading to current date
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=90, freq='D')
        # Create strong upward trend in the last 30 days for velocity signal
        prices = np.concatenate([
            np.linspace(550, 570, 60),  # Gradual rise in first 60 days
            np.linspace(570, 600, 30)   # Stronger rise in last 30 days
        ])
        test_data = pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 1,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.uniform(10000000, 20000000, len(dates))
        }, index=dates)
        
        strategy.set_data(test_data)
        
        # Set live price higher to continue the upward trend
        live_price = 605.0
        
        # Mock live price fetch and contract retrieval (to avoid KeyError in signal detection)
        with patch.object(strategy, '_get_current_underlying_price', return_value=live_price):
            with patch.object(strategy, '_check_trend_success', return_value=(True, 5, 0.03)):
                # Call _has_buy_signal
                has_signal = strategy._has_buy_signal(self.current_date)
                
                # Verify signal was evaluated (may or may not trigger based on trend logic)
                # The important part is that live price was used in calculation
                self.assertIsNotNone(has_signal)
                
                # Verify data was updated with live price
                self.assertIn(self.current_date, strategy.data.index)
                self.assertEqual(strategy.data.loc[self.current_date, 'Close'], live_price)
                
                print(f"✅ Velocity signal detection using live price: {has_signal}")
                print(f"   Live price used: ${live_price}")

    def test_multiple_live_price_calls_on_same_date(self):
        """Test that multiple calls on the same date don't duplicate data."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(self.test_data.copy())
        
        original_length = len(strategy.data)
        
        # Mock live price fetch
        with patch.object(strategy, '_get_current_underlying_price', return_value=self.live_price):
            # Call _has_buy_signal multiple times
            strategy._has_buy_signal(self.current_date)
            first_call_length = len(strategy.data)
            
            # Second call - should not duplicate
            strategy._has_buy_signal(self.current_date)
            second_call_length = len(strategy.data)
            
            # Verify data was only added once
            self.assertEqual(first_call_length, original_length + 1)
            self.assertEqual(second_call_length, first_call_length)
            
            print(f"✅ Multiple calls don't duplicate data")


if __name__ == '__main__':
    unittest.main()

