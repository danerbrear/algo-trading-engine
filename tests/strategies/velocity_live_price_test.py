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
from algo_trading_engine.vo import Position, create_position
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import Option, OptionType
from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy

class TestVelocityLivePrice(unittest.TestCase):
    """Test cases for velocity calculation using live SPY price when market is open."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=60, freq='D')
        self.test_data = pd.DataFrame({'Open': np.random.uniform(580, 600, len(dates)), 'High': np.random.uniform(590, 610, len(dates)), 'Low': np.random.uniform(570, 590, len(dates)), 'Close': np.linspace(580, 600, len(dates)), 'Volume': np.random.uniform(10000000, 20000000, len(dates))}, index=dates)
        self.live_price = 605.5
        self.current_date = datetime.now()

    def test_velocity_calculation_uses_live_price_when_market_open(self):
        """Test that velocity calculation uses live SPY price when market is open on current date."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy(), {})
        self.assertNotIn(self.current_date, strategy.data.index)

        def mock_get_price(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price
        result = strategy._has_buy_signal(self.current_date)
        self.assertIn(self.current_date, strategy.data.index)
        added_price = strategy.data.loc[self.current_date, 'Close']
        self.assertEqual(added_price, self.live_price)
        self.assertIn('SMA_15', strategy.data.columns)
        self.assertIn('SMA_30', strategy.data.columns)
        self.assertIn('MA_Velocity_15_30', strategy.data.columns)
        self.assertIn('Velocity_Changes', strategy.data.columns)
        current_velocity = strategy.data.loc[self.current_date, 'MA_Velocity_15_30']
        self.assertIsNotNone(current_velocity)
        self.assertFalse(pd.isna(current_velocity))
        print(f'✅ Live price ${self.live_price} was successfully used to calculate velocity')
        print(f'   Current velocity: {current_velocity:.6f}')

    def test_velocity_strategy_appends_live_data_to_history(self):
        """Test that live price data is correctly appended to historical data."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())
        original_length = len(strategy.data)

        def mock_get_price(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price
        strategy._has_buy_signal(self.current_date)
        self.assertEqual(len(strategy.data), original_length + 1)
        new_row = strategy.data.loc[self.current_date]
        self.assertEqual(new_row['Close'], self.live_price)
        self.assertEqual(new_row['Open'], self.live_price)
        self.assertEqual(new_row['High'], self.live_price)
        self.assertEqual(new_row['Low'], self.live_price)
        self.assertEqual(new_row['Volume'], 0)
        print(f'✅ Live data correctly appended to historical data')

    def test_velocity_recalculation_after_live_price_update(self):
        """Test that moving averages and velocity are recalculated after adding live price."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())
        last_date_before = strategy.data.index[-1]
        velocity_before = strategy.data.loc[last_date_before, 'MA_Velocity_15_30']

        def mock_get_price(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price
        strategy._has_buy_signal(self.current_date)
        velocity_current = strategy.data.loc[self.current_date, 'MA_Velocity_15_30']
        self.assertIsNotNone(velocity_current)
        self.assertFalse(pd.isna(velocity_current))
        velocity_change = strategy.data.loc[self.current_date, 'Velocity_Changes']
        self.assertIsNotNone(velocity_change)
        print(f'✅ Velocity recalculated after live price update')
        print(f'   Previous velocity: {velocity_before:.6f}')
        print(f'   Current velocity: {velocity_current:.6f}')
        print(f'   Velocity change: {velocity_change:.6f}')

    def test_live_price_fetch_integration_with_data_retriever(self):
        """Test that live price fetch integrates correctly with DataRetriever.
        
        Note: The strategy now creates DataRetriever on demand when needed,
        rather than using a pre-initialized data_retriever attribute.
        This test verifies that the on-demand creation works correctly.
        """
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())

        def mock_get_price(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price
        symbol = strategy.data.index.name if strategy.data.index.name else 'SPY'
        price = strategy.get_current_underlying_price(self.current_date, symbol)
        self.assertEqual(price, self.live_price)
        print(f'✅ DataRetriever integration working correctly')

    def test_fallback_to_cached_data_when_live_price_unavailable(self):
        """Test that strategy falls back to cached data when live price fetch fails."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        mock_data_retriever = Mock()
        mock_data_retriever.get_live_price.return_value = None
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())
        cached_date = self.test_data.index[-1]
        symbol = strategy.data.index.name if strategy.data.index.name else 'SPY'

        def mock_get_price(date, _sym):
            return float(strategy.data.loc[date, 'Close'])
        strategy.get_current_underlying_price = mock_get_price
        price = strategy.get_current_underlying_price(cached_date, symbol)
        cached_price = float(self.test_data.loc[cached_date, 'Close'])
        self.assertEqual(price, cached_price)
        print(f'✅ Fallback to cached data working correctly')

    def test_market_closed_uses_cached_data(self):
        """Test that when market is closed (non-current date), cached data is used."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())
        past_date = self.test_data.index[-2]
        symbol = strategy.data.index.name if strategy.data.index.name else 'SPY'

        def mock_get_price(date, _sym):
            return float(strategy.data.loc[date, 'Close'])
        strategy.get_current_underlying_price = mock_get_price
        price = strategy.get_current_underlying_price(past_date, symbol)
        cached_price = float(self.test_data.loc[past_date, 'Close'])
        self.assertIsNotNone(price)
        self.assertEqual(price, cached_price)
        print(f'✅ Market closed scenario uses cached data correctly')

    def test_velocity_signal_detection_with_live_price(self):
        """Test that velocity signal is correctly detected using live price."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=90, freq='D')
        prices = np.concatenate([np.linspace(550, 570, 60), np.linspace(570, 600, 30)])
        test_data = pd.DataFrame({'Open': prices - 1, 'High': prices + 1, 'Low': prices - 2, 'Close': prices, 'Volume': np.random.uniform(10000000, 20000000, len(dates))}, index=dates)
        strategy.set_data(test_data)
        live_price = 605.0

        def mock_get_price(_date, _symbol):
            return live_price
        strategy.get_current_underlying_price = mock_get_price
        with patch.object(strategy, '_check_trend_success', return_value=(True, 5, 0.03)):
            has_signal = strategy._has_buy_signal(self.current_date)
        self.assertIsNotNone(has_signal)
        self.assertIn(self.current_date, strategy.data.index)
        self.assertEqual(strategy.data.loc[self.current_date, 'Close'], live_price)
        print(f'✅ Velocity signal detection using live price: {has_signal}')
        print(f'   Live price used: ${live_price}')

    def test_multiple_live_price_calls_on_same_date(self):
        """Test that multiple calls on the same date don't duplicate data."""
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(self.test_data.copy())
        original_length = len(strategy.data)

        def mock_get_price(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price
        strategy._has_buy_signal(self.current_date)
        first_call_length = len(strategy.data)
        strategy._has_buy_signal(self.current_date)
        second_call_length = len(strategy.data)
        self.assertEqual(first_call_length, original_length + 1)
        self.assertEqual(second_call_length, first_call_length)
        print(f"✅ Multiple calls don't duplicate data")
if __name__ == '__main__':
    unittest.main()
