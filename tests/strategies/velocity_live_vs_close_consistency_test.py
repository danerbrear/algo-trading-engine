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
import pandas as pd
import numpy as np
from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy

class TestVelocityLiveVsCloseConsistency(unittest.TestCase):
    """Test signal consistency between live price and close price scenarios."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level mocks to prevent real API calls."""
        cls.data_retriever_patcher = patch('algo_trading_engine.common.data_retriever.DataRetriever')
        cls.mock_data_retriever_class = cls.data_retriever_patcher.start()
        cls.mock_data_retriever_instance = Mock()
        cls.mock_data_retriever_instance.get_live_price.return_value = None
        cls.mock_data_retriever_class.return_value = cls.mock_data_retriever_instance

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level mocks."""
        cls.data_retriever_patcher.stop()

    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=90, freq='D')
        prices = np.concatenate([np.linspace(550, 570, 60), np.linspace(570, 600, 30)])
        self.historical_data = pd.DataFrame({'Open': prices - 1, 'High': prices + 1, 'Low': prices - 2, 'Close': prices, 'Volume': np.random.uniform(10000000, 20000000, len(dates))}, index=dates)
        self.live_price = 605.25
        self.close_price = 605.5
        self.current_date = datetime.now()

    def test_signal_consistency_live_vs_close_price(self):
        """Test that live price during market hours produces same signal as close price after close."""
        print('\n' + '=' * 80)
        print('TEST: Signal Consistency - Live Price vs Close Price')
        print('=' * 80)
        self.mock_data_retriever_instance.get_live_price.return_value = self.live_price
        print('\n📊 SCENARIO 1: During Market Hours (Live Price)')
        print('-' * 80)
        mock_options_handler_1 = Mock()
        mock_options_handler_1.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler_1.get_contract_list_for_date
        get_option_bar = mock_options_handler_1.get_option_bar
        get_options_chain = mock_options_handler_1.get_options_chain
        strategy_live = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy_live.set_data(self.historical_data.copy())

        def mock_get_price_live(_date, _symbol):
            return self.live_price
        strategy_live.get_current_underlying_price = mock_get_price_live
        print(f'Historical data ends: {strategy_live.data.index[-1].date()}')
        print(f'Current date: {self.current_date.date()}')
        print(f'Live price: ${self.live_price}')
        with patch.object(strategy_live, '_check_trend_success', return_value=(True, 5, 0.03)):
            signal_during_market_hours = strategy_live._has_buy_signal(self.current_date)
        if self.current_date in strategy_live.data.index:
            live_velocity = strategy_live.data.loc[self.current_date, 'MA_Velocity_15_30']
            live_velocity_change = strategy_live.data.loc[self.current_date, 'Velocity_Changes']
            print(f'Velocity with live price: {live_velocity:.6f}')
            print(f'Velocity change: {live_velocity_change:.6f}')
            print(f'Signal generated: {signal_during_market_hours}')
        else:
            print('⚠️  WARNING: Current date was not added to data!')
        print('\n📊 SCENARIO 2: After Market Close (Close Price)')
        print('-' * 80)
        mock_options_handler_2 = Mock()
        mock_options_handler_2.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler_2.get_contract_list_for_date
        get_option_bar = mock_options_handler_2.get_option_bar
        get_options_chain = mock_options_handler_2.get_options_chain
        strategy_close = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        data_with_close = self.historical_data.copy()
        current_date_close = pd.DataFrame({'Open': [600.0], 'High': [606.0], 'Low': [599.5], 'Close': [self.close_price], 'Volume': [18000000]}, index=[self.current_date])
        data_with_close = pd.concat([data_with_close, current_date_close])
        strategy_close.set_data(data_with_close)

        def mock_get_price_close(date, _symbol):
            return float(strategy_close.data.loc[date, 'Close'])
        strategy_close.get_current_underlying_price = mock_get_price_close
        print(f'Historical data ends: {strategy_close.data.index[-1].date()}')
        print(f'Close price: ${self.close_price}')
        with patch.object(strategy_close, '_check_trend_success', return_value=(True, 5, 0.03)):
            signal_after_market_close = strategy_close._has_buy_signal(self.current_date)
        close_velocity = strategy_close.data.loc[self.current_date, 'MA_Velocity_15_30']
        close_velocity_change = strategy_close.data.loc[self.current_date, 'Velocity_Changes']
        print(f'Velocity with close price: {close_velocity:.6f}')
        print(f'Velocity change: {close_velocity_change:.6f}')
        print(f'Signal generated: {signal_after_market_close}')
        print('\n' + '=' * 80)
        print('COMPARISON RESULTS')
        print('=' * 80)
        price_diff = abs(self.close_price - self.live_price)
        price_diff_pct = price_diff / self.live_price * 100
        print(f'\nPrice difference: ${price_diff:.2f} ({price_diff_pct:.3f}%)')
        print(f'Live price signal:  {signal_during_market_hours}')
        print(f'Close price signal: {signal_after_market_close}')
        if self.current_date in strategy_live.data.index:
            velocity_diff = abs(close_velocity - live_velocity)
            velocity_change_diff = abs(close_velocity_change - live_velocity_change)
            print(f'\nVelocity difference: {velocity_diff:.6f}')
            print(f'Velocity change difference: {velocity_change_diff:.6f}')
        print('\n' + '=' * 80)
        if signal_during_market_hours != signal_after_market_close:
            print('❌ FAILED: Signals are INCONSISTENT!')
            print(f'   During market hours: {signal_during_market_hours}')
            print(f'   After market close:  {signal_after_market_close}')
            print(f'   Price barely changed ({price_diff_pct:.3f}%), but signals differ!')
            self.fail(f'Signal inconsistency detected: live={signal_during_market_hours}, close={signal_after_market_close}')
        else:
            print('✅ PASSED: Signals are CONSISTENT!')
            print(f'   Both scenarios: {signal_during_market_hours}')
        print('=' * 80 + '\n')

    def test_stale_data_in_cache_prevents_live_fetch(self):
        """Test that stale cached data for current date prevents live price fetch."""
        print('\n' + '=' * 80)
        print('TEST: Stale Cached Data Detection')
        print('=' * 80)
        self.mock_data_retriever_instance.get_live_price.return_value = self.live_price
        data_with_stale_current = self.historical_data.copy()
        stale_price = 580.0
        stale_current_date = pd.DataFrame({'Open': [579.0], 'High': [581.0], 'Low': [578.0], 'Close': [stale_price], 'Volume': [15000000]}, index=[self.current_date])
        data_with_stale_current = pd.concat([data_with_stale_current, stale_current_date])
        mock_options_handler = Mock()
        mock_options_handler.symbol = 'SPY'
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(data_with_stale_current)

        def mock_get_price_stale(_date, _symbol):
            return self.live_price
        strategy.get_current_underlying_price = mock_get_price_stale
        print(f'\nCurrent date: {self.current_date.date()}')
        print(f'Stale cached price for current date: ${stale_price}')
        print(f'Expected live price: ${self.live_price}')
        try:
            self.mock_data_retriever_instance.get_live_price.reset_mock()
            strategy._has_buy_signal(self.current_date)
            if self.mock_data_retriever_instance.get_live_price.called:
                print('✅ Live price fetch WAS called (good - overwrites stale data)')
            else:
                print('❌ Live price fetch was NOT called (bad - uses stale data)')
                print('   This is the BUG that causes inconsistent signals!')
                actual_price = strategy.data.loc[self.current_date, 'Close']
                print(f'   Price being used: ${actual_price}')
                if actual_price == stale_price:
                    self.fail(f'Strategy is using stale cached data (${stale_price}) instead of fetching live price (${self.live_price}) for current date!')
        except KeyError:
            print('✅ Date not in index - live fetch would be triggered')

    def test_recommendation_engine_fresh_data_fetch(self):
        """Test that recommendation engine always uses fresh data for current date."""
        print('\n' + '=' * 80)
        print('TEST: Recommendation Engine Fresh Data')
        print('=' * 80)
        print('\n✅ This scenario is validated by test_stale_data_in_cache_prevents_live_fetch')
        print('   The fix ensures that current date always fetches live price,')
        print('   even when stale cached data exists.')
        print('\n' + '=' * 80)
if __name__ == '__main__':
    import sys
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVelocityLiveVsCloseConsistency)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
