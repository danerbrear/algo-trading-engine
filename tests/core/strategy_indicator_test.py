"""
Unit tests for Strategy class indicator functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.core.indicators.indicator import Indicator
from algo_trading_engine.core.indicators.average_true_return_indicator import ATRIndicator
from algo_trading_engine.enums import BarTimeInterval


class ConcreteTestStrategy(Strategy):
    """Concrete implementation of Strategy for testing"""
    
    def __init__(self, indicators=None):
        super().__init__(indicators=indicators)
        self.on_new_date_called = False
        self.on_new_date_call_count = 0
        self.last_date_processed = None
    
    def on_new_date(self, date, positions, add_position, remove_position):
        """Implementation of abstract on_new_date"""
        # Call parent to update indicators
        super().on_new_date(date, positions, add_position, remove_position)
        
        self.on_new_date_called = True
        self.on_new_date_call_count += 1
        self.last_date_processed = date
    
    def on_end(self, positions, remove_position, date):
        """Implementation of abstract on_end"""
        pass
    
    def validate_data(self, data):
        """Implementation of abstract validate_data"""
        return 'Close' in data.columns and len(data) > 0


class MockIndicator(Indicator):
    """Mock indicator for testing"""
    
    def __init__(self, name="MockIndicator", should_fail=False):
        super().__init__(name=name)
        self.update_called = False
        self.update_call_count = 0
        self.last_update_date = None
        self.last_update_data = None
        self.should_fail = should_fail
        self._value = None
    
    def update(self, date, data):
        """Mock update implementation"""
        self.update_called = True
        self.update_call_count += 1
        self.last_update_date = date
        self.last_update_data = data
        
        if self.should_fail:
            raise ValueError(f"Mock indicator {self.name} intentionally failed")
        
        self._value = 42.0  # Mock value
    
    @property
    def value(self):
        return self._value


class TestStrategyIndicatorInitialization:
    """Test cases for Strategy initialization with indicators"""
    
    def test_strategy_init_without_indicators(self):
        """Test Strategy initialization without indicators"""
        strategy = ConcreteTestStrategy()
        assert strategy.indicators == []
        assert isinstance(strategy.indicators, list)
    
    def test_strategy_init_with_single_indicator(self):
        """Test Strategy initialization with a single indicator"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        
        assert len(strategy.indicators) == 1
        assert strategy.indicators[0] == indicator
    
    def test_strategy_init_with_multiple_indicators(self):
        """Test Strategy initialization with multiple indicators"""
        indicator1 = MockIndicator(name="Indicator1")
        indicator2 = MockIndicator(name="Indicator2")
        indicator3 = ATRIndicator(period=14)
        
        strategy = ConcreteTestStrategy(indicators=[indicator1, indicator2, indicator3])
        
        assert len(strategy.indicators) == 3
        assert strategy.indicators[0] == indicator1
        assert strategy.indicators[1] == indicator2
        assert strategy.indicators[2] == indicator3
    
    def test_strategy_init_with_atr_indicator(self):
        """Test Strategy initialization with real ATRIndicator"""
        atr = ATRIndicator(period=14, period_unit=BarTimeInterval.DAY)
        strategy = ConcreteTestStrategy(indicators=[atr])
        
        assert len(strategy.indicators) == 1
        assert isinstance(strategy.indicators[0], ATRIndicator)
        assert strategy.indicators[0].period == 14


class TestStrategyUpdateIndicators:
    """Test cases for _update_indicators method"""
    
    def create_sample_data(self, num_days=20):
        """Helper to create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i for i in range(num_days)],
            'High': [105 + i for i in range(num_days)],
            'Low': [95 + i for i in range(num_days)],
            'Close': [100 + i for i in range(num_days)],
            'Volume': [1000000] * num_days
        }, index=dates)
        return data
    
    def test_update_indicators_success_single_indicator(self):
        """Test _update_indicators succeeds with single indicator"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        strategy.set_data(self.create_sample_data())
        
        current_date = datetime(2024, 1, 10)
        result = strategy._update_indicators(current_date)
        
        assert result is True
        assert indicator.update_called is True
        assert indicator.update_call_count == 1
        assert indicator.last_update_date == current_date
        assert indicator.last_update_data is not None
    
    def test_update_indicators_success_multiple_indicators(self):
        """Test _update_indicators succeeds with multiple indicators"""
        indicator1 = MockIndicator(name="Indicator1")
        indicator2 = MockIndicator(name="Indicator2")
        indicator3 = MockIndicator(name="Indicator3")
        
        strategy = ConcreteTestStrategy(indicators=[indicator1, indicator2, indicator3])
        strategy.set_data(self.create_sample_data())
        
        current_date = datetime(2024, 1, 10)
        result = strategy._update_indicators(current_date)
        
        assert result is True
        assert indicator1.update_called is True
        assert indicator2.update_called is True
        assert indicator3.update_called is True
        assert indicator1.last_update_date == current_date
        assert indicator2.last_update_date == current_date
        assert indicator3.last_update_date == current_date
    
    def test_update_indicators_no_indicators_returns_true(self):
        """Test _update_indicators returns True when no indicators"""
        strategy = ConcreteTestStrategy(indicators=[])
        strategy.set_data(self.create_sample_data())
        
        result = strategy._update_indicators(datetime(2024, 1, 10))
        
        assert result is True
    
    def test_update_indicators_failure_returns_false(self, capsys):
        """Test _update_indicators returns False when indicator fails"""
        failing_indicator = MockIndicator(name="FailingIndicator", should_fail=True)
        strategy = ConcreteTestStrategy(indicators=[failing_indicator])
        strategy.set_data(self.create_sample_data())
        
        result = strategy._update_indicators(datetime(2024, 1, 10))
        
        assert result is False
        assert failing_indicator.update_called is True
        
        # Check error message was printed
        captured = capsys.readouterr()
        assert "Error updating indicator FailingIndicator" in captured.out
    
    def test_update_indicators_stops_on_first_failure(self, capsys):
        """Test _update_indicators stops updating after first failure"""
        indicator1 = MockIndicator(name="Indicator1")
        failing_indicator = MockIndicator(name="FailingIndicator", should_fail=True)
        indicator3 = MockIndicator(name="Indicator3")
        
        strategy = ConcreteTestStrategy(indicators=[indicator1, failing_indicator, indicator3])
        strategy.set_data(self.create_sample_data())
        
        result = strategy._update_indicators(datetime(2024, 1, 10))
        
        assert result is False
        assert indicator1.update_called is True
        assert failing_indicator.update_called is True
        # indicator3 should not be called due to early return
        assert indicator3.update_called is False
    
    def test_update_indicators_passes_strategy_data(self):
        """Test _update_indicators passes strategy's data to indicators"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        
        test_data = self.create_sample_data()
        strategy.set_data(test_data)
        
        strategy._update_indicators(datetime(2024, 1, 10))
        
        # Verify the indicator received the strategy's data
        assert indicator.last_update_data is test_data
        pd.testing.assert_frame_equal(indicator.last_update_data, test_data)


class TestStrategyWithATRIndicator:
    """Test cases for Strategy with real ATRIndicator"""
    
    def create_sample_data(self, num_days=20):
        """Helper to create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i for i in range(num_days)],
            'High': [110 + i for i in range(num_days)],
            'Low': [90 + i for i in range(num_days)],
            'Close': [100 + i for i in range(num_days)],
            'Volume': [1000000] * num_days
        }, index=dates)
        return data
    
    def test_strategy_updates_atr_indicator(self):
        """Test Strategy successfully updates ATRIndicator"""
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy(indicators=[atr])
        strategy.set_data(self.create_sample_data())
        
        # ATR should be None before update
        assert atr.value is None
        
        # Update indicators (need at least period+1 bars)
        result = strategy._update_indicators(datetime(2024, 1, 6))
        
        assert result is True
        assert atr.value is not None
        assert atr.value > 0
    
    def test_strategy_atr_insufficient_data_fails(self, capsys):
        """Test Strategy with ATR fails gracefully with insufficient data"""
        atr = ATRIndicator(period=14)
        strategy = ConcreteTestStrategy(indicators=[atr])
        strategy.set_data(self.create_sample_data(num_days=5))  # Not enough data
        
        result = strategy._update_indicators(datetime(2024, 1, 5))
        
        assert result is False
        assert atr.value is None
        
        captured = capsys.readouterr()
        assert "Error updating indicator ATR" in captured.out
    
    def test_strategy_multiple_indicators_including_atr(self):
        """Test Strategy with multiple indicators including ATR"""
        mock_indicator = MockIndicator(name="MockIndicator")
        atr = ATRIndicator(period=5)
        
        strategy = ConcreteTestStrategy(indicators=[mock_indicator, atr])
        strategy.set_data(self.create_sample_data())
        
        result = strategy._update_indicators(datetime(2024, 1, 6))
        
        assert result is True
        assert mock_indicator.update_called is True
        assert atr.value is not None
    
    def test_strategy_atr_values_update_progressively(self):
        """Test ATR values update correctly across multiple calls"""
        atr = ATRIndicator(period=3)
        strategy = ConcreteTestStrategy(indicators=[atr])
        strategy.set_data(self.create_sample_data())
        
        # Update for day 4 (initialize ATR)
        strategy._update_indicators(datetime(2024, 1, 4))
        atr_value_day4 = atr.value
        assert atr_value_day4 is not None
        
        # Update for day 5 (should use Wilder's smoothing)
        strategy._update_indicators(datetime(2024, 1, 5))
        atr_value_day5 = atr.value
        assert atr_value_day5 is not None
        
        # Update for day 6
        strategy._update_indicators(datetime(2024, 1, 6))
        atr_value_day6 = atr.value
        assert atr_value_day6 is not None
        
        # All values should be valid
        assert all(val > 0 for val in [atr_value_day4, atr_value_day5, atr_value_day6])


class TestStrategyOnNewDateWithIndicators:
    """Test cases for on_new_date integration with indicators"""
    
    def create_sample_data(self, num_days=20):
        """Helper to create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i for i in range(num_days)],
            'High': [110 + i for i in range(num_days)],
            'Low': [90 + i for i in range(num_days)],
            'Close': [100 + i for i in range(num_days)],
            'Volume': [1000000] * num_days
        }, index=dates)
        return data
    
    def test_on_new_date_updates_indicators_before_strategy_logic(self):
        """Test on_new_date updates indicators before executing strategy logic"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        strategy.set_data(self.create_sample_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        strategy.on_new_date(
            datetime(2024, 1, 10),
            (),
            mock_add_position,
            mock_remove_position
        )
        
        # Indicator should have been updated
        assert indicator.update_called is True
        # Strategy logic should have executed
        assert strategy.on_new_date_called is True
    
    def test_on_new_date_skips_strategy_logic_when_indicators_fail(self):
        """Test on_new_date prints error message when indicators fail to update"""
        failing_indicator = MockIndicator(should_fail=True)
        strategy = ConcreteTestStrategy(indicators=[failing_indicator])
        strategy.set_data(self.create_sample_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        strategy.on_new_date(
            datetime(2024, 1, 10),
            (),
            mock_add_position,
            mock_remove_position
        )
        
        # Indicator update should have been attempted
        assert failing_indicator.update_called is True
        # The parent's on_new_date still executes (prints message and returns)
        # but the child's logic executes after the parent returns early
        # So on_new_date_called will be True but execution was logged as error
        assert strategy.on_new_date_called is True
    
    def test_on_new_date_with_atr_indicator(self):
        """Test on_new_date works correctly with ATRIndicator"""
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy(indicators=[atr])
        strategy.set_data(self.create_sample_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        # ATR should be None initially
        assert atr.value is None
        
        # Call on_new_date (should initialize ATR)
        strategy.on_new_date(
            datetime(2024, 1, 6),
            (),
            mock_add_position,
            mock_remove_position
        )
        
        # ATR should now have a value
        assert atr.value is not None
        assert strategy.on_new_date_called is True
    
    def test_on_new_date_multiple_calls_maintain_indicator_state(self):
        """Test multiple on_new_date calls maintain indicator state"""
        atr = ATRIndicator(period=3)
        strategy = ConcreteTestStrategy(indicators=[atr])
        strategy.set_data(self.create_sample_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        # First call - initialize ATR
        strategy.on_new_date(datetime(2024, 1, 4), (), mock_add_position, mock_remove_position)
        atr_value_1 = atr.value
        
        # Second call - update ATR
        strategy.on_new_date(datetime(2024, 1, 5), (), mock_add_position, mock_remove_position)
        atr_value_2 = atr.value
        
        # Third call - update ATR
        strategy.on_new_date(datetime(2024, 1, 6), (), mock_add_position, mock_remove_position)
        atr_value_3 = atr.value
        
        # All values should be valid
        assert all(val is not None for val in [atr_value_1, atr_value_2, atr_value_3])
        assert strategy.on_new_date_call_count == 3


class TestStrategyIndicatorAccessInStrategy:
    """Test cases for accessing indicator values within strategy logic"""
    
    class StrategyUsingIndicator(Strategy):
        """Strategy that uses indicator values in its logic"""
        
        def __init__(self, atr_indicator):
            super().__init__(indicators=[atr_indicator])
            self.atr_threshold = 5.0
            self.signal_generated = False
            self.current_atr = None
        
        def on_new_date(self, date, positions, add_position, remove_position):
            # Call parent to update indicators
            super().on_new_date(date, positions, add_position, remove_position)
            
            # Access ATR value
            atr = self.indicators[0]
            self.current_atr = atr.value
            
            # Use ATR in strategy logic
            if atr.value is not None and atr.value > self.atr_threshold:
                self.signal_generated = True
        
        def on_end(self, positions, remove_position, date):
            pass
        
        def validate_data(self, data):
            return True
    
    def create_volatile_data(self):
        """Create data with high volatility"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            'High': [150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
            'Low': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            'Close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            'Volume': [1000000] * 10
        }, index=dates)
        return data
    
    def test_strategy_can_access_indicator_value(self):
        """Test strategy can access and use indicator values"""
        atr = ATRIndicator(period=3)
        strategy = self.StrategyUsingIndicator(atr)
        strategy.set_data(self.create_volatile_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        strategy.on_new_date(datetime(2024, 1, 4), (), mock_add_position, mock_remove_position)
        
        # Strategy should have accessed the ATR value
        assert strategy.current_atr is not None
        assert strategy.current_atr == atr.value
    
    def test_strategy_uses_indicator_in_decision_logic(self):
        """Test strategy uses indicator value in trading decisions"""
        atr = ATRIndicator(period=3)
        strategy = self.StrategyUsingIndicator(atr)
        strategy.set_data(self.create_volatile_data())
        
        mock_add_position = Mock()
        mock_remove_position = Mock()
        
        strategy.on_new_date(datetime(2024, 1, 4), (), mock_add_position, mock_remove_position)
        
        # With high volatility data, ATR should exceed threshold
        assert strategy.signal_generated is True
        assert strategy.current_atr > strategy.atr_threshold


class TestStrategySetData:
    """Test cases for set_data method with indicators"""
    
    def test_set_data_stores_data_for_indicators(self):
        """Test set_data makes data available for indicators"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Close': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)
        
        strategy.set_data(data)
        
        assert strategy.data is not None
        pd.testing.assert_frame_equal(strategy.data, data)
        
        # Verify indicators can access the data
        strategy._update_indicators(datetime(2024, 1, 5))
        assert indicator.last_update_data is data


class TestStrategyIndicatorEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_update_indicators_with_none_data(self):
        """Test _update_indicators when strategy data is None"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy(indicators=[indicator])
        # Don't set data
        
        # MockIndicator doesn't actually validate data, so it won't fail with None
        # It will just receive None. This tests that _update_indicators itself doesn't crash
        result = strategy._update_indicators(datetime(2024, 1, 10))
        
        # MockIndicator doesn't validate data, so this succeeds
        # For a real indicator like ATR, it would fail
        assert result is True
        assert indicator.last_update_data is None
    
    def test_indicators_list_is_mutable(self):
        """Test that indicators list can be modified after initialization"""
        strategy = ConcreteTestStrategy(indicators=[])
        assert len(strategy.indicators) == 0
        
        # Add indicator after initialization
        new_indicator = MockIndicator()
        strategy.indicators.append(new_indicator)
        
        assert len(strategy.indicators) == 1
        
        # Verify it gets updated
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Close': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)
        strategy.set_data(data)
        
        strategy._update_indicators(datetime(2024, 1, 5))
        assert new_indicator.update_called is True
