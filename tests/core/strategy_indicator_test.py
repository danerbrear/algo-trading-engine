"""
Unit tests for Strategy class indicator functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import pandas as pd
from algo_trading_engine.strategy import Strategy
from algo_trading_engine.indicators.indicator import Indicator
from algo_trading_engine.indicators.average_true_return_indicator import ATRIndicator
from algo_trading_engine.indicators.sma_indicator import SMAIndicator
from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine._internal.common.trading_engine import TradingEngine
from algo_trading_engine.models.config import PaperTradingConfig

class ConcreteTestStrategy(Strategy):
    """Concrete implementation of Strategy for testing"""

    def __init__(self):
        super().__init__()
        self.on_new_date_called = False
        self.on_new_date_call_count = 0
        self.last_date_processed = None

    def on_new_date(self, date, positions, add_position, remove_position):
        """Implementation of abstract on_new_date"""
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

    def __init__(self, name='MockIndicator', should_fail=False, mock_warm_up: int=5):
        super().__init__(name=name)
        self.update_called = False
        self.update_call_count = 0
        self.last_update_date = None
        self.last_update_data = None
        self.should_fail = should_fail
        self._value = None
        self._mock_warm_up = mock_warm_up

    @property
    def warm_up_period(self) -> int:
        return self._mock_warm_up

    def precompute(self, data):
        for date in data.index:
            self.update(date, data)

    def update(self, date, data):
        """Mock update implementation"""
        self.update_called = True
        self.update_call_count += 1
        self.last_update_date = date
        self.last_update_data = data
        if self.should_fail:
            raise ValueError(f'Mock indicator {self.name} intentionally failed')
        self._value = 42.0
        self._values[date] = 42.0

    def print(self):
        """Mock print implementation"""
        print(f'{self.name}: value={self._value}, updates={self.update_call_count}')

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
        """Test Strategy initialization with a single indicator using add_indicator"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(indicator)
        assert len(strategy.indicators) == 1
        assert strategy.indicators[0] == indicator

    def test_strategy_init_with_multiple_indicators(self):
        """Test Strategy initialization with multiple indicators using add_indicator"""
        indicator1 = MockIndicator(name='Indicator1')
        indicator2 = MockIndicator(name='Indicator2')
        indicator3 = ATRIndicator(period=14)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(indicator1)
        strategy.add_indicator(indicator2)
        strategy.add_indicator(indicator3)
        assert len(strategy.indicators) == 3
        assert strategy.indicators[0] == indicator1
        assert strategy.indicators[1] == indicator2
        assert strategy.indicators[2] == indicator3

    def test_strategy_init_with_atr_indicator(self):
        """Test Strategy initialization with real ATRIndicator"""
        atr = ATRIndicator(period=14, period_unit=BarTimeInterval.DAY)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        assert len(strategy.indicators) == 1
        assert isinstance(strategy.indicators[0], ATRIndicator)
        assert strategy.indicators[0].period == 14

class TestStrategyWarmUpPeriod:
    """Test cases for Strategy.warm_up_period property"""

    def test_warm_up_period_zero_with_no_indicators(self):
        strategy = ConcreteTestStrategy()
        assert strategy.warm_up_period == 0

    def test_warm_up_period_single_indicator(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(MockIndicator(mock_warm_up=10))
        assert strategy.warm_up_period == 10

    def test_warm_up_period_returns_max(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(MockIndicator(name='A', mock_warm_up=5))
        strategy.add_indicator(MockIndicator(name='B', mock_warm_up=20))
        strategy.add_indicator(MockIndicator(name='C', mock_warm_up=12))
        assert strategy.warm_up_period == 20

    def test_warm_up_period_with_real_indicators(self):
        from algo_trading_engine.indicators.sma_indicator import SMAIndicator
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(SMAIndicator(period=15))
        strategy.add_indicator(SMAIndicator(period=30))
        assert strategy.warm_up_period == 30

    def test_warm_up_period_with_atr(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(ATRIndicator(period=14))
        assert strategy.warm_up_period == 15

class TestStrategyWithATRIndicator:
    """Test cases for Strategy with real ATRIndicator"""

    def create_sample_data(self, num_days=20):
        """Helper to create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
        data = pd.DataFrame({'Open': [100 + i for i in range(num_days)], 'High': [110 + i for i in range(num_days)], 'Low': [90 + i for i in range(num_days)], 'Close': [100 + i for i in range(num_days)], 'Volume': [1000000] * num_days}, index=dates)
        return data

    def test_strategy_updates_atr_indicator(self):
        """Test Strategy successfully updates ATRIndicator"""
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        strategy.add_indicator(atr)
        data = self.create_sample_data()
        strategy.set_data(data)
        atr.update(datetime(2024, 1, 6), data)
        assert atr.value is not None
        assert atr.value > 0

    def test_strategy_atr_insufficient_data_noops_without_error(self):
        """Before period+1 bars exist, ATR update is a no-op."""
        atr = ATRIndicator(period=14)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        data = self.create_sample_data(num_days=5)
        strategy.set_data(data)
        atr.update(datetime(2024, 1, 5), data)
        assert atr.value is None

    def test_strategy_multiple_indicators_including_atr(self):
        """Test Strategy with multiple indicators including ATR"""
        mock_indicator = MockIndicator(name='MockIndicator')
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(mock_indicator)
        strategy.add_indicator(atr)
        data = self.create_sample_data()
        strategy.set_data(data)
        as_of = datetime(2024, 1, 6)
        mock_indicator.update(as_of, data)
        atr.update(as_of, data)
        assert mock_indicator.update_called is True
        assert atr.value is not None

    def test_strategy_atr_values_update_progressively(self):
        """Test ATR values update correctly across multiple calls"""
        atr = ATRIndicator(period=3)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        data = self.create_sample_data()
        strategy.set_data(data)
        atr.update(datetime(2024, 1, 4), data)
        atr_value_day4 = atr.value
        atr.update(datetime(2024, 1, 5), data)
        atr_value_day5 = atr.value
        atr.update(datetime(2024, 1, 6), data)
        atr_value_day6 = atr.value
        assert all((val is not None and val > 0 for val in [atr_value_day4, atr_value_day5, atr_value_day6]))

class TestStrategyOnNewDateWithIndicators:
    """Test cases for on_new_date integration with indicators"""

    def create_sample_data(self, num_days=20):
        """Helper to create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
        data = pd.DataFrame({'Open': [100 + i for i in range(num_days)], 'High': [110 + i for i in range(num_days)], 'Low': [90 + i for i in range(num_days)], 'Close': [100 + i for i in range(num_days)], 'Volume': [1000000] * num_days}, index=dates)
        return data

    def test_on_new_date_does_not_update_indicators(self):
        """Backtest dates are already precomputed, so on_new_date leaves indicators alone."""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(indicator)
        strategy.set_data(self.create_sample_data())
        strategy.on_new_date(datetime(2024, 1, 10), (), Mock(), Mock())
        assert indicator.update_called is False
        assert strategy.on_new_date_called is True

    def test_on_new_date_reads_precomputed_atr(self):
        """on_new_date uses the series written before the run starts."""
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        data = self.create_sample_data()
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        assert atr.value is not None
        strategy.on_new_date(datetime(2024, 1, 6), (), Mock(), Mock())
        assert strategy.on_new_date_called is True
        assert atr.value is not None

class TestStrategyIndicatorAccessInStrategy:
    """Test cases for accessing indicator values within strategy logic"""

    class StrategyUsingIndicator(Strategy):
        """Strategy that uses indicator values in its logic"""

        def __init__(self, atr_indicator):
            super().__init__()
            self.add_indicator(atr_indicator)
            self.atr_threshold = 5.0
            self.signal_generated = False
            self.current_atr = None

        def on_new_date(self, date, positions, add_position, remove_position):
            super().on_new_date(date, positions, add_position, remove_position)
            atr = self.indicators[0]
            self.current_atr = atr.value
            if atr.value is not None and atr.value > self.atr_threshold:
                self.signal_generated = True

        def on_end(self, positions, remove_position, date):
            pass

        def validate_data(self, data):
            return True

    def create_volatile_data(self):
        """Create data with high volatility"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'Open': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], 'High': [150, 150, 150, 150, 150, 150, 150, 150, 150, 150], 'Low': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], 'Close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], 'Volume': [1000000] * 10}, index=dates)
        return data

    def test_strategy_can_access_indicator_value(self):
        """Test strategy can access and use indicator values"""
        atr = ATRIndicator(period=3)
        strategy = self.StrategyUsingIndicator(atr)
        data = self.create_volatile_data()
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        strategy.on_new_date(datetime(2024, 1, 4), (), Mock(), Mock())
        assert strategy.current_atr is not None
        assert strategy.current_atr == atr.value

    def test_strategy_uses_indicator_in_decision_logic(self):
        """Test strategy uses indicator value in trading decisions"""
        atr = ATRIndicator(period=3)
        strategy = self.StrategyUsingIndicator(atr)
        data = self.create_volatile_data()
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        strategy.on_new_date(datetime(2024, 1, 4), (), Mock(), Mock())
        assert strategy.signal_generated is True
        assert strategy.current_atr > strategy.atr_threshold

class TestStrategySetData:
    """Test cases for set_data method with indicators"""

    def test_set_data_stores_data_for_indicators(self):
        """Test set_data makes data available for indicators"""
        indicator = MockIndicator()
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(indicator)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'Close': [100 + i for i in range(10)], 'High': [105 + i for i in range(10)], 'Low': [95 + i for i in range(10)], 'Volume': [1000000] * 10}, index=dates)
        strategy.set_data(data)
        assert strategy.data is not None
        pd.testing.assert_frame_equal(strategy.data, data)
        indicator.update(datetime(2024, 1, 5), data)
        assert indicator.last_update_data is data

class TestStrategyIndicatorEdgeCases:
    """Test edge cases and error scenarios"""

    def test_indicators_list_is_mutable(self):
        """Test that indicators list can be modified after initialization"""
        strategy = ConcreteTestStrategy()
        assert len(strategy.indicators) == 0
        new_indicator = MockIndicator()
        strategy.indicators.append(new_indicator)
        assert len(strategy.indicators) == 1
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'Close': [100 + i for i in range(10)], 'High': [105 + i for i in range(10)], 'Low': [95 + i for i in range(10)], 'Volume': [1000000] * 10}, index=dates)
        strategy.set_data(data)
        new_indicator.update(datetime(2024, 1, 5), data)
        assert new_indicator.update_called is True

class TestStrategyGetIndicator:
    """Test cases for get_indicator method"""

    def test_get_indicator_returns_correct_indicator(self):
        """Test get_indicator returns the correct indicator by type"""
        mock_indicator = MockIndicator(name='MockIndicator')
        atr = ATRIndicator(period=14)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(mock_indicator)
        strategy.add_indicator(atr)
        retrieved_atr = strategy.get_indicator(ATRIndicator)
        assert retrieved_atr is atr
        assert retrieved_atr.period == 14

    def test_get_indicator_returns_none_when_not_found(self):
        """Test get_indicator returns None when indicator not in list"""
        mock_indicator = MockIndicator(name='MockIndicator')
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(mock_indicator)
        result = strategy.get_indicator(ATRIndicator)
        assert result is None

    def test_get_indicator_returns_first_match_with_multiple_same_type(self):
        """Test get_indicator returns first match when multiple of same type"""
        atr1 = ATRIndicator(period=14)
        atr2 = ATRIndicator(period=20)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr1)
        strategy.add_indicator(atr2)
        result = strategy.get_indicator(ATRIndicator)
        assert result is atr1
        assert result.period == 14

    def test_get_indicator_with_no_indicators(self):
        """Test get_indicator returns None when strategy has no indicators"""
        strategy = ConcreteTestStrategy()
        result = strategy.get_indicator(ATRIndicator)
        assert result is None

    def test_get_indicator_usage_in_strategy(self):
        """Test practical usage of get_indicator in strategy logic"""
        atr = ATRIndicator(period=5)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'Open': [100 + i for i in range(10)], 'High': [110 + i for i in range(10)], 'Low': [90 + i for i in range(10)], 'Close': [100 + i for i in range(10)], 'Volume': [1000000] * 10}, index=dates)
        strategy.set_data(data)
        atr.update(datetime(2024, 1, 6), data)
        retrieved_atr = strategy.get_indicator(ATRIndicator)
        assert retrieved_atr is not None
        assert retrieved_atr.value is not None
        assert retrieved_atr.value > 0

class TestStrategyGetIndicatorByName:
    """Test cases for get_indicator_by_name method"""

    def test_returns_correct_indicator_by_name(self):
        mock1 = MockIndicator(name='Alpha')
        mock2 = MockIndicator(name='Beta')
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(mock1)
        strategy.add_indicator(mock2)
        assert strategy.get_indicator_by_name('Alpha') is mock1
        assert strategy.get_indicator_by_name('Beta') is mock2

    def test_returns_none_when_name_not_found(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(MockIndicator(name='Alpha'))
        assert strategy.get_indicator_by_name('NonExistent') is None

    def test_returns_none_when_no_indicators(self):
        strategy = ConcreteTestStrategy()
        assert strategy.get_indicator_by_name('ATR') is None

    def test_disambiguates_multiple_same_type_indicators(self):
        from algo_trading_engine.indicators.sma_indicator import SMAIndicator
        sma_20 = SMAIndicator(period=20)
        sma_50 = SMAIndicator(period=50)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(sma_20)
        strategy.add_indicator(sma_50)
        assert strategy.get_indicator_by_name('SMA_20') is sma_20
        assert strategy.get_indicator_by_name('SMA_50') is sma_50

    def test_works_alongside_get_indicator(self):
        atr = ATRIndicator(period=14)
        mock = MockIndicator(name='Custom')
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        strategy.add_indicator(mock)
        assert strategy.get_indicator(ATRIndicator) is atr
        assert strategy.get_indicator_by_name('ATR') is atr
        assert strategy.get_indicator_by_name('Custom') is mock

class TestStrategyAddIndicator:
    """Test cases for add_indicator method"""

    def test_add_indicator_to_empty_list(self):
        """Test adding an indicator to empty indicators list"""
        strategy = ConcreteTestStrategy()
        assert len(strategy.indicators) == 0
        atr = ATRIndicator(period=14)
        strategy.add_indicator(atr)
        assert len(strategy.indicators) == 1
        assert strategy.indicators[0] is atr

    def test_add_indicator_to_existing_list(self):
        """Test adding an indicator to existing indicators list"""
        mock_indicator = MockIndicator(name='MockIndicator')
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(mock_indicator)
        assert len(strategy.indicators) == 1
        atr = ATRIndicator(period=14)
        strategy.add_indicator(atr)
        assert len(strategy.indicators) == 2
        assert strategy.indicators[0] is mock_indicator
        assert strategy.indicators[1] is atr

    def test_add_multiple_indicators(self):
        """Test adding multiple indicators sequentially"""
        strategy = ConcreteTestStrategy()
        atr1 = ATRIndicator(period=14)
        atr2 = ATRIndicator(period=20)
        mock_indicator = MockIndicator(name='MockIndicator')
        strategy.add_indicator(atr1)
        strategy.add_indicator(atr2)
        strategy.add_indicator(mock_indicator)
        assert len(strategy.indicators) == 3
        assert strategy.indicators[0] is atr1
        assert strategy.indicators[1] is atr2
        assert strategy.indicators[2] is mock_indicator

    def test_add_indicator_raises_error_for_non_indicator(self):
        """Test add_indicator raises TypeError for non-Indicator objects"""
        strategy = ConcreteTestStrategy()
        with pytest.raises(TypeError, match='Expected Indicator instance'):
            strategy.add_indicator('not an indicator')
        with pytest.raises(TypeError, match='Expected Indicator instance'):
            strategy.add_indicator(42)
        with pytest.raises(TypeError, match='Expected Indicator instance'):
            strategy.add_indicator(None)

    def test_add_indicator_then_retrieve_with_get_indicator(self):
        """Test adding indicator and then retrieving it with get_indicator"""
        strategy = ConcreteTestStrategy()
        atr = ATRIndicator(period=14)
        strategy.add_indicator(atr)
        retrieved_atr = strategy.get_indicator(ATRIndicator)
        assert retrieved_atr is atr
        assert retrieved_atr.period == 14

    def test_add_indicator_then_update_works(self):
        """Test that added indicator gets updated correctly"""
        strategy = ConcreteTestStrategy()
        atr = ATRIndicator(period=5)
        strategy.add_indicator(atr)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'Open': [100 + i for i in range(10)], 'High': [110 + i for i in range(10)], 'Low': [90 + i for i in range(10)], 'Close': [100 + i for i in range(10)], 'Volume': [1000000] * 10}, index=dates)
        strategy.set_data(data)
        assert atr.value is None
        atr.update(datetime(2024, 1, 6), data)
        assert atr.value is not None
        assert atr.value > 0

    def test_add_indicator_in_strategy_init(self):
        """Test adding indicators in strategy __init__ method"""

        class StrategyWithDynamicIndicators(Strategy):

            def __init__(self):
                super().__init__()
                self.add_indicator(ATRIndicator(period=14))
                self.add_indicator(MockIndicator(name='Mock'))

            def on_new_date(self, date, positions, add_position, remove_position):
                pass

            def on_end(self, positions, remove_position, date):
                pass

            def validate_data(self, data):
                return True
        strategy = StrategyWithDynamicIndicators()
        assert len(strategy.indicators) == 2
        assert isinstance(strategy.indicators[0], ATRIndicator)
        assert isinstance(strategy.indicators[1], MockIndicator)

def _hourly_ohlcv(n: int=10) -> pd.DataFrame:
    dates = pd.date_range(start='2026-05-19 09:30:00', periods=n, freq='h')
    return pd.DataFrame({'Open': [500 + i for i in range(n)], 'High': [510 + i for i in range(n)], 'Low': [490 + i for i in range(n)], 'Close': [500 + i for i in range(n)], 'Volume': [1000000] * n}, index=dates)

def _daily_ohlcv(n: int=30) -> pd.DataFrame:
    dates = pd.date_range(start='2026-04-01', periods=n, freq='D')
    return pd.DataFrame({'Open': [500 + i for i in range(n)], 'High': [510 + i for i in range(n)], 'Low': [490 + i for i in range(n)], 'Close': [500 + i for i in range(n)], 'Volume': [1000000] * n}, index=dates)

class _PrecomputeTestEngine(TradingEngine):
    """Minimal engine harness for indicator precompute tests."""

    def run(self):
        return True

    def get_positions(self):
        return []

    @property
    def strategy(self):
        return self._strategy

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


def _engine_precompute(strategy: Strategy, data: pd.DataFrame) -> None:
    config = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")
    _PrecomputeTestEngine(strategy, data, config)


class TestEnginePrecomputeIndicators:
    """Tests for TradingEngine.precompute_indicators()."""

    def test_raises_when_no_data(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(ATRIndicator(period=3))
        config = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")
        with pytest.raises(ValueError, match="Cannot precompute indicators without data"):
            _PrecomputeTestEngine(strategy, pd.DataFrame(), config)

    def test_raises_when_empty_data(self):
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(ATRIndicator(period=3))
        config = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")
        with pytest.raises(ValueError, match="Cannot precompute indicators without data"):
            _PrecomputeTestEngine(strategy, pd.DataFrame(), config)

    def test_no_op_when_no_indicators(self):
        strategy = ConcreteTestStrategy()
        strategy.set_data(_hourly_ohlcv())
        _engine_precompute(strategy, strategy.data)

    def test_multiple_indicators_all_precomputed(self):
        atr = ATRIndicator(period=3)
        sma = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        strategy = ConcreteTestStrategy()
        strategy.add_indicator(atr)
        strategy.add_indicator(sma)
        data = _hourly_ohlcv()
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        test_bar = data.index[6]
        assert atr.get_value_at(test_bar) is not None
        assert sma.get_value_at(test_bar) is not None


class TestEnginePrecomputeBacktestParity:
    """Vectorized precompute values must match per-bar update iteration."""

    def test_atr_values_match_backtest_iteration(self):
        data = _hourly_ohlcv()
        atr_bt = ATRIndicator(period=3)
        bt = ConcreteTestStrategy()
        bt.add_indicator(atr_bt)
        bt.set_data(data)
        for date in data.index:
            atr_bt.update(date, data)
        atr_pc = ATRIndicator(period=3)
        pt = ConcreteTestStrategy()
        pt.add_indicator(atr_pc)
        pt.set_data(data)
        _engine_precompute(pt, data)
        for bar in data.index[atr_bt.warm_up_period - 1 :]:
            assert atr_bt.get_value_at(bar) == pytest.approx(atr_pc.get_value_at(bar))

    def test_sma_values_match_backtest_iteration(self):
        data = _hourly_ohlcv()
        sma_bt = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        bt = ConcreteTestStrategy()
        bt.add_indicator(sma_bt)
        bt.set_data(data)
        for date in data.index:
            sma_bt.update(date, data)
        sma_pc = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        pt = ConcreteTestStrategy()
        pt.add_indicator(sma_pc)
        pt.set_data(data)
        _engine_precompute(pt, data)
        for bar in data.index[sma_bt.warm_up_period - 1 :]:
            assert sma_bt.get_value_at(bar) == pytest.approx(sma_pc.get_value_at(bar))

class TestStrategyHistoricalLookupWithWarmUp:
    """Strategy that looks up indicator values at historical bar dates
    (e.g. ATR K bars ago) must succeed after warm-up and fail without it."""

    class _LookbackStrategy(Strategy):

        def __init__(self, atr, lookback_bars=4):
            super().__init__()
            self.add_indicator(atr)
            self.lookback_bars = lookback_bars
            self.historical_atr_value = 'NOT_CHECKED'
            self.current_atr_value = 'NOT_CHECKED'

        def on_new_date(self, date, positions, add_position, remove_position):
            super().on_new_date(date, positions, add_position, remove_position)
            atr = self.indicators[0]
            self.current_atr_value = atr.get_value_at(date)
            bar_idx = self.data.index.get_loc(date)
            if bar_idx >= self.lookback_bars:
                hist_date = self.data.index[bar_idx - self.lookback_bars]
                self.historical_atr_value = atr.get_value_at(hist_date)

        def on_end(self, positions, remove_position, date):
            pass

        def validate_data(self, data):
            return True

    def test_backtest_iteration_historical_lookup_succeeds(self):
        atr = ATRIndicator(period=3)
        data = _hourly_ohlcv()
        strategy = self._LookbackStrategy(atr, lookback_bars=4)
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        for date in data.index:
            strategy.on_new_date(date, (), Mock(), Mock())
        assert strategy.current_atr_value is not None
        assert strategy.historical_atr_value is not None

    def test_single_on_new_date_without_precompute_lookup_returns_none(self):
        atr = ATRIndicator(period=3)
        data = _hourly_ohlcv()
        strategy = self._LookbackStrategy(atr, lookback_bars=4)
        strategy.set_data(data)
        strategy.on_new_date(data.index[-1], (), Mock(), Mock())
        assert strategy.current_atr_value is None
        assert strategy.historical_atr_value is None

    def test_precompute_then_single_on_new_date_historical_lookup_succeeds(self):
        atr = ATRIndicator(period=3)
        data = _hourly_ohlcv()
        strategy = self._LookbackStrategy(atr, lookback_bars=4)
        strategy.set_data(data)
        _engine_precompute(strategy, data)
        strategy.on_new_date(data.index[-1], (), Mock(), Mock())
        assert strategy.current_atr_value is not None
        assert strategy.historical_atr_value is not None

    def test_daily_bars_without_warmup_historical_lookup_returns_none(self):
        atr = ATRIndicator(period=5)
        data = _daily_ohlcv(20)
        strategy = self._LookbackStrategy(atr, lookback_bars=4)
        strategy.set_data(data)
        strategy.on_new_date(data.index[-1], (), Mock(), Mock())
        assert strategy.current_atr_value is None
        assert strategy.historical_atr_value is None
