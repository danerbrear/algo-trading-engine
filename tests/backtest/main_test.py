"""
Integration tests for volume validation in backtest scenarios.

This module tests volume validation integration with BacktestEngine and Strategy components.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock
import pandas as pd
from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.backtest.config import VolumeConfig
from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.vo import Position, create_position
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import Option, OptionType

class MockStrategy(Strategy):
    """Mock strategy for testing volume validation integration."""

    def __init__(self, options_handler=None, symbol='SPY'):
        super().__init__()
        self.options_handler = options_handler
        self.symbol = symbol
        dates = pd.date_range('2024-01-01', periods=3)
        self.data = pd.DataFrame({'Open': [99.0, 100.0, 101.0], 'High': [102.0, 103.0, 104.0], 'Low': [98.0, 99.0, 100.0], 'Close': [100.0, 101.0, 102.0], 'Volume': [1000000, 1100000, 1200000], 'Returns': [0.01, 0.01, 0.01], 'Log_Returns': [0.00995, 0.00995, 0.00995], 'Volatility': [0.15, 0.15, 0.15], 'RSI': [50.0, 50.0, 50.0], 'MACD_Hist': [0.0, 0.0, 0.0], 'Volume_Ratio': [1.0, 1.0, 1.0], 'Market_State': [0, 0, 0], 'Put_Call_Ratio': [0.5, 0.5, 0.5], 'Option_Volume_Ratio': [1.0, 1.0, 1.0], 'Days_Until_Next_CPI': [30, 29, 28], 'Days_Since_Last_CPI': [15, 16, 17], 'Days_Until_Next_CC': [45, 44, 43], 'Days_Since_Last_CC': [10, 11, 12], 'Days_Until_Next_FFR': [60, 59, 58], 'Days_Since_Last_FFR': [5, 6, 7]}, index=dates)

    def on_new_date(self, date, positions, add_position, remove_position):
        """Mock strategy that creates positions with volume validation."""
        if len(positions) == 0:
            option1 = Mock(spec=Option)
            option1.symbol = 'SPY240315C00500000'
            option1.volume = 15
            option1.strike = 500.0
            option1.expiration = '2024-03-15'
            option1.option_type = OptionType.CALL
            option2 = Mock(spec=Option)
            option2.symbol = 'SPY240315C00510000'
            option2.volume = 20
            option2.strike = 510.0
            option2.expiration = '2024-03-15'
            option2.option_type = OptionType.CALL
            position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[option1, option2])
            add_position(position)

    def on_end(self, positions, remove_position, date):
        """Mock strategy end method."""
        pass

    def validate_data(self, data):
        """Mock validate_data method."""
        return True

class TestVolumeValidationIntegration:
    """Integration tests for volume validation in backtest scenarios."""

    def test_backtest_engine_with_volume_validation_enabled(self):
        """Test BacktestEngine with volume validation enabled."""
        strategy = MockStrategy()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 1
        assert engine.volume_stats.options_checked == 2
        assert engine.volume_stats.positions_rejected_volume == 0

    def test_backtest_engine_with_volume_validation_disabled(self):
        """Test BacktestEngine with volume validation disabled."""
        strategy = MockStrategy()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 1
        assert engine.volume_stats.options_checked == 0
        assert engine.volume_stats.positions_rejected_volume == 0

    def test_backtest_engine_rejects_position_with_insufficient_volume(self):
        """Test BacktestEngine rejects position when options have insufficient volume."""

        class MockStrategyWithLowVolume(MockStrategy):

            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with insufficient volume."""
                if len(positions) == 0:
                    option1 = Mock(spec=Option)
                    option1.symbol = 'SPY240315C00500000'
                    option1.volume = 5
                    option1.strike = 500.0
                    option1.expiration = '2024-03-15'
                    option1.option_type = OptionType.CALL
                    option2 = Mock(spec=Option)
                    option2.symbol = 'SPY240315C00510000'
                    option2.volume = 20
                    option2.strike = 510.0
                    option2.expiration = '2024-03-15'
                    option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[option1, option2])
                    add_position(position)

        def on_end(_self, _positions, _remove_position, _date):
            """Mock strategy end method."""
            pass
        strategy = MockStrategyWithLowVolume()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 0
        assert engine.volume_stats.options_checked == 3
        assert engine.volume_stats.positions_rejected_volume == 3

    def test_backtest_engine_rejects_position_with_missing_volume_data(self):
        """Test BacktestEngine rejects position when options have missing volume data."""

        class MockStrategyWithMissingVolume(MockStrategy):

            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with missing volume data."""
                if len(positions) == 0:
                    option1 = Mock(spec=Option)
                    option1.symbol = 'SPY240315C00500000'
                    option1.volume = None
                    option1.strike = 500.0
                    option1.expiration = '2024-03-15'
                    option1.option_type = OptionType.CALL
                    option2 = Mock(spec=Option)
                    option2.symbol = 'SPY240315C00510000'
                    option2.volume = 20
                    option2.strike = 510.0
                    option2.expiration = '2024-03-15'
                    option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[option1, option2])
                    add_position(position)

        def on_end(_self, _positions, _remove_position, _date):
            """Mock strategy end method."""
            pass
        strategy = MockStrategyWithMissingVolume()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 0
        assert engine.volume_stats.options_checked == 3
        assert engine.volume_stats.positions_rejected_volume == 3

    def test_volume_config_custom_min_volume(self):
        """Test BacktestEngine with custom minimum volume threshold."""

        class MockStrategyWithCustomVolume(MockStrategy):

            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with specific volume levels."""
                if len(positions) == 0:
                    option1 = Mock(spec=Option)
                    option1.symbol = 'SPY240315C00500000'
                    option1.volume = 8
                    option1.strike = 500.0
                    option1.expiration = '2024-03-15'
                    option1.option_type = OptionType.CALL
                    option2 = Mock(spec=Option)
                    option2.symbol = 'SPY240315C00510000'
                    option2.volume = 20
                    option2.strike = 510.0
                    option2.expiration = '2024-03-15'
                    option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[option1, option2])
                    add_position(position)

        def on_end(_self, _positions, _remove_position, _date):
            """Mock strategy end method."""
            pass
        strategy = MockStrategyWithCustomVolume()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=15, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 0
        assert engine.volume_stats.options_checked == 3
        assert engine.volume_stats.positions_rejected_volume == 3

    def test_volume_statistics_reporting(self):
        """Test that volume statistics are properly tracked and reported."""
        strategy = MockStrategy()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert engine.volume_stats.options_checked == 2
        assert engine.volume_stats.positions_rejected_volume == 0
        summary = engine.volume_stats.get_summary()
        assert summary['positions_rejected_volume'] == 0
        assert summary['options_checked'] == 2
        assert summary['rejection_rate'] == 0.0

    def test_backtest_engine_with_position_without_spread_options(self):
        """Test BacktestEngine handles positions without spread_options gracefully."""

        class MockStrategyWithoutSpreadOptions(MockStrategy):

            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions without spread_options."""
                if len(positions) == 0:
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.LONG_CALL, strike_price=500.0, entry_date=date, entry_price=100.0, spread_options=None)
                    add_position(position)

        def on_end(_self, _positions, _remove_position, _date):
            """Mock strategy end method."""
            pass
        strategy = MockStrategyWithoutSpreadOptions()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        success = engine.run()
        assert success is True
        assert len(engine.positions) == 1
        assert engine.volume_stats.options_checked == 0
        assert engine.volume_stats.positions_rejected_volume == 0

    def test_position_closure_volume_validation(self):
        """Test that position closures are skipped when volume is insufficient."""
        data = pd.DataFrame({'Close': [100, 101, 102, 103, 104], 'Volume': [1000, 1100, 1200, 1300, 1400], 'Returns': [0.01, 0.01, 0.01, 0.01, 0.01], 'Log_Returns': [0.01, 0.01, 0.01, 0.01, 0.01], 'Volatility': [0.02, 0.02, 0.02, 0.02, 0.02], 'RSI': [50, 50, 50, 50, 50], 'MACD_Hist': [0, 0, 0, 0, 0], 'Volume_Ratio': [1.0, 1.0, 1.0, 1.0, 1.0], 'Market_State': [0, 0, 0, 0, 0], 'Put_Call_Ratio': [1.0, 1.0, 1.0, 1.0, 1.0], 'Option_Volume_Ratio': [1.0, 1.0, 1.0, 1.0, 1.0], 'Days_Until_Next_CPI': [30, 29, 28, 27, 26], 'Days_Since_Last_CPI': [5, 6, 7, 8, 9], 'Days_Until_Next_CC': [30, 29, 28, 27, 26], 'Days_Since_Last_CC': [5, 6, 7, 8, 9], 'Days_Until_Next_FFR': [30, 29, 28, 27, 26], 'Days_Since_Last_FFR': [5, 6, 7, 8, 9]}, index=pd.date_range('2024-01-01', periods=5))
        strategy = MockStrategy()
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=100000, volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True))
        atm_option = Option(ticker='SPY', symbol='SPY240119C00100000', strike=100.0, expiration='2024-01-19', option_type=OptionType.CALL, last_price=1.5, volume=5, open_interest=100, bid=1.45, ask=1.55)
        otm_option = Option(ticker='SPY', symbol='SPY240119C00105000', strike=105.0, expiration='2024-01-19', option_type=OptionType.CALL, last_price=0.75, volume=5, open_interest=100, bid=0.7, ask=0.8)
        position = create_position(symbol='SPY', expiration_date=datetime(2024, 1, 19), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=100.0, entry_date=datetime(2024, 1, 1), entry_price=0.75, spread_options=[atm_option, otm_option])
        position.set_quantity(1)
        engine.positions.append(position)
        initial_capital = engine.capital
        initial_positions_count = len(engine.positions)
        engine._remove_position(date=datetime(2024, 1, 2), position=position, exit_price=1.25, current_volumes=[5, 5])
        assert len(engine.positions) == initial_positions_count, 'Position should not be closed due to insufficient volume'
        assert engine.capital == initial_capital, 'Capital should not change when position closure is skipped'
        volume_summary = engine.volume_stats.get_summary()
        assert volume_summary['positions_rejected_closure_volume'] == 1, 'Should track rejected closure'
        assert volume_summary['skipped_closures'] == 1, 'Should track skipped closure'
        assert volume_summary['options_checked'] >= 1, 'Should have checked at least one option'

class TestUniversalCloseCallback:
    """Tests that strategy callback on_remove_position_success is invoked when a universal close condition is met."""

    def test_on_remove_position_success_called_when_universal_close_expiration(self):
        """When a position expires (days to expiration < 1), the engine closes it and invokes the strategy callback."""
        callback_invocations = []
        first_date = datetime(2024, 1, 1)

        class MockStrategyWithCallback(MockStrategy):

            def on_new_date(self, date, positions, add_position, remove_position):
                if len(positions) == 0 and date == first_date:
                    option1 = Mock(spec=Option)
                    option1.symbol = 'SPY240101C00500000'
                    option1.ticker = 'O:SPY240101C00500000'
                    option1.volume = 15
                    option1.strike = 500.0
                    option1.expiration = '2024-01-01'
                    option1.last_price = 1.5
                    option1.option_type = OptionType.CALL
                    option2 = Mock(spec=Option)
                    option2.symbol = 'SPY240101C00510000'
                    option2.ticker = 'O:SPY240101C00510000'
                    option2.volume = 20
                    option2.strike = 510.0
                    option2.expiration = '2024-01-01'
                    option2.last_price = 0.75
                    option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 1, 1), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[option1, option2])
                    add_position(position)

            def on_remove_position_success(self, date, position, exit_price, underlying_price=None, current_volumes=None):
                callback_invocations.append({'date': date, 'position': position, 'exit_price': exit_price, 'underlying_price': underlying_price, 'current_volumes': current_volumes})
                super().on_remove_position_success(date, position, exit_price, underlying_price, current_volumes)
        strategy = MockStrategyWithCallback()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False))
        success = engine.run()
        assert success is True
        assert len(callback_invocations) == 1, 'on_remove_position_success should be called exactly once (universal close due to expiration)'
        inv = callback_invocations[0]
        assert inv['date'] == datetime(2024, 1, 1)
        assert inv['exit_price'] == 0.0
        assert inv['underlying_price'] == 100.0
        assert inv['position'] is not None

    def test_on_remove_position_success_called_when_universal_close_profit_target(self):
        """When profit target is hit, the engine closes the position and invokes the strategy callback."""
        from decimal import Decimal
        from algo_trading_engine.dto import OptionBarDTO
        callback_invocations = []

        def make_bar(ticker: str, close_price: float, expiration_date: datetime) -> OptionBarDTO:
            return OptionBarDTO(ticker=ticker, timestamp=expiration_date, open_price=Decimal(str(close_price)), high_price=Decimal(str(close_price)), low_price=Decimal(str(close_price)), close_price=Decimal(str(close_price)), volume=15, volume_weighted_avg_price=Decimal(str(close_price)), number_of_transactions=100, adjusted=True)
        first_date = datetime(2024, 1, 1)

        class MockStrategyProfitTarget(MockStrategy):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.profit_target = 0.5
                self._option1 = None
                self._option2 = None

            def on_new_date(self, date, positions, add_position, remove_position):
                if len(positions) == 0 and date == first_date:
                    self._option1 = Mock(spec=Option)
                    self._option1.symbol = 'SPY240315C00500000'
                    self._option1.ticker = 'O:SPY240315C00500000'
                    self._option1.volume = 15
                    self._option1.strike = 500.0
                    self._option1.expiration = '2024-03-15'
                    self._option1.last_price = 1.5
                    self._option1.option_type = OptionType.CALL
                    self._option2 = Mock(spec=Option)
                    self._option2.symbol = 'SPY240315C00510000'
                    self._option2.ticker = 'O:SPY240315C00510000'
                    self._option2.volume = 20
                    self._option2.strike = 510.0
                    self._option2.expiration = '2024-03-15'
                    self._option2.last_price = 0.75
                    self._option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[self._option1, self._option2])
                    add_position(position)

            def get_option_bar(self, option, date, multiplier=1, timespan=None):
                _ = date, multiplier, timespan
                if option is self._option1:
                    return make_bar(option.ticker, 1.0, date)
                if option is self._option2:
                    return make_bar(option.ticker, 0.25, date)
                return None

            def on_remove_position_success(self, date, position, exit_price, underlying_price=None, current_volumes=None):
                callback_invocations.append({'date': date, 'position': position, 'exit_price': exit_price, 'underlying_price': underlying_price, 'current_volumes': current_volumes})
                super().on_remove_position_success(date, position, exit_price, underlying_price, current_volumes)
        strategy = MockStrategyProfitTarget()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False))
        success = engine.run()
        assert success is True
        assert len(callback_invocations) == 1, 'on_remove_position_success should be called exactly once (universal close due to profit target)'
        inv = callback_invocations[0]
        assert inv['exit_price'] == 0.75
        assert inv['position'] is not None

    def test_universal_profit_target_skipped_when_excluded_from_config(self):
        """When PROFIT_TARGET is omitted from universal_close_conditions, engine does not close on PT."""
        from decimal import Decimal
        from algo_trading_engine.dto import OptionBarDTO
        from algo_trading_engine.enums import UniversalCloseCondition
        callback_invocations = []

        def make_bar(ticker: str, close_price: float, expiration_date: datetime) -> OptionBarDTO:
            return OptionBarDTO(ticker=ticker, timestamp=expiration_date, open_price=Decimal(str(close_price)), high_price=Decimal(str(close_price)), low_price=Decimal(str(close_price)), close_price=Decimal(str(close_price)), volume=15, volume_weighted_avg_price=Decimal(str(close_price)), number_of_transactions=100, adjusted=True)
        first_date = datetime(2024, 1, 1)

        class MockStrategyProfitTargetExcluded(MockStrategy):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.profit_target = 0.5
                self.universal_close_conditions = frozenset({UniversalCloseCondition.ASSIGNMENT, UniversalCloseCondition.STOP_LOSS})
                self._option1 = None
                self._option2 = None

            def on_new_date(self, date, positions, add_position, remove_position):
                if len(positions) == 0 and date == first_date:
                    self._option1 = Mock(spec=Option)
                    self._option1.symbol = 'SPY240315C00500000'
                    self._option1.ticker = 'O:SPY240315C00500000'
                    self._option1.volume = 15
                    self._option1.strike = 500.0
                    self._option1.expiration = '2024-03-15'
                    self._option1.last_price = 1.5
                    self._option1.option_type = OptionType.CALL
                    self._option2 = Mock(spec=Option)
                    self._option2.symbol = 'SPY240315C00510000'
                    self._option2.ticker = 'O:SPY240315C00510000'
                    self._option2.volume = 20
                    self._option2.strike = 510.0
                    self._option2.expiration = '2024-03-15'
                    self._option2.last_price = 0.75
                    self._option2.option_type = OptionType.CALL
                    position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=2.5, spread_options=[self._option1, self._option2])
                    add_position(position)

            def get_option_bar(self, option, date, multiplier=1, timespan=None):
                _ = date, multiplier, timespan
                if option is self._option1:
                    return make_bar(option.ticker, 1.0, date)
                if option is self._option2:
                    return make_bar(option.ticker, 0.25, date)
                return None

            def on_remove_position_success(self, date, position, exit_price, underlying_price=None, current_volumes=None):
                callback_invocations.append({'date': date, 'position': position, 'exit_price': exit_price})
                super().on_remove_position_success(date, position, exit_price, underlying_price, current_volumes)
        strategy = MockStrategyProfitTargetExcluded()
        data = strategy.data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False))
        success = engine.run()
        assert success is True
        assert len(callback_invocations) == 0, 'Universal profit target should not close when PROFIT_TARGET is excluded'
        assert len(engine.positions) == 1, 'Position should remain open'

def _minimal_spy_df(periods: int) -> pd.DataFrame:
    """Minimal OHLCV frame for backtest run() (strategy.validate_data may still pass)."""
    dates = pd.date_range('2024-01-01', periods=periods, freq='B')
    return pd.DataFrame({'Open': [100.0] * periods, 'High': [102.0] * periods, 'Low': [98.0] * periods, 'Close': [100.0 + i for i in range(periods)], 'Volume': [1000000] * periods}, index=dates)

class OnNewDateOrderStrategy(Strategy):
    """Records on_new_date call order; indicators are fed full history in each update."""

    def __init__(self):
        super().__init__()
        self.dates_called_in_on_new_date: list = []

    def on_new_date(self, date, positions, add_position, remove_position):
        self.dates_called_in_on_new_date.append(date)

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return True

class TestBacktestOnNewDateDateRange:
    """Backtest calls on_new_date once per bar between engine start_date and end_date (inclusive)."""

    def test_on_new_date_visits_every_bar_in_range(self):
        n = 5
        data = _minimal_spy_df(n)
        strategy = OnNewDateOrderStrategy()
        strategy.data = data
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=data.index[0].to_pydatetime(), end_date=data.index[-1].to_pydatetime(), volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False), enable_progress_tracking=False, quiet_mode=True)
        assert engine.run() is True
        assert len(strategy.dates_called_in_on_new_date) == n
        assert strategy.dates_called_in_on_new_date[0] == data.index[0]
        assert strategy.dates_called_in_on_new_date == list(data.index)

class TestCreditSpreadAssignmentCapital:
    """Capital accounting when a credit spread is closed by assignment at expiration.

    On the assignment path the strategy passes a placeholder exit_price of 0.0. Capital must
    still reflect the polymorphic assignment return (intrinsic settlement), not the placeholder.
    Capital is starting equity + realized P&L for every strategy type.
    """

    def _make_engine(self):
        strategy = MockStrategy()
        return BacktestEngine(data=strategy.data, strategy=strategy, initial_capital=10000, volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False), enable_progress_tracking=False, quiet_mode=True)

    def _make_put_credit_spread(self, entry_price: float, short_strike: float, long_strike: float):
        short_put = Option(ticker='SPY', symbol='SPY240119P00658000', strike=short_strike, expiration='2024-01-19', option_type=OptionType.PUT, last_price=1.0, volume=100, open_interest=100, bid=0.95, ask=1.05)
        long_put = Option(ticker='SPY', symbol='SPY240119P00654000', strike=long_strike, expiration='2024-01-19', option_type=OptionType.PUT, last_price=0.5, volume=100, open_interest=100, bid=0.45, ask=0.55)
        position = create_position(symbol='SPY', expiration_date=datetime(2024, 1, 19), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=short_strike, entry_date=datetime(2024, 1, 1), entry_price=entry_price, spread_options=[short_put, long_put])
        position.set_quantity(1)
        return position

    def test_assignment_in_the_money_debits_intrinsic_loss(self):
        """Fully-ITM put credit spread at expiration loses (width - credit) x 100."""
        engine = self._make_engine()
        position = self._make_put_credit_spread(entry_price=1.06, short_strike=658.0, long_strike=654.0)
        engine.positions.append(position)
        engine._remove_position(date=datetime(2024, 1, 19), position=position, exit_price=0.0, underlying_price=647.65)
        expected_return = -294.0
        assert engine.closed_positions[-1]['return_dollars'] == pytest.approx(expected_return)
        assert engine.capital == pytest.approx(10000 + expected_return)

    def test_assignment_out_of_the_money_keeps_full_credit(self):
        """Put credit spread expiring worthless keeps the full credit."""
        engine = self._make_engine()
        position = self._make_put_credit_spread(entry_price=1.06, short_strike=658.0, long_strike=654.0)
        engine.positions.append(position)
        credit = position.entry_price * position.quantity * 100
        engine._remove_position(date=datetime(2024, 1, 19), position=position, exit_price=0.0, underlying_price=670.0)
        assert engine.closed_positions[-1]['return_dollars'] == pytest.approx(credit)
        assert engine.capital == pytest.approx(10000 + credit)
if __name__ == '__main__':
    pytest.main([__file__])
