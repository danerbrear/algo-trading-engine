"""
Integration tests for volume validation in backtest scenarios.

This module tests volume validation integration with BacktestEngine and Strategy components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd

from src.backtest.main import BacktestEngine
from src.backtest.config import VolumeConfig, VolumeStats
from src.backtest.models import Strategy, Position, StrategyType
from src.common.models import Option, OptionType


class MockStrategy(Strategy):
    """Mock strategy for testing volume validation integration."""
    
    def __init__(self, options_handler=None):
        super().__init__()
        self.options_handler = options_handler
        # Create comprehensive mock data with all required columns
        dates = pd.date_range('2024-01-01', periods=3)
        self.data = pd.DataFrame({
            'Open': [99.0, 100.0, 101.0],
            'High': [102.0, 103.0, 104.0],
            'Low': [98.0, 99.0, 100.0],
            'Close': [100.0, 101.0, 102.0],
            'Volume': [1000000, 1100000, 1200000],
            'Returns': [0.01, 0.01, 0.01],
            'Log_Returns': [0.00995, 0.00995, 0.00995],
            'Volatility': [0.15, 0.15, 0.15],
            'RSI': [50.0, 50.0, 50.0],
            'MACD_Hist': [0.0, 0.0, 0.0],
            'Volume_Ratio': [1.0, 1.0, 1.0],
            'Market_State': [0, 0, 0],
            'Put_Call_Ratio': [0.5, 0.5, 0.5],
            'Option_Volume_Ratio': [1.0, 1.0, 1.0],
            'Days_Until_Next_CPI': [30, 29, 28],
            'Days_Since_Last_CPI': [15, 16, 17],
            'Days_Until_Next_CC': [45, 44, 43],
            'Days_Since_Last_CC': [10, 11, 12],
            'Days_Until_Next_FFR': [60, 59, 58],
            'Days_Since_Last_FFR': [5, 6, 7]
        }, index=dates)
    
    def on_new_date(self, date, positions, add_position, remove_position):
        """Mock strategy that creates positions with volume validation."""
        if len(positions) == 0:
            # Create a position with options that have volume data
            option1 = Mock(spec=Option)
            option1.symbol = "SPY240315C00500000"
            option1.volume = 15
            option1.strike = 500.0
            option1.expiration = "2024-03-15"
            option1.option_type = Mock()
            option1.option_type.value = "C"
            
            option2 = Mock(spec=Option)
            option2.symbol = "SPY240315C00510000"
            option2.volume = 20
            option2.strike = 510.0
            option2.expiration = "2024-03-15"
            option2.option_type = Mock()
            option2.option_type.value = "C"
            
            position = Position(
                symbol="SPY",
                expiration_date=datetime(2024, 3, 15),
                strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date,
                entry_price=2.50,
                spread_options=[option1, option2]
            )
            
            add_position(position)


class TestVolumeValidationIntegration:
    """Integration tests for volume validation in backtest scenarios."""
    
    def test_backtest_engine_with_volume_validation_enabled(self):
        """Test BacktestEngine with volume validation enabled."""
        strategy = MockStrategy()
        data = strategy.data
        
        # Create BacktestEngine with volume validation enabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 1  # Position should be added
        assert engine.volume_stats.options_checked == 2  # Two options checked
        assert engine.volume_stats.positions_rejected_volume == 0  # No rejections
    
    def test_backtest_engine_with_volume_validation_disabled(self):
        """Test BacktestEngine with volume validation disabled."""
        strategy = MockStrategy()
        data = strategy.data
        
        # Create BacktestEngine with volume validation disabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(enable_volume_validation=False)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 1  # Position should be added
        assert engine.volume_stats.options_checked == 0  # No options checked
        assert engine.volume_stats.positions_rejected_volume == 0  # No rejections
    
    def test_backtest_engine_rejects_position_with_insufficient_volume(self):
        """Test BacktestEngine rejects position when options have insufficient volume."""
        
        class MockStrategyWithLowVolume(MockStrategy):
            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with insufficient volume."""
                if len(positions) == 0:
                    # Create a position with options that have insufficient volume
                    option1 = Mock(spec=Option)
                    option1.symbol = "SPY240315C00500000"
                    option1.volume = 5  # Below minimum
                    option1.strike = 500.0
                    option1.expiration = "2024-03-15"
                    option1.option_type = Mock()
                    option1.option_type.value = "C"
                    
                    option2 = Mock(spec=Option)
                    option2.symbol = "SPY240315C00510000"
                    option2.volume = 20  # Above minimum
                    option2.strike = 510.0
                    option2.expiration = "2024-03-15"
                    option2.option_type = Mock()
                    option2.option_type.value = "C"
                    
                    position = Position(
                        symbol="SPY",
                        expiration_date=datetime(2024, 3, 15),
                        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                        strike_price=500.0,
                        entry_date=date,
                        entry_price=2.50,
                        spread_options=[option1, option2]
                    )
                    
                    add_position(position)
        
        strategy = MockStrategyWithLowVolume()
        data = strategy.data
        
        # Create BacktestEngine with volume validation enabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 0  # Position should be rejected
        assert engine.volume_stats.options_checked == 3  # Three options checked (3 days)
        assert engine.volume_stats.positions_rejected_volume == 3  # Three rejections (3 days)
    
    def test_backtest_engine_rejects_position_with_missing_volume_data(self):
        """Test BacktestEngine rejects position when options have missing volume data."""
        
        class MockStrategyWithMissingVolume(MockStrategy):
            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with missing volume data."""
                if len(positions) == 0:
                    # Create a position with options that have missing volume data
                    option1 = Mock(spec=Option)
                    option1.symbol = "SPY240315C00500000"
                    option1.volume = None  # Missing volume data
                    option1.strike = 500.0
                    option1.expiration = "2024-03-15"
                    option1.option_type = Mock()
                    option1.option_type.value = "C"
                    
                    option2 = Mock(spec=Option)
                    option2.symbol = "SPY240315C00510000"
                    option2.volume = 20  # Above minimum
                    option2.strike = 510.0
                    option2.expiration = "2024-03-15"
                    option2.option_type = Mock()
                    option2.option_type.value = "C"
                    
                    position = Position(
                        symbol="SPY",
                        expiration_date=datetime(2024, 3, 15),
                        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                        strike_price=500.0,
                        entry_date=date,
                        entry_price=2.50,
                        spread_options=[option1, option2]
                    )
                    
                    add_position(position)
        
        strategy = MockStrategyWithMissingVolume()
        data = strategy.data
        
        # Create BacktestEngine with volume validation enabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 0  # Position should be rejected
        assert engine.volume_stats.options_checked == 3  # Three options checked (3 days)
        assert engine.volume_stats.positions_rejected_volume == 3  # Three rejections (3 days)
    
    def test_volume_config_custom_min_volume(self):
        """Test BacktestEngine with custom minimum volume threshold."""
        
        class MockStrategyWithCustomVolume(MockStrategy):
            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions with specific volume levels."""
                if len(positions) == 0:
                    # Create a position with options that have volume = 8
                    option1 = Mock(spec=Option)
                    option1.symbol = "SPY240315C00500000"
                    option1.volume = 8  # Below min_volume=15
                    option1.strike = 500.0
                    option1.expiration = "2024-03-15"
                    option1.option_type = Mock()
                    option1.option_type.value = "C"
                    
                    option2 = Mock(spec=Option)
                    option2.symbol = "SPY240315C00510000"
                    option2.volume = 20  # Above min_volume=15
                    option2.strike = 510.0
                    option2.expiration = "2024-03-15"
                    option2.option_type = Mock()
                    option2.option_type.value = "C"
                    
                    position = Position(
                        symbol="SPY",
                        expiration_date=datetime(2024, 3, 15),
                        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                        strike_price=500.0,
                        entry_date=date,
                        entry_price=2.50,
                        spread_options=[option1, option2]
                    )
                    
                    add_position(position)
        
        strategy = MockStrategyWithCustomVolume()
        data = strategy.data
        
        # Create BacktestEngine with custom minimum volume
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=15, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 0  # Position should be rejected
        assert engine.volume_stats.options_checked == 3  # Three options checked (3 days)
        assert engine.volume_stats.positions_rejected_volume == 3  # Three rejections (3 days)
    
    def test_volume_statistics_reporting(self):
        """Test that volume statistics are properly tracked and reported."""
        strategy = MockStrategy()
        data = strategy.data
        
        # Create BacktestEngine with volume validation enabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        
        # Check that statistics are properly tracked
        assert engine.volume_stats.options_checked == 2
        assert engine.volume_stats.positions_rejected_volume == 0
        
        # Check that summary statistics are calculated correctly
        summary = engine.volume_stats.get_summary()
        assert summary['positions_rejected_volume'] == 0
        assert summary['options_checked'] == 2
        assert summary['rejection_rate'] == 0.0  # 0/2 * 100
    
    def test_backtest_engine_with_position_without_spread_options(self):
        """Test BacktestEngine handles positions without spread_options gracefully."""
        
        class MockStrategyWithoutSpreadOptions(MockStrategy):
            def on_new_date(self, date, positions, add_position, remove_position):
                """Mock strategy that creates positions without spread_options."""
                if len(positions) == 0:
                    # Create a position without spread_options
                    position = Position(
                        symbol="SPY",
                        expiration_date=datetime(2024, 3, 15),
                        strategy_type=StrategyType.LONG_STOCK,
                        strike_price=500.0,
                        entry_date=date,
                        entry_price=100.0,
                        spread_options=None  # No spread options
                    )
                    
                    add_position(position)
        
        strategy = MockStrategyWithoutSpreadOptions()
        data = strategy.data
        
        # Create BacktestEngine with volume validation enabled
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
        )
        
        # Run backtest
        success = engine.run()
        
        assert success is True
        assert len(engine.positions) == 1  # Position should be added
        assert engine.volume_stats.options_checked == 0  # No options checked
        assert engine.volume_stats.positions_rejected_volume == 0  # No rejections


if __name__ == "__main__":
    pytest.main([__file__]) 