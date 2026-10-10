"""
Tests for the public API.

This module tests that:
1. The public API exports work correctly
2. BacktestEngine and PaperTradingEngine can be instantiated via from_config()
3. Config classes have all necessary parameters
"""

import pytest
from datetime import datetime, timedelta

# Test that we can import from the public API
def test_public_api_imports():
    """Test that all public API exports are available."""
    # This should work for external users
    from algo_trading_engine import (
        BacktestEngine,
        PaperTradingEngine,
        BacktestConfig,
        PaperTradingConfig,
        VolumeConfig,
        VolumeStats,
        Strategy,
        PerformanceMetrics,
        PositionStats,
        DataRetriever,
        OptionsHandler,
    )
    
    # Verify that all imports succeeded
    assert BacktestEngine is not None
    assert PaperTradingEngine is not None
    assert BacktestConfig is not None
    assert PaperTradingConfig is not None
    assert VolumeConfig is not None
    assert VolumeStats is not None
    assert Strategy is not None
    assert PerformanceMetrics is not None
    assert PositionStats is not None

    from algo_trading_engine import OptionsRetrieverHelper

    assert OptionsRetrieverHelper is not None
    assert DataRetriever is not None
    assert OptionsHandler is not None


def test_backtest_config_creation():
    """Test creating BacktestConfig with all parameters."""
    from algo_trading_engine import BacktestConfig
    
    # Test with minimal required parameters
    config = BacktestConfig(
        initial_capital=10000,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        symbol="SPY",
        strategy_type="credit_spread"
    )
    
    assert config.initial_capital == 10000
    assert config.start_date == datetime(2024, 1, 1)
    assert config.end_date == datetime(2024, 12, 31)
    assert config.symbol == "SPY"
    assert config.strategy_type == "credit_spread"
    assert config.volume_config is not None  # Should have default


def test_backtest_config_validation():
    """Test BacktestConfig validation rules."""
    from algo_trading_engine import BacktestConfig
    
    # Test that start_date must be before end_date
    with pytest.raises(ValueError, match="Start date must be before end date"):
        BacktestConfig(
            initial_capital=10000,
            start_date=datetime(2024, 12, 31),
            end_date=datetime(2024, 1, 1),  # Wrong order
            symbol="SPY",
            strategy_type="credit_spread"
        )
    
    # Test that initial_capital must be positive
    with pytest.raises(ValueError, match="Initial capital must be greater than 0"):
        BacktestConfig(
            initial_capital=-1000,  # Negative
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            symbol="SPY",
            strategy_type="credit_spread"
        )


def test_paper_trading_config_creation():
    """Test creating PaperTradingConfig with all parameters."""
    from algo_trading_engine import PaperTradingConfig
    
    # Test with minimal required parameters
    config = PaperTradingConfig(
        symbol="SPY",
        strategy_type="velocity_momentum"
    )
    
    assert config.symbol == "SPY"
    assert config.strategy_type == "velocity_momentum"


def test_paper_trading_config_validation():
    """Test PaperTradingConfig validation rules."""
    from algo_trading_engine import PaperTradingConfig
    
    # Test that max_position_size validation works
    with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
        PaperTradingConfig(
            symbol="SPY",
            strategy_type="velocity_momentum",
            max_position_size=1.5  # Invalid: > 1
        )


def test_volume_config_creation():
    """Test creating VolumeConfig."""
    from algo_trading_engine import VolumeConfig
    
    # Test default values
    config = VolumeConfig()
    assert config.min_volume == 10
    assert config.enable_volume_validation is True
    
    # Test custom values
    config = VolumeConfig(min_volume=20, enable_volume_validation=False)
    assert config.min_volume == 20
    assert config.enable_volume_validation is False


def test_volume_config_validation():
    """Test VolumeConfig validation rules."""
    from algo_trading_engine import VolumeConfig
    
    # Test that min_volume cannot be negative
    with pytest.raises(ValueError, match="Minimum volume cannot be negative"):
        VolumeConfig(min_volume=-5)
    
    # Test that min_volume cannot be zero
    with pytest.raises(ValueError, match="Minimum volume must be greater than 0"):
        VolumeConfig(min_volume=0)


def test_strategy_base_class():
    """Test that Strategy base class is abstract."""
    from algo_trading_engine.strategy import Strategy
    
    # Should not be able to instantiate Strategy directly
    with pytest.raises(TypeError):
        Strategy()  # pylint: disable=abstract-class-instantiated


def test_backtest_engine_class_exists():
    """Test that BacktestEngine class exists and has from_config method."""
    from algo_trading_engine import BacktestEngine
    
    # Verify class exists
    assert BacktestEngine is not None
    
    # Verify from_config method exists
    assert hasattr(BacktestEngine, 'from_config')
    assert callable(BacktestEngine.from_config)
    
    # Verify run method exists
    assert hasattr(BacktestEngine, 'run')
    
    # Verify get_performance_metrics method exists
    assert hasattr(BacktestEngine, 'get_performance_metrics')


def test_paper_trading_engine_class_exists():
    """Test that PaperTradingEngine class exists and has from_config method."""
    from algo_trading_engine import PaperTradingEngine
    
    # Verify class exists
    assert PaperTradingEngine is not None
    
    # Verify from_config method exists
    assert hasattr(PaperTradingEngine, 'from_config')
    assert callable(PaperTradingEngine.from_config)
    
    # Verify run method exists
    assert hasattr(PaperTradingEngine, 'run')


def test_backtest_config_cli_parameters():
    """Test that BacktestConfig has all CLI parameters."""
    from algo_trading_engine import BacktestConfig
    
    # Create config with all CLI-accessible parameters
    config = BacktestConfig(
        initial_capital=3000,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        symbol="SPY",
        strategy_type="velocity_momentum",
        max_position_size=0.40,
        api_key="test_key",
        use_free_tier=True,
        quiet_mode=False,
        lstm_start_date_offset=120,
        stop_loss=0.6,
        profit_target=0.5
    )
    
    # Verify all parameters are set
    assert config.initial_capital == 3000
    assert config.start_date == datetime(2024, 1, 1)
    assert config.end_date == datetime(2024, 12, 31)
    assert config.symbol == "SPY"
    assert config.strategy_type == "velocity_momentum"
    assert config.max_position_size == 0.40
    assert config.api_key == "test_key"
    assert config.use_free_tier is True
    assert config.quiet_mode is False
    assert config.lstm_start_date_offset == 120
    assert config.stop_loss == 0.6
    assert config.profit_target == 0.5


def test_paper_trading_config_cli_parameters():
    """Test that PaperTradingConfig has all CLI parameters."""
    from algo_trading_engine import PaperTradingConfig
    
    # Create config with all CLI-accessible parameters
    # Note: Capital is managed via config/strategies/capital_allocations.json
    config = PaperTradingConfig(
        symbol="SPY",
        strategy_type="credit_spread",
        max_position_size=0.40,
        api_key="test_key",
        use_free_tier=True,
        stop_loss=0.6,
        profit_target=0.5
    )
    
    # Verify all parameters are set
    assert config.symbol == "SPY"
    assert config.strategy_type == "credit_spread"
    assert config.max_position_size == 0.40
    assert config.api_key == "test_key"
    assert config.use_free_tier is True
    assert config.stop_loss == 0.6
    assert config.profit_target == 0.5


def test_performance_metrics_structure():
    """Test that PerformanceMetrics has expected attributes."""
    from dataclasses import fields as dataclass_fields
    from algo_trading_engine.models.metrics import PerformanceMetrics
    
    field_names = {field.name for field in dataclass_fields(PerformanceMetrics)}
    assert 'total_return' in field_names
    assert 'total_return_pct' in field_names
    assert 'sharpe_ratio' in field_names
    assert 'max_drawdown' in field_names
    assert 'win_rate' in field_names
    assert 'total_positions' in field_names
    assert 'closed_positions' in field_names
    assert 'strategy_stats' in field_names


def test_position_stats_structure():
    """Test that PositionStats has expected attributes."""
    from dataclasses import fields as dataclass_fields
    from algo_trading_engine.models.metrics import PositionStats
    
    field_names = {field.name for field in dataclass_fields(PositionStats)}
    assert 'strategy_type' in field_names
    assert 'entry_date' in field_names
    assert 'exit_date' in field_names
    assert 'entry_price' in field_names
    assert 'exit_price' in field_names
    assert 'return_dollars' in field_names
    assert 'return_percentage' in field_names
    assert 'days_held' in field_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

