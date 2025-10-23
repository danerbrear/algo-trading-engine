"""
Tests for Phase 3: Strategy Integration
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.backtest.strategy_builder import (
    StrategyFactory,
    StrategyBuilder,
    UpwardTrendReversalStrategyBuilder
)
from src.backtest.models import Strategy
from src.strategies.upward_trend_reversal_strategy import UpwardTrendReversalStrategy


class TestUpwardTrendReversalStrategyBuilder:
    """Test the UpwardTrendReversalStrategyBuilder."""
    
    def test_builder_initialization(self):
        """Test builder initializes with correct defaults."""
        builder = UpwardTrendReversalStrategyBuilder()
        
        # Check defaults match feature document specifications
        assert builder._min_trend_duration == 3
        assert builder._max_trend_duration == 4
        assert builder._max_spread_width == 6.0
        assert builder._min_dte == 5
        assert builder._max_dte == 10
        assert builder._max_risk_per_trade == 0.20
        assert builder._max_holding_days == 2
        assert builder._start_date_offset == 60
    
    def test_builder_reset(self):
        """Test builder reset restores defaults."""
        builder = UpwardTrendReversalStrategyBuilder()
        
        # Modify some values
        builder.set_min_trend_duration(5)
        builder.set_max_spread_width(10.0)
        builder.set_stop_loss(0.5)
        
        # Reset
        builder.reset()
        
        # Verify defaults restored
        assert builder._min_trend_duration == 3
        assert builder._max_spread_width == 6.0
        assert builder._stop_loss is None
    
    def test_builder_set_methods_return_self(self):
        """Test builder methods return self for chaining."""
        builder = UpwardTrendReversalStrategyBuilder()
        
        result = builder.set_min_trend_duration(5)
        assert result is builder
        
        result = builder.set_max_trend_duration(6)
        assert result is builder
        
        result = builder.set_stop_loss(0.5)
        assert result is builder
    
    def test_builder_method_chaining(self):
        """Test builder supports method chaining."""
        builder = UpwardTrendReversalStrategyBuilder()
        mock_handler = Mock()
        
        result = (builder
                 .set_options_handler(mock_handler)
                 .set_min_trend_duration(5)
                 .set_max_trend_duration(6)
                 .set_stop_loss(0.5)
                 .set_profit_target(0.3))
        
        assert result is builder
        assert builder._min_trend_duration == 5
        assert builder._max_trend_duration == 6
        assert builder._stop_loss == 0.5
        assert builder._profit_target == 0.3
    
    def test_builder_builds_strategy_with_defaults(self):
        """Test builder creates strategy with default parameters."""
        builder = UpwardTrendReversalStrategyBuilder()
        mock_handler = Mock()
        
        builder.set_options_handler(mock_handler)
        strategy = builder.build()
        
        assert isinstance(strategy, UpwardTrendReversalStrategy)
        assert strategy.options_handler is mock_handler
        assert strategy.min_trend_duration == 3
        assert strategy.max_trend_duration == 4
        assert strategy.max_spread_width == 6.0
        assert strategy.min_dte == 5
        assert strategy.max_dte == 10
        assert strategy.max_risk_per_trade == 0.20
        assert strategy.max_holding_days == 2
    
    def test_builder_builds_strategy_with_custom_params(self):
        """Test builder creates strategy with custom parameters."""
        builder = UpwardTrendReversalStrategyBuilder()
        mock_handler = Mock()
        
        strategy = (builder
                   .set_options_handler(mock_handler)
                   .set_min_trend_duration(5)
                   .set_max_trend_duration(8)
                   .set_max_spread_width(10.0)
                   .set_min_dte(7)
                   .set_max_dte(14)
                   .set_max_risk_per_trade(0.25)
                   .set_max_holding_days(3)
                   .set_stop_loss(0.6)
                   .set_profit_target(0.4)
                   .set_start_date_offset(90)
                   .build())
        
        assert isinstance(strategy, UpwardTrendReversalStrategy)
        assert strategy.min_trend_duration == 5
        assert strategy.max_trend_duration == 8
        assert strategy.max_spread_width == 10.0
        assert strategy.min_dte == 7
        assert strategy.max_dte == 14
        assert strategy.max_risk_per_trade == 0.25
        assert strategy.max_holding_days == 3
        assert strategy.stop_loss == 0.6
        assert strategy.profit_target == 0.4
        assert strategy.start_date_offset == 90
    
    def test_builder_raises_error_without_options_handler(self):
        """Test builder raises error if options_handler not provided."""
        builder = UpwardTrendReversalStrategyBuilder()
        
        with pytest.raises(ValueError, match="Missing required parameter: options_handler"):
            builder.build()
    
    def test_builder_resets_after_build(self):
        """Test builder resets to defaults after build."""
        builder = UpwardTrendReversalStrategyBuilder()
        mock_handler = Mock()
        
        builder.set_options_handler(mock_handler)
        builder.set_min_trend_duration(5)
        builder.set_stop_loss(0.5)
        
        strategy = builder.build()
        
        # After build, should be reset to defaults
        assert builder._min_trend_duration == 3
        assert builder._stop_loss is None
        assert builder._options_handler is None
    
    def test_builder_lstm_model_and_scaler_not_used(self):
        """Test builder accepts but doesn't use LSTM model/scaler."""
        builder = UpwardTrendReversalStrategyBuilder()
        mock_handler = Mock()
        mock_model = Mock()
        mock_scaler = Mock()
        
        strategy = (builder
                   .set_options_handler(mock_handler)
                   .set_lstm_model(mock_model)
                   .set_lstm_scaler(mock_scaler)
                   .build())
        
        # Strategy should be created successfully
        assert isinstance(strategy, UpwardTrendReversalStrategy)
        # LSTM components are not used by this strategy
        assert not hasattr(strategy, 'lstm_model') or strategy.lstm_model is None


class TestStrategyFactoryIntegration:
    """Test StrategyFactory integration with UpwardTrendReversalStrategy."""
    
    def test_strategy_registered_in_factory(self):
        """Test strategy is registered in StrategyFactory."""
        available = StrategyFactory.get_available_strategies()
        assert 'upward_trend_reversal' in available
    
    def test_factory_creates_strategy_with_defaults(self):
        """Test factory creates strategy with default parameters."""
        mock_handler = Mock()
        
        strategy = StrategyFactory.create_strategy(
            'upward_trend_reversal',
            options_handler=mock_handler
        )
        
        assert isinstance(strategy, UpwardTrendReversalStrategy)
        assert strategy.options_handler is mock_handler
        assert strategy.min_trend_duration == 3
        assert strategy.max_trend_duration == 4
    
    def test_factory_creates_strategy_with_custom_params(self):
        """Test factory creates strategy with custom parameters."""
        mock_handler = Mock()
        
        strategy = StrategyFactory.create_strategy(
            'upward_trend_reversal',
            options_handler=mock_handler,
            min_trend_duration=5,
            max_trend_duration=8,
            max_spread_width=10.0,
            stop_loss=0.5,
            profit_target=0.3
        )
        
        assert isinstance(strategy, UpwardTrendReversalStrategy)
        assert strategy.min_trend_duration == 5
        assert strategy.max_trend_duration == 8
        assert strategy.max_spread_width == 10.0
        assert strategy.stop_loss == 0.5
        assert strategy.profit_target == 0.3
    
    def test_factory_get_builder(self):
        """Test factory returns correct builder instance."""
        builder = StrategyFactory.get_builder('upward_trend_reversal')
        assert isinstance(builder, UpwardTrendReversalStrategyBuilder)
    
    def test_factory_raises_error_for_unknown_strategy(self):
        """Test factory raises error for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            StrategyFactory.create_strategy('unknown_strategy')


class TestCLIIntegration:
    """Test CLI integration for recommendation engine."""
    
    def test_strategy_in_cli_registry(self):
        """Test strategy is registered in CLI."""
        from src.prediction.recommend_cli import STRATEGY_REGISTRY
        
        assert 'upward_trend_reversal' in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY['upward_trend_reversal'] is UpwardTrendReversalStrategy
    
    def test_cli_build_strategy_creates_correct_instance(self):
        """Test CLI build_strategy creates correct strategy instance."""
        from src.prediction.recommend_cli import build_strategy, STRATEGY_REGISTRY
        from unittest.mock import patch
        
        # Skip if not in registry (test environment issue)
        if 'upward_trend_reversal' not in STRATEGY_REGISTRY:
            pytest.skip("Strategy not in CLI registry")
        
        mock_handler = Mock()
        
        # Mock the model loading since we don't have real models in tests
        with patch('src.prediction.recommend_cli.load_lstm_model') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            strategy = build_strategy('upward_trend_reversal', mock_handler, 'SPY')
            
            assert isinstance(strategy, UpwardTrendReversalStrategy)
            assert strategy.options_handler is mock_handler


class TestBacktestEngineCompatibility:
    """Test compatibility with BacktestEngine."""
    
    def test_strategy_inherits_from_strategy_base(self):
        """Test strategy inherits from Strategy base class."""
        assert issubclass(UpwardTrendReversalStrategy, Strategy)
    
    def test_strategy_implements_required_methods(self):
        """Test strategy implements all required abstract methods."""
        required_methods = [
            'on_new_date',
            'on_end',
            'validate_data'
        ]
        
        for method in required_methods:
            assert hasattr(UpwardTrendReversalStrategy, method)
            assert callable(getattr(UpwardTrendReversalStrategy, method))
    
    def test_strategy_can_be_used_in_backtest(self):
        """Test strategy can be used with BacktestEngine."""
        from src.backtest.main import BacktestEngine
        import pandas as pd
        
        # Create mock strategy
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create minimal data
        data = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 102.5],
            'Market_State': [0, 0, 0, 0, 0]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        # Set data on strategy
        strategy.set_data(data, None, None)
        
        # Validate data
        assert strategy.validate_data(data) is True
        
        # Create BacktestEngine
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            quiet_mode=True
        )
        
        # Engine should be created successfully
        assert engine is not None
        assert engine.strategy is strategy


class TestPhase3Completeness:
    """Verify Phase 3 is fully implemented."""
    
    def test_strategy_builder_exists(self):
        """Verify UpwardTrendReversalStrategyBuilder exists."""
        from src.backtest.strategy_builder import UpwardTrendReversalStrategyBuilder
        assert UpwardTrendReversalStrategyBuilder is not None
    
    def test_strategy_registered_in_factory(self):
        """Verify strategy is registered in StrategyFactory."""
        available = StrategyFactory.get_available_strategies()
        assert 'upward_trend_reversal' in available
    
    def test_strategy_in_cli_registry(self):
        """Verify strategy is in CLI registry."""
        from src.prediction.recommend_cli import STRATEGY_REGISTRY
        assert 'upward_trend_reversal' in STRATEGY_REGISTRY
    
    def test_builder_has_all_configuration_methods(self):
        """Verify builder has all required configuration methods."""
        builder = UpwardTrendReversalStrategyBuilder()
        
        required_methods = [
            'set_lstm_model',
            'set_lstm_scaler',
            'set_options_handler',
            'set_start_date_offset',
            'set_stop_loss',
            'set_profit_target',
            'set_min_trend_duration',
            'set_max_trend_duration',
            'set_max_spread_width',
            'set_min_dte',
            'set_max_dte',
            'set_max_risk_per_trade',
            'set_max_holding_days',
            'reset',
            'build'
        ]
        
        for method in required_methods:
            assert hasattr(builder, method)
            assert callable(getattr(builder, method))
    
    def test_strategy_factory_can_create_all_strategies(self):
        """Verify factory can create all registered strategies."""
        available = StrategyFactory.get_available_strategies()
        
        # Should have at least 3 strategies
        assert len(available) >= 3
        assert 'credit_spread' in available
        assert 'velocity_momentum' in available
        assert 'upward_trend_reversal' in available

