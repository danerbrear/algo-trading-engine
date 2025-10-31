"""
Tests for Phase 5: Strategy Configuration and Testing

This phase validates:
1. Strategy configuration options
2. Integration with backtest framework
3. Performance metrics and tracking
4. Comparison against benchmarks
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.strategies.upward_trend_reversal_strategy import UpwardTrendReversalStrategy
from src.backtest.main import BacktestEngine
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestStrategyConfiguration:
    """Test strategy configuration options."""
    
    def test_default_configuration(self):
        """Test strategy initializes with default parameters."""
        mock_handler = Mock()
        
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Verify defaults from feature document
        assert strategy.min_trend_duration == 3
        assert strategy.max_trend_duration == 4
        assert strategy.max_spread_width == 6.0
        assert strategy.min_dte == 5
        assert strategy.max_dte == 10
        assert strategy.max_risk_per_trade == 0.20
        assert strategy.max_holding_days == 2
        assert strategy.profit_target is None
        assert strategy.stop_loss is None
        assert strategy.start_date_offset == 60
    
    def test_custom_configuration(self):
        """Test strategy accepts custom configuration."""
        mock_handler = Mock()
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            min_trend_duration=5,
            max_trend_duration=8,
            max_spread_width=10.0,
            min_dte=7,
            max_dte=14,
            max_risk_per_trade=0.25,
            max_holding_days=3,
            profit_target=0.30,
            stop_loss=0.50,
            start_date_offset=90
        )
        
        assert strategy.min_trend_duration == 5
        assert strategy.max_trend_duration == 8
        assert strategy.max_spread_width == 10.0
        assert strategy.min_dte == 7
        assert strategy.max_dte == 14
        assert strategy.max_risk_per_trade == 0.25
        assert strategy.max_holding_days == 3
        assert strategy.profit_target == 0.30
        assert strategy.stop_loss == 0.50
        assert strategy.start_date_offset == 90
    
    def test_hmm_configuration(self):
        """Test HMM training configuration options."""
        mock_handler = Mock()
        mock_retriever = Mock()
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            data_retriever=mock_retriever,
            train_hmm=True,
            hmm_training_years=3,
            save_trained_hmm=True,
            hmm_model_dir='/custom/path'
        )
        
        assert strategy.train_hmm is True
        assert strategy.hmm_training_years == 3
        assert strategy.save_trained_hmm is True
        assert strategy.hmm_model_dir == '/custom/path'
        assert strategy.data_retriever == mock_retriever


class TestBacktestIntegration:
    """Test integration with backtest engine."""
    
    def test_strategy_works_with_backtest_engine(self):
        """Test strategy integrates with BacktestEngine."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Returns': np.random.randn(100) * 0.01,
            'Volatility': np.random.rand(100) * 0.02 + 0.01,
            'Volume_Change': np.random.randn(100) * 0.1,
            'Market_State': np.random.randint(0, 5, 100)
        }, index=dates)
        
        strategy.set_data(data, None, None)
        
        # Validate data
        assert strategy.validate_data(data) is True
        
        # Create backtest engine
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            quiet_mode=True
        )
        
        assert engine is not None
        assert engine.strategy == strategy
        assert engine.initial_capital == 10000
    
    def test_strategy_requires_options_handler(self):
        """Test that strategy requires options_handler."""
        with pytest.raises(TypeError):
            # Should fail without options_handler
            strategy = UpwardTrendReversalStrategy()
    
    def test_strategy_validates_data_columns(self):
        """Test that strategy validates required data columns."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Valid data
        valid_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Market_State': [0, 0, 1]
        })
        assert strategy.validate_data(valid_data) is True
        
        # Missing Close column
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Market_State': [0, 0, 1]
        })
        assert strategy.validate_data(invalid_data) is False
        
        # Missing Market_State column (CRITICAL - should fail)
        invalid_data_no_hmm = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        assert strategy.validate_data(invalid_data_no_hmm) is False
    
    def test_options_handler_has_get_contract_list_method(self):
        """Test that OptionsHandler has get_contract_list_for_date method (regression test)."""
        from src.common.options_handler import OptionsHandler
        
        # Verify the method exists
        assert hasattr(OptionsHandler, 'get_contract_list_for_date')
        assert callable(getattr(OptionsHandler, 'get_contract_list_for_date'))
        
        # Create a mock OptionsHandler and verify the strategy can use it
        mock_handler = Mock(spec=OptionsHandler)
        mock_handler.get_contract_list_for_date = Mock(return_value=[])
        
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Verify the strategy can call the method without AttributeError
        from datetime import datetime
        from src.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO
        
        try:
            strategy.options_handler.get_contract_list_for_date(
                date=datetime(2024, 1, 1),
                strike_range=StrikeRangeDTO(min_strike=450.0, max_strike=460.0),
                expiration_range=ExpirationRangeDTO(
                    min_days=5,
                    max_days=10,
                    current_date=datetime(2024, 1, 1).date()
                )
            )
            # If we get here, the method exists and is callable
            assert True
        except AttributeError as e:
            pytest.fail(f"OptionsHandler missing get_contract_list_for_date method: {e}")


class TestPositionManagement:
    """Test position management and sizing."""
    
    def test_position_sizing_calculation(self):
        """Test position sizing based on risk parameters."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            max_risk_per_trade=0.20
        )
        
        # Create mock position with known risk
        atm_put = Option(
            ticker='O:SPY240115P00450000',
            symbol='SPY',
            strike=450.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=5.0,
            volume=100
        )
        
        otm_put = Option(
            ticker='O:SPY240115P00445000',
            symbol='SPY',
            strike=445.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=3.0,
            volume=100
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 15),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.0,  # Net debit
            spread_options=[atm_put, otm_put]
        )
        
        # Test position sizing with $10,000 capital
        capital = 10000
        max_risk_pct = 0.20
        
        quantity = strategy._calculate_position_size(position, max_risk_pct, capital)
        
        # With $10k capital and 20% max risk = $2000 max risk
        # Position max risk = $2.00 * 100 = $200 per contract
        # Expected quantity = $2000 / $200 = 10 contracts
        assert quantity == 10
    
    def test_position_sizing_minimum_one_contract(self):
        """Test that position sizing returns at least 1 contract."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create position with very high risk
        atm_put = Option(
            ticker='O:SPY240115P00450000',
            symbol='SPY',
            strike=450.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=100.0,  # Very expensive
            volume=100
        )
        
        otm_put = Option(
            ticker='O:SPY240115P00445000',
            symbol='SPY',
            strike=445.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=95.0,
            volume=100
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 15),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=5.0,
            spread_options=[atm_put, otm_put]
        )
        
        # Even with insufficient capital, should return at least 1
        quantity = strategy._calculate_position_size(position, 0.20, 100)
        assert quantity >= 1
    
    def test_max_holding_period(self):
        """Test that positions respect max holding period."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            max_holding_days=2
        )
        
        assert strategy.max_holding_days == 2


class TestRiskManagement:
    """Test risk management features."""
    
    def test_stop_loss_configuration(self):
        """Test stop loss can be configured."""
        mock_handler = Mock()
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            stop_loss=0.50
        )
        
        assert strategy.stop_loss == 0.50
    
    def test_profit_target_configuration(self):
        """Test profit target can be configured."""
        mock_handler = Mock()
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            profit_target=0.30
        )
        
        assert strategy.profit_target == 0.30
    
    def test_max_risk_per_trade(self):
        """Test max risk per trade configuration."""
        mock_handler = Mock()
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            max_risk_per_trade=0.15
        )
        
        assert strategy.max_risk_per_trade == 0.15


class TestMarketRegimeFiltering:
    """Test market regime filtering logic."""
    
    def test_momentum_uptrend_detection(self):
        """Test detection of momentum uptrend regime."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create data with momentum uptrend (state 1)
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Market_State': [1, 1, 1]  # Momentum uptrend
        }, index=pd.date_range('2024-01-01', periods=3))
        
        strategy.set_data(data, None, None)
        
        # Should detect momentum uptrend
        date = datetime(2024, 1, 1)
        is_momentum = strategy._is_momentum_uptrend_regime(date)
        assert is_momentum is True
    
    def test_non_momentum_regime(self):
        """Test detection of non-momentum regimes."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create data with other regime
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Market_State': [0, 2, 3]  # Not momentum uptrend
        }, index=pd.date_range('2024-01-01', periods=3))
        
        strategy.set_data(data, None, None)
        
        # Should not detect momentum uptrend
        date = datetime(2024, 1, 1)
        is_momentum = strategy._is_momentum_uptrend_regime(date)
        assert is_momentum is False
    
    def test_missing_market_state_column(self):
        """Test handling of missing Market_State column."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        # Create data without Market_State
        data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        strategy.set_data(data, None, None)
        
        # Should gracefully handle missing column
        date = datetime(2024, 1, 1)
        is_momentum = strategy._is_momentum_uptrend_regime(date)
        assert is_momentum is False  # Default to False when column missing


class TestPerformanceMetrics:
    """Test performance metrics and tracking."""
    
    def test_strategy_tracks_detected_trends(self):
        """Test that strategy tracks detected trends."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        assert hasattr(strategy, 'detected_trends')
        assert isinstance(strategy.detected_trends, list)
    
    def test_backtest_engine_tracks_positions(self):
        """Test that backtest engine tracks closed positions."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(options_handler=mock_handler)
        
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Market_State': [0, 0, 0]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        strategy.set_data(data, None, None)
        
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            quiet_mode=True
        )
        
        # Backtest engine should have closed_positions list
        assert hasattr(engine, 'closed_positions')
        assert isinstance(engine.closed_positions, list)


class TestSpreadSelection:
    """Test put debit spread selection logic."""
    
    def test_spread_width_constraint(self):
        """Test that spread width is constrained."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            max_spread_width=6.0
        )
        
        assert strategy.max_spread_width == 6.0
    
    def test_dte_constraints(self):
        """Test DTE (days to expiration) constraints."""
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            min_dte=5,
            max_dte=10
        )
        
        assert strategy.min_dte == 5
        assert strategy.max_dte == 10


class TestPhase5Completeness:
    """Verify Phase 5 implementation is complete."""
    
    def test_all_configuration_options_available(self):
        """Test all configuration options from feature document are available."""
        mock_handler = Mock()
        
        # Should be able to configure all parameters
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            min_trend_duration=3,
            max_trend_duration=4,
            max_spread_width=6.0,
            min_dte=5,
            max_dte=10,
            max_risk_per_trade=0.20,
            max_holding_days=2,
            profit_target=0.30,
            stop_loss=0.50,
            start_date_offset=60,
            # HMM options
            data_retriever=None,
            train_hmm=False,
            hmm_training_years=2,
            save_trained_hmm=False,
            hmm_model_dir=None
        )
        
        assert strategy is not None
    
    def test_strategy_integrates_with_backtest_cli(self):
        """Test strategy is available in backtest CLI."""
        from src.backtest.strategy_builder import StrategyFactory
        
        available_strategies = StrategyFactory.get_available_strategies()
        assert 'upward_trend_reversal' in available_strategies
    
    def test_strategy_can_be_created_via_factory(self):
        """Test strategy can be created via StrategyFactory."""
        from src.backtest.strategy_builder import StrategyFactory
        
        mock_handler = Mock()
        
        strategy = StrategyFactory.create_strategy(
            'upward_trend_reversal',
            options_handler=mock_handler
        )
        
        assert isinstance(strategy, UpwardTrendReversalStrategy)
    
    def test_strategy_builder_has_all_setters(self):
        """Test strategy builder has setters for all configuration options."""
        from src.backtest.strategy_builder import UpwardTrendReversalStrategyBuilder
        
        builder = UpwardTrendReversalStrategyBuilder()
        
        required_setters = [
            'set_options_handler',
            'set_min_trend_duration',
            'set_max_trend_duration',
            'set_max_spread_width',
            'set_min_dte',
            'set_max_dte',
            'set_max_risk_per_trade',
            'set_max_holding_days',
            'set_profit_target',
            'set_stop_loss',
            'set_start_date_offset',
            # HMM setters
            'set_data_retriever',
            'set_train_hmm',
            'set_hmm_training_years',
            'set_save_trained_hmm',
            'set_hmm_model_dir'
        ]
        
        for setter in required_setters:
            assert hasattr(builder, setter), f"Missing setter: {setter}"
            assert callable(getattr(builder, setter))


class TestSuccessCriteria:
    """Test success criteria from feature document."""
    
    def test_debit_spread_support(self):
        """Test that backtest engine supports put debit spreads."""
        from src.backtest.models import StrategyType
        
        # Verify StrategyType enum has debit spreads
        assert hasattr(StrategyType, 'PUT_DEBIT_SPREAD')
        assert hasattr(StrategyType, 'CALL_DEBIT_SPREAD')
    
    def test_position_get_max_risk_for_debit_spreads(self):
        """Test Position.get_max_risk() works for debit spreads."""
        atm_put = Option(
            ticker='O:SPY240115P00450000',
            symbol='SPY',
            strike=450.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=5.0,
            volume=100
        )
        
        otm_put = Option(
            ticker='O:SPY240115P00445000',
            symbol='SPY',
            strike=445.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=3.0,
            volume=100
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 15),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.0,
            spread_options=[atm_put, otm_put]
        )
        
        max_risk = position.get_max_risk()
        
        # For debit spread, max risk = net debit paid = 2.0 * 100 = 200
        assert max_risk == 200.0
    
    def test_trend_detection_available(self):
        """Test trend detection functionality is available."""
        from src.common.trend_detector import TrendDetector
        
        assert hasattr(TrendDetector, 'detect_forward_trends')
        assert callable(TrendDetector.detect_forward_trends)
    
    def test_strategy_uses_correct_options_handler(self):
        """Test strategy uses src/common/options_handler.py."""
        from src.strategies.upward_trend_reversal_strategy import UpwardTrendReversalStrategy
        
        # Check imports in the strategy file
        import inspect
        source = inspect.getsource(UpwardTrendReversalStrategy)
        
        # Should import from src.common.options_handler
        assert 'from src.common.options_handler import OptionsHandler' in source
        
        # Should NOT import from src.model.options_handler
        assert 'from src.model.options_handler' not in source

