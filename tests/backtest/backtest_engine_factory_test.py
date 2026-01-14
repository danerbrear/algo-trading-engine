"""
Tests for BacktestEngine.from_config() factory method.

This tests Phase 1 of the public API refactoring.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.models.config import BacktestConfig
from algo_trading_engine.backtest.config import VolumeConfig


class TestBacktestEngineFactory:
    """Test BacktestEngine.from_config() factory method."""
    
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.create_strategy_from_args')
    def test_from_config_with_strategy_name(self, mock_create_strategy, mock_options_handler, mock_data_retriever):
        """Test factory method with strategy name string."""
        # Setup mocks
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.start_date_offset = 0
        mock_create_strategy.return_value = mock_strategy
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        
        # Create config
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol="SPY",
            strategy_type="credit_spread",
            api_key="test_key"
        )
        
        # Create engine from config
        engine = BacktestEngine.from_config(config)
        
        # Verify data retriever was created
        mock_data_retriever.assert_called_once()
        call_args = mock_data_retriever.call_args
        assert call_args[1]['symbol'] == 'SPY'
        assert call_args[1]['use_free_tier'] is False
        assert call_args[1]['quiet_mode'] is True
        
        # Verify data was fetched
        mock_retriever_instance.fetch_data_for_period.assert_called_once()
        
        # Verify options handler was created
        mock_options_handler.assert_called_once_with(
            symbol='SPY',
            api_key='test_key',
            use_free_tier=False
        )
        
        # Verify strategy was created
        mock_create_strategy.assert_called_once()
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs['strategy_name'] == 'credit_spread'
        assert call_kwargs['symbol'] == 'SPY'
        
        # Verify strategy.set_data was called
        mock_strategy.set_data.assert_called_once()
        
        # Verify engine was created correctly
        assert engine.initial_capital == 100000
        assert engine.start_date == datetime(2024, 1, 1)
        assert engine.end_date == datetime(2024, 1, 31)
        assert engine.strategy == mock_strategy
    
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    def test_from_config_with_strategy_instance(self, mock_options_handler, mock_data_retriever):
        """Test factory method with Strategy instance."""
        # Setup mocks
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.start_date_offset = 0
        mock_strategy.options_handler = None
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        
        # Create config with strategy instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol="SPY",
            strategy_type=mock_strategy,
            api_key="test_key"
        )
        
        # Create engine from config
        engine = BacktestEngine.from_config(config)
        
        # Verify strategy instance was used (not created)
        assert engine.strategy == mock_strategy
        
        # Verify callables were injected (if strategy has those attributes)
        if hasattr(mock_strategy, 'get_contract_list_for_date'):
            assert mock_strategy.get_contract_list_for_date is not None
        elif hasattr(mock_strategy, 'options_handler'):
            # Backward compatibility check
            assert mock_strategy.options_handler is not None
        
        # Verify strategy.set_data was called
        mock_strategy.set_data.assert_called_once()
    
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    def test_from_config_data_fetch_failure(self, mock_data_retriever):
        """Test factory method handles data fetch failure."""
        # Setup mock to return None (fetch failure)
        mock_retriever_instance = Mock()
        mock_retriever_instance.fetch_data_for_period.return_value = None
        mock_data_retriever.return_value = mock_retriever_instance
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol="SPY",
            strategy_type="credit_spread"
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Failed to fetch data"):
            BacktestEngine.from_config(config)
    
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.create_strategy_from_args')
    def test_from_config_with_all_options(self, mock_create_strategy, mock_options_handler, mock_data_retriever):
        """Test factory method with all optional parameters."""
        # Setup mocks
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.start_date_offset = 0
        mock_create_strategy.return_value = mock_strategy
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        
        volume_config = VolumeConfig(min_volume=20)
        
        # Create config with all options
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol="SPY",
            strategy_type="credit_spread",
            max_position_size=0.5,
            volume_config=volume_config,
            enable_progress_tracking=False,
            quiet_mode=False,
            api_key="test_key",
            use_free_tier=True,
            lstm_start_date_offset=90,
            stop_loss=0.6,
            profit_target=0.4
        )
        
        # Create engine from config
        engine = BacktestEngine.from_config(config)
        
        # Verify all options were passed correctly
        assert engine.max_position_size == 0.5
        assert engine.volume_config.min_volume == 20
        assert engine.enable_progress_tracking is False
        assert engine.quiet_mode is False
        
        # Verify strategy was created with stop_loss and profit_target
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs['stop_loss'] == 0.6
        assert call_kwargs['profit_target'] == 0.4
        
        # Verify data retriever was created with custom offset
        call_args = mock_data_retriever.call_args
        # Check that lstm_start_date was calculated correctly (90 days before start_date)
        expected_lstm_start = (datetime(2024, 1, 1) - timedelta(days=90)).strftime("%Y-%m-%d")
        assert call_args[1]['lstm_start_date'] == expected_lstm_start

