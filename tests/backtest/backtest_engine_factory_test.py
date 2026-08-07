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

def _passthrough_credit_spread_ml_prep(data, _retriever, _symbol):
    """Avoid calendar/HMM I/O when unit-testing BacktestEngine.from_config."""
    return data

class TestBacktestEngineFactory:
    """Test BacktestEngine.from_config() factory method."""

    @patch('algo_trading_engine.common.ml_pipeline.prepare_credit_spread_backtest_data', side_effect=_passthrough_credit_spread_ml_prep)
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.create_strategy_from_args')
    def test_from_config_with_strategy_name(self, mock_create_strategy, mock_options_handler, mock_data_retriever, _mock_ml_prep):
        """Test factory method with strategy name string."""
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [100, 101, 102], 'Open': [99, 100, 101], 'High': [101, 102, 103], 'Low': [98, 99, 100], 'Volume': [1000000, 1100000, 1200000]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(initial_capital=100000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), symbol='SPY', strategy_type='credit_spread', api_key='test_key')
        engine = BacktestEngine.from_config(config)
        mock_data_retriever.assert_called_once()
        call_args = mock_data_retriever.call_args
        assert call_args[1]['symbol'] == 'SPY'
        assert 'lstm_start_date' in call_args[1]
        mock_retriever_instance.fetch_data_for_period.assert_called_once()
        mock_options_handler.assert_called_once_with(symbol='SPY', api_key='test_key', use_free_tier=False)
        mock_create_strategy.assert_called_once()
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs['strategy_name'] == 'credit_spread'
        assert call_kwargs['symbol'] == 'SPY'
        mock_strategy.set_data.assert_called_once()
        _mock_ml_prep.assert_called_once()
        assert engine.initial_capital == 100000
        assert engine.start_date == datetime(2024, 1, 1)
        assert engine.end_date == datetime(2024, 1, 31)
        assert engine.strategy == mock_strategy

    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    def test_from_config_with_strategy_instance(self, mock_data_retriever, _mock_options_handler):
        """Test factory method with Strategy instance."""
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_strategy.options_handler = None
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [100, 101, 102], 'Open': [99, 100, 101], 'High': [101, 102, 103], 'Low': [98, 99, 100], 'Volume': [1000000, 1100000, 1200000]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(initial_capital=100000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), symbol='SPY', strategy_type=mock_strategy, api_key='test_key')
        engine = BacktestEngine.from_config(config)
        assert engine.strategy == mock_strategy
        assert mock_strategy.symbol == 'SPY'
        if hasattr(mock_strategy, 'get_contract_list_for_date'):
            assert mock_strategy.get_contract_list_for_date is not None
        elif hasattr(mock_strategy, 'options_handler'):
            assert mock_strategy.options_handler is not None
        mock_strategy.set_data.assert_called_once()

    @patch('algo_trading_engine.common.ml_pipeline.prepare_credit_spread_backtest_data', side_effect=_passthrough_credit_spread_ml_prep)
    @patch('algo_trading_engine.backtest.main.create_strategy_from_args')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    def test_from_config_data_fetch_failure(self, mock_data_retriever, mock_options_handler, mock_create_strategy, _mock_ml_prep):
        """Test factory method handles data fetch failure (after strategy is constructed)."""
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_retriever_instance.fetch_data_for_period.return_value = None
        mock_retriever_instance.treasury_rates = None
        mock_data_retriever.return_value = mock_retriever_instance
        mock_options_handler.return_value = Mock()
        config = BacktestConfig(initial_capital=100000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), symbol='SPY', strategy_type='credit_spread')
        with pytest.raises(ValueError, match='Failed to fetch data'):
            BacktestEngine.from_config(config)

    @patch('algo_trading_engine.common.ml_pipeline.prepare_credit_spread_backtest_data', side_effect=_passthrough_credit_spread_ml_prep)
    @patch('algo_trading_engine.backtest.main.create_strategy_from_args')
    @patch('algo_trading_engine.backtest.main.OptionsHandler')
    @patch('algo_trading_engine.backtest.main.DataRetriever')
    def test_from_config_with_all_options(self, mock_data_retriever, _mock_options_handler, mock_create_strategy, _mock_ml_prep):
        """Test factory method with all optional parameters."""
        mock_strategy = Mock()
        mock_strategy.set_data = Mock()
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [100, 101, 102], 'Open': [99, 100, 101], 'High': [101, 102, 103], 'Low': [98, 99, 100], 'Volume': [1000000, 1100000, 1200000]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        volume_config = VolumeConfig(min_volume=20)
        config = BacktestConfig(initial_capital=100000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 31), symbol='SPY', strategy_type='credit_spread', max_position_size=0.5, volume_config=volume_config, enable_progress_tracking=False, quiet_mode=False, api_key='test_key', use_free_tier=True, lstm_start_date_offset=90, stop_loss=0.6, profit_target=0.4)
        engine = BacktestEngine.from_config(config)
        assert engine.max_position_size == 0.5
        assert engine.volume_config.min_volume == 20
        assert engine.enable_progress_tracking is False
        assert engine.quiet_mode is False
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs['stop_loss'] == 0.6
        assert call_kwargs['profit_target'] == 0.4
        call_args = mock_data_retriever.call_args
        expected_lstm_start = (datetime(2024, 1, 1) - timedelta(days=90)).strftime('%Y-%m-%d')
        assert call_args[1]['lstm_start_date'] == expected_lstm_start
