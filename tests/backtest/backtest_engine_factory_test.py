"""
Tests for BacktestEngine.from_config() factory method.

This tests Phase 1 of the public API refactoring.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine.models.config import BacktestConfig
from algo_trading_engine.models.config import VolumeConfig
from tests.conftest import universe_from_frame
from algo_trading_engine.vo.universe import default_ohlcv_schema, identity_post_load


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [100, 101, 102],
            "Open": [99, 100, 101],
            "High": [101, 102, 103],
            "Low": [98, 99, 100],
            "Volume": [1000000, 1100000, 1200000],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )


def _schema_errors_from_extra_columns() -> SchemaErrors:
    frame = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000_000.0],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
            "Capital Gains": [0.0],
        },
        index=pd.date_range("2024-01-01", periods=1, freq="D"),
    )
    with pytest.raises(SchemaErrors) as exc_info:
        default_ohlcv_schema().validate(frame)
    return exc_info.value


def _mock_strategy():
    loaded = universe_from_frame("SPY", _sample_frame())
    loaded.load = Mock()
    mock_strategy = Mock()
    mock_strategy.universe = loaded
    mock_strategy.set_treasury_rates = Mock()
    mock_strategy.indicators = []
    mock_strategy.warm_up_period = 0
    mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
    mock_strategy.on_universe_load_complete = Mock(return_value=identity_post_load)
    return mock_strategy


class TestBacktestEngineFactory:
    """Test BacktestEngine.from_config() factory method."""

    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_with_strategy_name(self, mock_data_retriever, mock_options_handler, mock_create_strategy):
        """Test factory method with strategy name string."""
        mock_strategy = _mock_strategy()
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_retriever_instance.fetch_data_for_period.return_value = _sample_frame()
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            api_key="test_key",
        )
        engine = BacktestEngine.from_config(config)
        mock_data_retriever.assert_called_once_with(use_cache=True)
        mock_strategy.universe.load.assert_called_once()
        assert mock_strategy.universe.load.call_args[0][1] == config.bar_interval
        mock_options_handler.assert_called_once_with(api_key="test_key", use_free_tier=False)
        mock_create_strategy.assert_called_once()
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs["strategy_name"] == "credit_spread"
        mock_strategy.set_treasury_rates.assert_called_once()
        assert engine.initial_capital == 100000
        assert engine.start_date == datetime(2024, 1, 1)
        assert engine.end_date == datetime(2024, 1, 31)
        assert engine.strategy == mock_strategy

    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_with_strategy_instance(self, mock_data_retriever, _mock_options_handler):
        """Test factory method with Strategy instance."""
        mock_strategy = _mock_strategy()
        mock_strategy.options_handler = None
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_retriever_instance.fetch_data_for_period.return_value = _sample_frame()
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type=mock_strategy,
            api_key="test_key",
        )
        engine = BacktestEngine.from_config(config)
        assert engine.strategy == mock_strategy
        mock_strategy.set_treasury_rates.assert_called_once()

    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_loads_universe_with_config_bar_interval(self, mock_data_retriever, _mock_options_handler):
        """Universe load uses the configured bar interval, not a hardcoded daily interval."""
        mock_strategy = _mock_strategy()
        mock_strategy.options_handler = None
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_retriever_instance.fetch_data_for_period.return_value = _sample_frame()
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type=mock_strategy,
            api_key="test_key",
            bar_interval=BarTimeInterval.HOUR,
        )
        BacktestEngine.from_config(config)
        load_args = mock_strategy.universe.load.call_args[0]
        assert load_args[1] == BarTimeInterval.HOUR

    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_data_fetch_failure(self, mock_data_retriever, mock_options_handler, mock_create_strategy):
        """Test factory method handles data fetch failure (after strategy is constructed)."""
        mock_strategy = _mock_strategy()
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_strategy.universe.load.side_effect = ValueError("Failed to fetch data for SPY")
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_data_retriever.return_value = mock_retriever_instance
        mock_options_handler.return_value = Mock()
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
        )
        with pytest.raises(ValueError, match="Failed to fetch data"):
            BacktestEngine.from_config(config)

    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_propagates_schema_errors(self, mock_data_retriever, _mock_options_handler):
        """Callers such as CLI main and custom_strategy catch SchemaErrors from from_config."""
        mock_strategy = _mock_strategy()
        mock_strategy.options_handler = None
        mock_strategy.universe.load.side_effect = _schema_errors_from_extra_columns()
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type=mock_strategy,
            api_key="test_key",
        )
        with pytest.raises(SchemaErrors):
            BacktestEngine.from_config(config)

    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_propagates_schema_error(self, mock_data_retriever, _mock_options_handler):
        """Callers such as CLI main and custom_strategy catch SchemaError from from_config."""
        mock_strategy = _mock_strategy()
        mock_strategy.options_handler = None
        mock_strategy.universe.load.side_effect = SchemaError(None, None, "missing Close")
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_data_retriever.return_value = mock_retriever_instance
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type=mock_strategy,
            api_key="test_key",
        )
        with pytest.raises(SchemaError, match="missing Close"):
            BacktestEngine.from_config(config)

    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.DataRetriever")
    def test_from_config_with_all_options(self, mock_data_retriever, _mock_options_handler, mock_create_strategy):
        """Test factory method with all optional parameters."""
        mock_strategy = _mock_strategy()
        mock_create_strategy.return_value = mock_strategy
        mock_retriever_instance = Mock()
        mock_retriever_instance.load_treasury_rates.return_value = None
        mock_retriever_instance.fetch_data_for_period.return_value = _sample_frame()
        mock_data_retriever.return_value = mock_retriever_instance
        volume_config = VolumeConfig(min_volume=20)
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            max_position_size=0.5,
            volume_config=volume_config,
            enable_progress_tracking=False,
            quiet_mode=False,
            api_key="test_key",
            use_free_tier=True,
            lstm_start_date_offset=90,
            stop_loss=0.6,
            profit_target=0.4,
        )
        engine = BacktestEngine.from_config(config)
        assert engine.max_position_size == 0.5
        assert engine.volume_config.min_volume == 20
        assert engine.enable_progress_tracking is False
        assert engine.quiet_mode is False
        call_kwargs = mock_create_strategy.call_args[1]
        assert call_kwargs["stop_loss"] == 0.6
        assert call_kwargs["profit_target"] == 0.4
