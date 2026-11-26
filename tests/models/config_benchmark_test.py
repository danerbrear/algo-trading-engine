"""
Unit tests for BacktestConfig benchmark_ticker support.
"""

from tests.conftest import universe_from_frame

import pytest
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from algo_trading_engine.models.config import BacktestConfig
from algo_trading_engine.backtest.main import BacktestEngine


class TestBacktestConfigBenchmarkTicker:
    """Test cases for BacktestConfig benchmark_ticker parameter."""

    def test_default_benchmark_ticker_is_none(self):
        """Test that benchmark_ticker defaults to None (uses trading symbol)."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
        )
        assert config.benchmark_ticker is None

    def test_benchmark_ticker_can_be_set(self):
        """Test that benchmark_ticker can be explicitly set."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="SPY",
        )
        assert config.benchmark_ticker == "SPY"

    def test_benchmark_ticker_same_as_symbol(self):
        """Test that benchmark_ticker can be the same as symbol."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="SPY",
        )
        assert config.benchmark_ticker == "SPY"

    def test_benchmark_ticker_is_immutable(self):
        """Test that benchmark_ticker cannot be modified (frozen dataclass)."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="SPY",
        )
        with pytest.raises(Exception):
            setattr(config, "benchmark_ticker", "QQQ")


def _make_price_data(start, periods, base_price=100.0):
    """Helper to create a mock price DataFrame."""
    return pd.DataFrame(
        {
            "Close": [base_price + i for i in range(periods)],
            "Open": [base_price + i - 1 for i in range(periods)],
            "High": [base_price + i + 1 for i in range(periods)],
            "Low": [base_price + i - 2 for i in range(periods)],
            "Volume": [1000000] * periods,
        },
        index=pd.date_range(start, periods=periods, freq="D"),
    )


def _mock_strategy_with_universe(symbol: str = "SPY", periods: int = 5) -> Mock:
    loaded = universe_from_frame(symbol, _make_price_data("2024-01-01", periods))
    loaded.load = Mock()
    mock_strategy = Mock()
    mock_strategy.universe = loaded
    mock_strategy.set_universe = Mock()
    mock_strategy.indicators = []
    mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
    mock_strategy.validate_universe = Mock(return_value=True)
    return mock_strategy


class TestBacktestEngineBenchmarkFromConfig:
    """Test BacktestEngine.from_config with benchmark_ticker."""

    @patch("algo_trading_engine.backtest.main.DataRetriever")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    def test_no_benchmark_ticker_uses_trading_data(
        self, mock_create_strategy, _mock_options_handler, mock_data_retriever
    ):
        """When benchmark_ticker is None, benchmark uses trading symbol data."""
        mock_strategy = _mock_strategy_with_universe()
        mock_create_strategy.return_value = mock_strategy
        mock_retriever = Mock()
        mock_retriever.treasury_rates = None
        mock_retriever.load_treasury_rates.return_value = None
        mock_retriever.fetch_data_for_period.return_value = _make_price_data("2024-01-01", 5)
        mock_data_retriever.return_value = mock_retriever
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
        )
        engine = BacktestEngine.from_config(config)
        assert engine.benchmark_data is None
        assert mock_data_retriever.call_count == 1

    @patch("algo_trading_engine.backtest.main.DataRetriever")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    def test_benchmark_ticker_same_as_symbol_skips_extra_fetch(
        self, mock_create_strategy, _mock_options_handler, mock_data_retriever
    ):
        """When benchmark_ticker equals symbol, no extra data fetch is made."""
        mock_strategy = _mock_strategy_with_universe("SPY")
        mock_create_strategy.return_value = mock_strategy
        mock_retriever = Mock()
        mock_retriever.treasury_rates = None
        mock_retriever.load_treasury_rates.return_value = None
        mock_retriever.fetch_data_for_period.return_value = _make_price_data("2024-01-01", 5)
        mock_data_retriever.return_value = mock_retriever
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="SPY",
        )
        engine = BacktestEngine.from_config(config)
        assert engine.benchmark_data is None
        assert mock_data_retriever.call_count == 1
        mock_retriever.fetch_data_for_period.assert_not_called()

    @patch("algo_trading_engine.backtest.main.DataRetriever")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    def test_different_benchmark_ticker_fetches_separate_data(
        self, mock_create_strategy, _mock_options_handler, mock_data_retriever
    ):
        """When benchmark_ticker differs from the strategy universe, fetch that ticker separately."""
        mock_strategy = _mock_strategy_with_universe("DIA")
        mock_create_strategy.return_value = mock_strategy
        benchmark_data = _make_price_data("2024-01-01", 5, base_price=400.0)
        mock_retriever = Mock()
        mock_retriever.treasury_rates = None
        mock_retriever.load_treasury_rates.return_value = None
        mock_retriever.fetch_data_for_period.return_value = benchmark_data
        mock_data_retriever.return_value = mock_retriever
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="SPY",
        )
        engine = BacktestEngine.from_config(config)
        assert engine.benchmark_data is not None
        assert len(engine.benchmark_data) == 5
        assert engine.benchmark_data.loc[engine.benchmark_data.index[0], "Close"] == 400.0
        mock_retriever.fetch_data_for_period.assert_called_once()
        assert mock_retriever.fetch_data_for_period.call_args[0][0] == "SPY"

    @patch("algo_trading_engine.backtest.main.DataRetriever")
    @patch("algo_trading_engine.backtest.main.OptionsHandler")
    @patch("algo_trading_engine.backtest.main.create_strategy_from_args")
    def test_benchmark_fetch_failure_falls_back(self, mock_create_strategy, _mock_options_handler, mock_data_retriever):
        """When benchmark data fetch fails, engine falls back to trading data."""
        mock_strategy = _mock_strategy_with_universe("DIA")
        mock_create_strategy.return_value = mock_strategy
        mock_retriever = Mock()
        mock_retriever.treasury_rates = None
        mock_retriever.load_treasury_rates.return_value = None
        mock_retriever.fetch_data_for_period.return_value = None
        mock_data_retriever.return_value = mock_retriever
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_type="credit_spread",
            benchmark_ticker="INVALID",
        )
        engine = BacktestEngine.from_config(config)
        assert engine.benchmark_data is None


class TestBacktestEngineBenchmarkRun:
    """Test that benchmark prices come from benchmark_data during run."""

    def test_benchmark_uses_separate_data_for_prices(self):
        """When benchmark_data is provided, benchmark prices come from it."""
        mock_strategy = Mock()
        mock_strategy.validate_data = Mock(return_value=True)
        mock_strategy.on_new_date = Mock()
        mock_strategy.on_end = Mock()
        mock_strategy.indicators = []
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_strategy.validate_universe = Mock(return_value=True)
        trading_data = _make_price_data("2024-01-01", 5, base_price=200.0)
        mock_strategy.universe = universe_from_frame("SPY", trading_data)
        benchmark_data = _make_price_data("2024-01-01", 5, base_price=400.0)
        engine = BacktestEngine(
            strategy=mock_strategy,
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            enable_progress_tracking=False,
            quiet_mode=True,
            benchmark_data=benchmark_data,
        )
        engine.run()
        assert engine.benchmark.start_price == 400.0
        assert engine.benchmark.end_price == 404.0

    def test_benchmark_falls_back_to_trading_data_when_no_benchmark_data(self):
        """When benchmark_data is None, benchmark uses trading data."""
        mock_strategy = Mock()
        mock_strategy.validate_data = Mock(return_value=True)
        mock_strategy.on_new_date = Mock()
        mock_strategy.on_end = Mock()
        mock_strategy.indicators = []
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_strategy.validate_universe = Mock(return_value=True)
        trading_data = _make_price_data("2024-01-01", 5, base_price=200.0)
        mock_strategy.universe = universe_from_frame("SPY", trading_data)
        engine = BacktestEngine(
            strategy=mock_strategy,
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            enable_progress_tracking=False,
            quiet_mode=True,
            benchmark_data=cast(pd.DataFrame, None),
        )
        engine.run()
        assert engine.benchmark.start_price == 200.0
        assert engine.benchmark.end_price == 204.0


class TestBacktestCLIBenchmarkArgument:
    """Test CLI --benchmark argument parsing."""

    def test_benchmark_arg_default_is_none(self):
        """Test that --benchmark defaults to None."""
        import sys
        from algo_trading_engine.backtest.main import parse_arguments

        with patch.object(sys, "argv", ["prog"]):
            args = parse_arguments()
            assert args.benchmark is None

    def test_benchmark_arg_can_be_specified(self):
        """Test that --benchmark can be set via CLI."""
        import sys
        from algo_trading_engine.backtest.main import parse_arguments

        with patch.object(sys, "argv", ["prog", "--benchmark", "QQQ"]):
            args = parse_arguments()
            assert args.benchmark == "QQQ"
