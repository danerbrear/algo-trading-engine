"""Unit tests for shared EngineConfig fields."""
from datetime import datetime

import pytest

from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine.models.config import BacktestConfig, EngineConfig, PaperTradingConfig


def _backtest_config(**overrides) -> BacktestConfig:
    values = {
        "initial_capital": 100000,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 1, 31),
        "symbol": "SPY",
        "strategy_type": "credit_spread",
    }
    values.update(overrides)
    return BacktestConfig(**values)


def test_backtest_and_paper_configs_are_engine_configs():
    backtest = _backtest_config()
    paper = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")

    assert isinstance(backtest, EngineConfig)
    assert isinstance(paper, EngineConfig)


def test_shared_fields_have_the_same_defaults_on_both_configs():
    backtest = _backtest_config()
    paper = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")

    assert backtest.bar_interval == paper.bar_interval == BarTimeInterval.DAY
    assert backtest.max_position_size is paper.max_position_size is None
    assert backtest.api_key is paper.api_key is None
    assert backtest.use_free_tier is paper.use_free_tier is False
    assert backtest.stop_loss is paper.stop_loss is None
    assert backtest.profit_target is paper.profit_target is None
    assert backtest.observer is paper.observer is None
    assert backtest.use_cache is paper.use_cache is True


def test_max_position_size_validation_is_shared():
    with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
        _backtest_config(max_position_size=1.5)

    with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
        PaperTradingConfig(symbol="SPY", strategy_type="credit_spread", max_position_size=1.5)


def test_use_cache_can_be_disabled_on_both_configs():
    backtest = _backtest_config(use_cache=False)
    paper = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread", use_cache=False)

    assert backtest.use_cache is False
    assert paper.use_cache is False
