"""
Algo Trading Engine - Options trading backtesting and paper trading framework.

This package provides a clean public API for building and testing trading strategies.

Public API exports are loaded lazily (PEP 562) so importing sub-packages such as
``algo_trading_engine.dto`` or ``algo_trading_engine.enums`` does not pull in
backtest engines, data retrievers, or ML dependencies.

Public API:
-----------
Engines:
    - BacktestEngine: Backtest trading strategies on historical data
    - PaperTradingEngine: Run trading strategies in paper trading mode

Configuration:
    - BacktestConfig: Configuration for backtesting
    - PaperTradingConfig: Configuration for paper trading
    - VolumeConfig: Volume validation configuration

Strategy Base:
    - Strategy: Abstract base class for custom strategies

Metrics:
    - PerformanceMetrics: Performance statistics from backtesting
    - PositionStats: Statistics for individual positions

Helpers:
    - OptionsRetrieverHelper: Static utility methods for filtering, finding, and calculating options data
    - DataRetriever: Market and treasury data fetching for strategies and engines
    - OptionsHandler: Options chain and bar retrieval (Polygon / cache)

Sub-packages:
    - dto: Data Transfer Objects for API communication
    - vo: Value Objects and runtime types
    - enums: Public enums
    - indicators: Technical indicators (Indicator, ATRIndicator, etc.)
    - logging: Logger configuration for backtest and paper trading
    - gui: GUI run observers (JsonLinesRunObserver, observer_from_env)

Example Usage:
--------------
    from algo_trading_engine import BacktestEngine, BacktestConfig
    from algo_trading_engine.enums import StrategyType
    from algo_trading_engine.vo import Position
    from datetime import datetime

    config = BacktestConfig(
        initial_capital=10000,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 1),
        symbol="SPY",
        strategy_type="credit_spread"
    )

    engine = BacktestEngine.from_config(config)
    success = engine.run()

    if success:
        metrics = engine.get_performance_metrics()
        print(f"Total Return: {metrics.total_return_pct:.2f}%")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from algo_trading_engine.backtest.main import BacktestEngine
    from algo_trading_engine.data_retriever import DataRetriever
    from algo_trading_engine.options_handler import OptionsHandler
    from algo_trading_engine.options_helpers import OptionsRetrieverHelper
    from algo_trading_engine.trade import PaperTradingEngine
    from algo_trading_engine.strategy import Strategy
    from algo_trading_engine.models.config import (
        BacktestConfig,
        PaperTradingConfig,
        VolumeConfig,
        VolumeStats,
    )
    from algo_trading_engine.models.metrics import PerformanceMetrics, PositionStats

    from . import database, dto, enums, gui, indicators, logging, plotting, vo

__all__ = [
    # Engines
    "BacktestEngine",
    "PaperTradingEngine",
    # Configuration
    "BacktestConfig",
    "PaperTradingConfig",
    "VolumeConfig",
    "VolumeStats",
    # Strategy Base
    "Strategy",
    # Metrics
    "PerformanceMetrics",
    "PositionStats",
    # Helpers
    "OptionsRetrieverHelper",
    "DataRetriever",
    "OptionsHandler",
    # Sub-packages (for strategy development)
    "dto",
    "vo",
    "enums",
    "indicators",
    "plotting",
    "database",
    "logging",
    "gui",
]

_LAZY_SUBMODULES = frozenset({"dto", "vo", "enums", "indicators", "plotting", "database", "logging", "gui"})

# module_path, attribute_name
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestEngine": ("algo_trading_engine.backtest.main", "BacktestEngine"),
    "PaperTradingEngine": ("algo_trading_engine.trade", "PaperTradingEngine"),
    "Strategy": ("algo_trading_engine.strategy", "Strategy"),
    "BacktestConfig": ("algo_trading_engine.models.config", "BacktestConfig"),
    "PaperTradingConfig": ("algo_trading_engine.models.config", "PaperTradingConfig"),
    "VolumeConfig": ("algo_trading_engine.models.config", "VolumeConfig"),
    "VolumeStats": ("algo_trading_engine.models.config", "VolumeStats"),
    "PerformanceMetrics": ("algo_trading_engine.models.metrics", "PerformanceMetrics"),
    "PositionStats": ("algo_trading_engine.models.metrics", "PositionStats"),
    "OptionsRetrieverHelper": (
        "algo_trading_engine.options_helpers",
        "OptionsRetrieverHelper",
    ),
    "DataRetriever": ("algo_trading_engine.data_retriever", "DataRetriever"),
    "OptionsHandler": ("algo_trading_engine.options_handler", "OptionsHandler"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"algo_trading_engine.{name}")
        globals()[name] = module
        return module

    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
