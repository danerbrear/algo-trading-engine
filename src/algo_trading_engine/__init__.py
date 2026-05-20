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

Sub-packages:
    - dto: Data Transfer Objects for API communication
    - vo: Value Objects and runtime types
    - enums: Public enums
    - indicators: Technical indicators (Indicator, ATRIndicator, etc.)

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
from typing import Any

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
    # Sub-packages (for strategy development)
    "dto",
    "vo",
    "enums",
    "indicators",
    "database",
]

_LAZY_SUBMODULES = frozenset({"dto", "vo", "enums", "indicators", "database"})

# module_path, attribute_name
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestEngine": ("algo_trading_engine.backtest.main", "BacktestEngine"),
    "PaperTradingEngine": ("algo_trading_engine.core.engine", "PaperTradingEngine"),
    "Strategy": ("algo_trading_engine.core.strategy", "Strategy"),
    "BacktestConfig": ("algo_trading_engine.models.config", "BacktestConfig"),
    "PaperTradingConfig": ("algo_trading_engine.models.config", "PaperTradingConfig"),
    "VolumeConfig": ("algo_trading_engine.models.config", "VolumeConfig"),
    "VolumeStats": ("algo_trading_engine.models.config", "VolumeStats"),
    "PerformanceMetrics": ("algo_trading_engine.models.metrics", "PerformanceMetrics"),
    "PositionStats": ("algo_trading_engine.models.metrics", "PositionStats"),
    "OptionsRetrieverHelper": (
        "algo_trading_engine.common.options_helpers",
        "OptionsRetrieverHelper",
    ),
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
