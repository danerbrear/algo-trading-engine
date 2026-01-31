"""
Algo Trading Engine - Options trading backtesting and paper trading framework.

This package provides a clean public API for building and testing trading strategies.

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

# Import public classes
from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.core.engine import PaperTradingEngine
from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.models.config import (
    BacktestConfig,
    PaperTradingConfig,
    VolumeConfig,
    VolumeStats,
)
from algo_trading_engine.models.metrics import (
    PerformanceMetrics,
    PositionStats,
)
from algo_trading_engine.common.options_helpers import OptionsRetrieverHelper
# Import sub-packages for easy access
from algo_trading_engine import dto
from algo_trading_engine import vo, enums, indicators

# Define public API
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
]
