"""
Core interfaces for the trading engine.

This package provides abstract base classes and protocols for strategies,
engines, and data providers.
"""

from .strategy import Strategy
from .engine import TradingEngine, PaperTradingEngine
from .data_provider import DataProvider

# BacktestEngine is imported here for convenience, but defined in backtest.main
# to avoid circular imports. Import it lazily or directly from backtest.main where needed.
# from algo_trading_engine.backtest.main import BacktestEngine  # Removed to break circular import

__all__ = [
    'Strategy',
    'TradingEngine',
    'PaperTradingEngine',
    'DataProvider',
]

