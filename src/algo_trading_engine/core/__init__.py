"""
Core interfaces for the trading engine.

This package provides abstract base classes and protocols for strategies,
engines, and data providers.
"""

from .strategy import Strategy
from .engine import TradingEngine, PaperTradingEngine
from .data_provider import DataProvider

# BacktestEngine is in backtest.main to avoid circular imports
# Import it here for convenience
try:
    from algo_trading_engine.backtest.main import BacktestEngine
except ImportError:
    # During initial setup, this might not be available
    BacktestEngine = None

__all__ = [
    'Strategy',
    'TradingEngine',
    'BacktestEngine',
    'PaperTradingEngine',
    'DataProvider',
]

