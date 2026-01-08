"""
Core interfaces for the trading engine.

Internal implementation details - use the public API through the main package:
    from algo_trading_engine import Strategy, PaperTradingEngine
"""

from .strategy import Strategy
from .engine import TradingEngine, PaperTradingEngine
from .data_provider import DataProvider

__all__ = [
    'Strategy',
    'TradingEngine',
    'PaperTradingEngine',
    'DataProvider',
]

