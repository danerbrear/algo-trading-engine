"""
Core interfaces for the trading engine.

Internal implementation details - use the public API through the main package:
    from algo_trading_engine import Strategy, PaperTradingEngine
"""

from .strategy import Strategy
from .engine import TradingEngine, PaperTradingEngine

__all__ = [
    'Strategy',
    'TradingEngine',
    'PaperTradingEngine',
]

