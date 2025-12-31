"""
Core interfaces for the trading engine.

This package provides abstract base classes and protocols for strategies,
engines, and data providers.
"""

from .strategy import Strategy
from .engine import TradingEngine, PaperTradingEngine
from .data_provider import DataProvider

from algo_trading_engine.backtest.main import BacktestEngine

__all__ = [
    'Strategy',
    'TradingEngine',
    'BacktestEngine',
    'PaperTradingEngine',
    'DataProvider',
]

