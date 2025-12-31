"""
Trading engine interfaces and implementations.

This module provides the abstract base class for trading engines
and concrete implementations for backtesting and paper trading.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
import pandas as pd

from .strategy import Strategy
from .data_provider import DataProvider
from algo_trading_engine.models.config import BacktestConfig, PaperTradingConfig
from algo_trading_engine.models.metrics import PerformanceMetrics

if TYPE_CHECKING:
    from algo_trading_engine.backtest.models import Position


class TradingEngine(ABC):
    """
    Abstract base class for trading engines.
    
    Both backtesting and paper trading engines implement this interface,
    allowing for unified usage patterns.
    """
    
    @abstractmethod
    def run(self) -> bool:
        """
        Execute the trading simulation.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance statistics for the trading session.
        
        Returns:
            PerformanceMetrics object with all performance data
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List['Position']:
        """
        Get current open positions.
        
        Returns:
            List of currently open Position objects
        """
        pass
    
    @property
    @abstractmethod
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
        pass
    
    @property
    @abstractmethod
    def capital(self) -> float:
        """Get current capital."""
        pass


# BacktestEngine is defined in backtest.main and implements TradingEngine
# We'll import it here for convenience, but it's defined in backtest/main.py
# to avoid circular imports


class PaperTradingEngine(TradingEngine):
    """
    Paper trading engine implementation.
    
    This engine runs strategies against live market data in real-time,
    simulating trades without actually executing them.
    """
    
    def __init__(
        self,
        strategy: Strategy,
        data_provider: DataProvider,
        config: PaperTradingConfig
    ):
        """
        Initialize paper trading engine.
        
        Args:
            strategy: Trading strategy to execute
            data_provider: Data provider for live market data
            config: Paper trading configuration
        """
        self._strategy = strategy
        self._data_provider = data_provider
        self._config = config
        self._capital = config.initial_capital
        self._positions: List['Position'] = []
        self._closed_positions: List[dict] = []
        self._running = False
    
    @property
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
        return self._strategy
    
    @property
    def capital(self) -> float:
        """Get current capital."""
        return self._capital
    
    def run(self) -> bool:
        """
        Execute paper trading (placeholder implementation).
        
        TODO: Implement real-time paper trading logic.
        """
        raise NotImplementedError("PaperTradingEngine.run() not yet implemented")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance statistics (placeholder implementation).
        
        TODO: Implement performance metrics calculation.
        """
        raise NotImplementedError("PaperTradingEngine.get_performance_metrics() not yet implemented")
    
    def get_positions(self) -> List['Position']:
        """Get current open positions."""
        return self._positions.copy()

