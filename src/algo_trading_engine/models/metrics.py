"""
Performance metrics Value Objects.

This module provides immutable value objects for tracking and reporting
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# StrategyType is not imported here to avoid circular imports
# With __future__.annotations, all type hints are strings, so no runtime import needed
# Type checkers will resolve 'StrategyType' from algo_trading_engine.common.models


@dataclass(frozen=True)
class PositionStats:
    """
    Statistics for a single closed position.
    
    This is an immutable value object representing performance
    metrics for one position.
    """
    strategy_type: 'StrategyType'
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    return_dollars: float
    return_percentage: float
    days_held: int
    max_risk: float
    
    def __post_init__(self):
        """Validate position stats after initialization."""
        if self.entry_date >= self.exit_date:
            raise ValueError("Entry date must be before exit date")
        if self.days_held < 0:
            raise ValueError("Days held cannot be negative")


@dataclass(frozen=True)
class StrategyPerformanceStats:
    """
    Performance statistics for a specific strategy type.
    
    This aggregates performance metrics for all positions
    of a particular strategy type.
    """
    strategy_type: 'StrategyType'
    positions_count: int
    win_rate: float  # Percentage (0-100)
    total_pnl: float
    average_return: float
    min_drawdown: float
    mean_drawdown: float
    max_drawdown: float
    
    def __post_init__(self):
        """Validate strategy stats after initialization."""
        if self.positions_count < 0:
            raise ValueError("Positions count cannot be negative")
        if not 0 <= self.win_rate <= 100:
            raise ValueError("Win rate must be between 0 and 100")


@dataclass(frozen=True)
class OverallPerformanceStats:
    """
    Overall performance statistics across all strategies.
    
    This aggregates performance metrics for all positions
    regardless of strategy type.
    """
    total_positions: int
    win_rate: float  # Percentage (0-100)
    total_pnl: float
    average_return: float
    min_drawdown: float
    mean_drawdown: float
    max_drawdown: float
    
    def __post_init__(self):
        """Validate overall stats after initialization."""
        if self.total_positions < 0:
            raise ValueError("Total positions cannot be negative")
        if not 0 <= self.win_rate <= 100:
            raise ValueError("Win rate must be between 0 and 100")


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Complete performance metrics for a trading session.
    
    This is the main value object returned by engines to report
    overall performance.
    """
    total_return: float  # Dollar return
    total_return_pct: float  # Percentage return
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float  # Percentage (0-100)
    total_positions: int
    closed_positions: List[PositionStats]
    strategy_stats: List[StrategyPerformanceStats]
    overall_stats: OverallPerformanceStats
    benchmark_return: Optional[float] = None  # Dollar return
    benchmark_return_pct: Optional[float] = None  # Percentage return
    
    def __post_init__(self):
        """Validate performance metrics after initialization."""
        if self.total_positions < 0:
            raise ValueError("Total positions cannot be negative")
        if not 0 <= self.win_rate <= 100:
            raise ValueError("Win rate must be between 0 and 100")
        if len(self.closed_positions) > self.total_positions:
            raise ValueError("Closed positions cannot exceed total positions")

