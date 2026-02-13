"""
Performance metrics Value Objects.

This module provides immutable value objects for tracking and reporting
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from algo_trading_engine.common.logger import log_and_echo

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

    def print_summary(self) -> None:
        """Print a human-readable summary of this position's stats."""
        log_and_echo(f"  {self.strategy_type.value}: entry {self.entry_date.date()} -> exit {self.exit_date.date()}, "
                     f"P&L ${self.return_dollars:+,.2f} ({self.return_percentage:+.2f}%), days held {self.days_held}, max risk ${self.max_risk:.2f}")


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

    def print_summary(self) -> None:
        """Print a human-readable summary of this strategy's performance."""
        name = self.strategy_type.value.replace('_', ' ').title()
        log_and_echo(f"  {name}:")
        log_and_echo(f"    Positions: {self.positions_count}")
        log_and_echo(f"    Win rate: {self.win_rate:.1f}%")
        log_and_echo(f"    Total P&L: ${self.total_pnl:+,.2f}")
        log_and_echo(f"    Avg return: ${self.average_return:+,.2f}")
        log_and_echo(f"    Drawdowns: min {self.min_drawdown:.2f}% / mean {self.mean_drawdown:.2f}% / max {self.max_drawdown:.2f}%")


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

    def print_summary(self) -> None:
        """Print a human-readable summary of overall performance."""
        log_and_echo("  Overall:")
        log_and_echo(f"    Total positions: {self.total_positions}")
        log_and_echo(f"    Win rate: {self.win_rate:.1f}%")
        log_and_echo(f"    Total P&L: ${self.total_pnl:+,.2f}")
        log_and_echo(f"    Avg return: ${self.average_return:+,.2f}")
        log_and_echo(f"    Drawdowns: min {self.min_drawdown:.2f}% / mean {self.mean_drawdown:.2f}% / max {self.max_drawdown:.2f}%")


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

    def print_summary(self) -> None:
        """Print a human-readable summary of all performance metrics."""
        log_and_echo("Performance Metrics Summary")
        log_and_echo("=" * 50)
        log_and_echo(f"Total return: ${self.total_return:+,.2f} ({self.total_return_pct:+.2f}%)")
        log_and_echo(f"Sharpe ratio: {self.sharpe_ratio:.3f}")
        log_and_echo(f"Max drawdown: {self.max_drawdown:.2f}%")
        log_and_echo(f"Win rate: {self.win_rate:.1f}%")
        log_and_echo(f"Total positions: {self.total_positions}")
        if self.benchmark_return is not None and self.benchmark_return_pct is not None:
            log_and_echo(f"Benchmark return: ${self.benchmark_return:+,.2f} ({self.benchmark_return_pct:+.2f}%)")
        log_and_echo("")
        if self.overall_stats:
            log_and_echo("Overall:")
            self.overall_stats.print_summary()
        log_and_echo("")
        if self.strategy_stats:
            log_and_echo("By strategy:")
            for s in self.strategy_stats:
                s.print_summary()
        if self.closed_positions:
            log_and_echo("")
            log_and_echo("Closed positions:")
            for p in self.closed_positions:
                p.print_summary()

