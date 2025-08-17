"""
Configuration system for backtesting volume validation.

This module provides DTOs for volume validation configuration and statistics tracking.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum


class StrategyType(Enum):
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"


@dataclass(frozen=True)
class VolumeConfig:
    """Configuration for volume validation settings"""
    min_volume: int = 10
    enable_volume_validation: bool = True
    skip_closure_on_insufficient_volume: bool = True
    
    def __post_init__(self):
        """Validate min_volume after initialization"""
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")
        if self.min_volume == 0:
            raise ValueError("Minimum volume must be greater than 0")


@dataclass(frozen=True)
class VolumeStats:
    """Statistics tracking for volume validation"""
    options_checked: int = 0
    positions_rejected_volume: int = 0
    positions_rejected_closure_volume: int = 0
    skipped_closures: int = 0

    def increment_options_checked(self) -> 'VolumeStats':
        """Increment the number of options checked"""
        return VolumeStats(
            options_checked=self.options_checked + 1,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures
        )

    def increment_rejected_positions(self) -> 'VolumeStats':
        """Increment the number of positions rejected due to volume"""
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume + 1,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures
        )

    def increment_rejected_closures(self) -> 'VolumeStats':
        """Increment the number of position closures rejected due to volume"""
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume + 1,
            skipped_closures=self.skipped_closures + 1
        )

    def increment_skipped_closures(self) -> 'VolumeStats':
        """Increment the number of skipped closures"""
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures + 1
        )

    def get_summary(self) -> dict:
        """Get a summary of volume statistics"""
        total_rejections = self.positions_rejected_volume + self.positions_rejected_closure_volume
        total_checked = self.options_checked
        
        return {
            'options_checked': self.options_checked,
            'positions_rejected_volume': self.positions_rejected_volume,
            'positions_rejected_closure_volume': self.positions_rejected_closure_volume,
            'skipped_closures': self.skipped_closures,
            'total_rejections': total_rejections,
            'rejection_rate': (total_rejections / total_checked * 100) if total_checked > 0 else 0
        }


@dataclass(frozen=True)
class PositionStatistics:
    """Statistics for a single position"""
    strategy_type: StrategyType
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_dollars: float
    return_percentage: float
    days_held: int
    max_risk: float


@dataclass(frozen=True)
class StrategyPerformanceStats:
    """Performance statistics for a specific strategy type"""
    strategy_type: StrategyType
    positions_count: int
    win_rate: float
    total_pnl: float
    average_return: float
    average_drawdown: float


@dataclass(frozen=True)
class OverallPerformanceStats:
    """Overall performance statistics"""
    total_positions: int
    win_rate: float
    total_pnl: float
    average_return: float
    average_drawdown: float 