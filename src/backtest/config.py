"""
Configuration system for backtesting volume validation.

This module provides DTOs for volume validation configuration and statistics tracking.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VolumeConfig:
    """
    DTO for volume validation configuration.
    
    Immutable configuration object for volume validation settings.
    """
    min_volume: int = 10
    enable_volume_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")
        if self.min_volume == 0:
            raise ValueError("Minimum volume must be greater than 0")


@dataclass(frozen=True)
class VolumeStats:
    """
    DTO for tracking volume validation statistics.
    
    Immutable statistics object for tracking volume validation metrics.
    """
    positions_rejected_volume: int = 0
    options_checked: int = 0
    
    def increment_rejected_positions(self) -> 'VolumeStats':
        """Increment the count of positions rejected due to volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume + 1,
            options_checked=self.options_checked
        )
    
    def increment_options_checked(self) -> 'VolumeStats':
        """Increment the count of options checked for volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked + 1
        )
    
    def get_summary(self) -> dict:
        """Get a summary of volume validation statistics."""
        return {
            'positions_rejected_volume': self.positions_rejected_volume,
            'options_checked': self.options_checked,
            'rejection_rate': (self.positions_rejected_volume / max(self.options_checked, 1)) * 100
        } 