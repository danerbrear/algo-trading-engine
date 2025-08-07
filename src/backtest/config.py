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
    skip_closure_on_insufficient_volume: bool = True  # Skip closure instead of forcing it
    
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
    positions_rejected_closure_volume: int = 0
    options_checked: int = 0
    skipped_closures: int = 0  # Track skipped closures instead of forced closures
    
    def increment_rejected_positions(self) -> 'VolumeStats':
        """Increment the count of positions rejected due to volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume + 1,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            options_checked=self.options_checked,
            skipped_closures=self.skipped_closures
        )
    
    def increment_rejected_closures(self) -> 'VolumeStats':
        """Increment the count of position closures rejected due to volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume + 1,
            options_checked=self.options_checked,
            skipped_closures=self.skipped_closures + 1  # Increment skipped closures
        )
    
    def increment_skipped_closures(self) -> 'VolumeStats':
        """Increment the count of skipped closures."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            options_checked=self.options_checked,
            skipped_closures=self.skipped_closures + 1
        )
    
    def increment_options_checked(self) -> 'VolumeStats':
        """Increment the count of options checked for volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            options_checked=self.options_checked + 1,
            skipped_closures=self.skipped_closures
        )
    
    def get_summary(self) -> dict:
        """Get a summary of volume validation statistics."""
        total_rejections = self.positions_rejected_volume + self.positions_rejected_closure_volume
        rejection_rate = (total_rejections / max(self.options_checked, 1)) * 100
        
        return {
            'options_checked': self.options_checked,
            'positions_rejected_volume': self.positions_rejected_volume,
            'positions_rejected_closure_volume': self.positions_rejected_closure_volume,
            'skipped_closures': self.skipped_closures,
            'total_rejections': total_rejections,
            'rejection_rate': rejection_rate
        } 