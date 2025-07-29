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
    api_fetch_failures: int = 0
    api_errors: int = 0
    cache_updates: int = 0
    
    def increment_rejected_positions(self) -> 'VolumeStats':
        """Increment the count of positions rejected due to volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume + 1,
            options_checked=self.options_checked,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates
        )
    
    def increment_options_checked(self) -> 'VolumeStats':
        """Increment the count of options checked for volume."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked + 1,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates
        )
    
    def increment_api_fetch_failures(self) -> 'VolumeStats':
        """Increment the count of API fetch failures."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked,
            api_fetch_failures=self.api_fetch_failures + 1,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates
        )
    
    def increment_api_errors(self) -> 'VolumeStats':
        """Increment the count of API errors."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors + 1,
            cache_updates=self.cache_updates
        )
    
    def increment_cache_updates(self) -> 'VolumeStats':
        """Increment the count of cache updates."""
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates + 1
        )
    
    def get_summary(self) -> dict:
        """Get a summary of volume validation statistics."""
        return {
            'positions_rejected_volume': self.positions_rejected_volume,
            'options_checked': self.options_checked,
            'api_fetch_failures': self.api_fetch_failures,
            'api_errors': self.api_errors,
            'cache_updates': self.cache_updates,
            'rejection_rate': (self.positions_rejected_volume / max(self.options_checked, 1)) * 100,
            'api_success_rate': ((self.options_checked - self.api_fetch_failures - self.api_errors) / max(self.options_checked, 1)) * 100
        } 