"""
Configuration DTOs for backtesting and paper trading.

This module provides immutable configuration objects for engines and strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, TYPE_CHECKING

from algo_trading_engine.enums import BarTimeInterval

if TYPE_CHECKING:
    from algo_trading_engine._internal.common.run_observer import RunObserver
    from algo_trading_engine.database.decision_store import DecisionStore
    from algo_trading_engine.plotting.config import PlotConfig
    from algo_trading_engine.strategy import Strategy


@dataclass(frozen=True)
class VolumeConfig:
    """Configuration for volume validation settings."""

    min_volume: int = 10
    enable_volume_validation: bool = True
    skip_closure_on_insufficient_volume: bool = True

    def __post_init__(self):
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")
        if self.min_volume == 0:
            raise ValueError("Minimum volume must be greater than 0")


@dataclass(frozen=True)
class VolumeStats:
    """Statistics tracking for volume validation."""

    options_checked: int = 0
    positions_rejected_volume: int = 0
    positions_rejected_closure_volume: int = 0
    skipped_closures: int = 0

    def increment_options_checked(self) -> VolumeStats:
        return VolumeStats(
            options_checked=self.options_checked + 1,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures,
        )

    def increment_rejected_positions(self) -> VolumeStats:
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume + 1,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures,
        )

    def increment_rejected_closures(self) -> VolumeStats:
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume + 1,
            skipped_closures=self.skipped_closures + 1,
        )

    def increment_skipped_closures(self) -> VolumeStats:
        return VolumeStats(
            options_checked=self.options_checked,
            positions_rejected_volume=self.positions_rejected_volume,
            positions_rejected_closure_volume=self.positions_rejected_closure_volume,
            skipped_closures=self.skipped_closures + 1,
        )

    def get_summary(self) -> dict:
        total_rejections = self.positions_rejected_volume + self.positions_rejected_closure_volume
        total_checked = self.options_checked
        return {
            "options_checked": self.options_checked,
            "positions_rejected_volume": self.positions_rejected_volume,
            "positions_rejected_closure_volume": self.positions_rejected_closure_volume,
            "skipped_closures": self.skipped_closures,
            "total_rejections": total_rejections,
            "rejection_rate": (total_rejections / total_checked * 100) if total_checked > 0 else 0,
        }


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration for backtesting engine.
    
    This is an immutable DTO that contains all configuration needed
    to run a backtest.
    """
    initial_capital: float
    start_date: datetime
    end_date: datetime
    symbol: str
    strategy_type: Union[str, 'Strategy']  # Strategy name (str) or Strategy instance (from core.strategy)
    bar_interval: BarTimeInterval = BarTimeInterval.DAY  # Time interval for market data bars
    max_position_size: Optional[float] = None  # Fraction of capital (e.g., 0.4 = 40%)
    volume_config: Optional[VolumeConfig] = None
    enable_progress_tracking: bool = True
    quiet_mode: bool = True
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False  # Use free tier rate limiting (13 second timeout)
    lstm_start_date_offset: int = 120  # Days before start_date for LSTM data
    stop_loss: Optional[float] = None  # Optional stop loss percentage
    profit_target: Optional[float] = None  # Optional profit target percentage
    benchmark_ticker: Optional[str] = None  # Ticker for buy-and-hold benchmark comparison (defaults to symbol)
    plot_config: Optional["PlotConfig"] = None  # None disables local plotting and matplotlib import
    observer: Optional["RunObserver"] = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.max_position_size is not None:
            if not 0 < self.max_position_size <= 1:
                raise ValueError("Max position size must be between 0 and 1")
        if self.volume_config is None:
            # Set default volume config
            object.__setattr__(self, 'volume_config', VolumeConfig())
        
        # Validate bar_interval for date range (yfinance limitations)
        if self.bar_interval != BarTimeInterval.DAY:
            days_diff = (self.end_date - self.start_date).days
            
            if self.bar_interval == BarTimeInterval.HOUR and days_diff > 729:
                raise ValueError(
                    f"Hourly bars are limited to 729 days (yfinance restriction). "
                    f"Requested range: {days_diff} days ({self.start_date.date()} to {self.end_date.date()}). "
                    f"Use daily bars or reduce the date range."
                )
            elif self.bar_interval == BarTimeInterval.MINUTE and days_diff > 59:
                raise ValueError(
                    f"Minute bars are limited to 59 days (yfinance restriction). "
                    f"Requested range: {days_diff} days ({self.start_date.date()} to {self.end_date.date()}). "
                    f"Use hourly/daily bars or reduce the date range."
                )


@dataclass(frozen=True)
class PaperTradingConfig:
    """
    Configuration for paper trading engine.
    
    This is an immutable DTO that contains all configuration needed
    to run paper trading. Capital allocation is managed separately
    through config/strategies/capital_allocations.json.
    """
    symbol: str
    strategy_type: Union[str, 'Strategy']  # Strategy name (str) or Strategy instance (from core.strategy)
    bar_interval: BarTimeInterval = BarTimeInterval.DAY  # Time interval for market data bars
    max_position_size: Optional[float] = None  # Fraction of capital
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False  # Use free tier rate limiting (13 second timeout)
    use_cache: bool = True  # Whether to write fetched data to the local filesystem cache
    stop_loss: Optional[float] = None  # Optional stop loss percentage
    profit_target: Optional[float] = None  # Optional profit target percentage
    decision_store: Optional['DecisionStore'] = None # Database immplementation for storing decisions
    auto_yes: bool = False  # Skip interactive prompts (e.g., for Lambda / headless environments)
    observer: Optional["RunObserver"] = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_position_size is not None:
            if not 0 < self.max_position_size <= 1:
                raise ValueError("Max position size must be between 0 and 1")


# Placeholder for future slippage model
class SlippageModel:
    """Placeholder for slippage model implementation."""
    pass

