"""
Configuration DTOs for backtesting and paper trading.

This module provides immutable configuration objects for engines and strategies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from algo_trading_engine.backtest.config import VolumeConfig as BaseVolumeConfig, VolumeStats as BaseVolumeStats


# Re-export VolumeConfig and VolumeStats from backtest.config for backward compatibility
VolumeConfig = BaseVolumeConfig
VolumeStats = BaseVolumeStats


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
    max_position_size: Optional[float] = None  # Fraction of capital (e.g., 0.4 = 40%)
    volume_config: Optional[VolumeConfig] = None
    enable_progress_tracking: bool = True
    quiet_mode: bool = True
    
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


@dataclass(frozen=True)
class PaperTradingConfig:
    """
    Configuration for paper trading engine.
    
    This is an immutable DTO that contains all configuration needed
    to run paper trading.
    """
    initial_capital: float
    symbol: str
    max_position_size: Optional[float] = None  # Fraction of capital
    volume_config: Optional[VolumeConfig] = None
    execution_delay_seconds: int = 0  # Simulate execution delay
    slippage_model: Optional['SlippageModel'] = None  # Optional slippage simulation
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0")
        if self.max_position_size is not None:
            if not 0 < self.max_position_size <= 1:
                raise ValueError("Max position size must be between 0 and 1")
        if self.execution_delay_seconds < 0:
            raise ValueError("Execution delay cannot be negative")
        if self.volume_config is None:
            # Set default volume config
            object.__setattr__(self, 'volume_config', VolumeConfig())


# Placeholder for future slippage model
class SlippageModel:
    """Placeholder for slippage model implementation."""
    pass

