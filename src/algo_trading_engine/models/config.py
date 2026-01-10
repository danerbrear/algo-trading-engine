"""
Configuration DTOs for backtesting and paper trading.

This module provides immutable configuration objects for engines and strategies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, TYPE_CHECKING

from algo_trading_engine.backtest.config import VolumeConfig as BaseVolumeConfig, VolumeStats as BaseVolumeStats

if TYPE_CHECKING:
    from algo_trading_engine.core.strategy import Strategy


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
    strategy_type: Union[str, 'Strategy']  # Strategy name (str) or Strategy instance (from core.strategy)
    max_position_size: Optional[float] = None  # Fraction of capital (e.g., 0.4 = 40%)
    volume_config: Optional[VolumeConfig] = None
    enable_progress_tracking: bool = True
    quiet_mode: bool = True
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False  # Use free tier rate limiting (13 second timeout)
    lstm_start_date_offset: int = 120  # Days before start_date for LSTM data
    stop_loss: Optional[float] = None  # Optional stop loss percentage
    profit_target: Optional[float] = None  # Optional profit target percentage
    
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
    strategy_type: Union[str, 'Strategy']  # Strategy name (str) or Strategy instance (from core.strategy)
    max_position_size: Optional[float] = None  # Fraction of capital
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False  # Use free tier rate limiting (13 second timeout)
    stop_loss: Optional[float] = None  # Optional stop loss percentage
    profit_target: Optional[float] = None  # Optional profit target percentage
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0")
        if self.max_position_size is not None:
            if not 0 < self.max_position_size <= 1:
                raise ValueError("Max position size must be between 0 and 1")


# Placeholder for future slippage model
class SlippageModel:
    """Placeholder for slippage model implementation."""
    pass

