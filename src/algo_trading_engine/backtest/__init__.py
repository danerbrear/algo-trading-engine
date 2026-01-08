"""
Backtesting framework.

Internal implementation details - use the public API through the main package:
    from algo_trading_engine import BacktestEngine, BacktestConfig
"""

# Limit public access - only export what's needed for internal use
from .main import BacktestEngine
from .config import VolumeConfig, VolumeStats

__all__ = [
    "BacktestEngine",
    "VolumeConfig",
    "VolumeStats",
]

