"""
Backtesting framework.

Internal implementation details - use the public API through the main package:
    from algo_trading_engine import BacktestEngine, BacktestConfig
"""

from __future__ import annotations

from typing import Any

from .config import VolumeConfig, VolumeStats

__all__ = [
    "BacktestEngine",
    "VolumeConfig",
    "VolumeStats",
]


def __getattr__(name: str) -> Any:
    if name == "BacktestEngine":
        from .main import BacktestEngine

        globals()["BacktestEngine"] = BacktestEngine
        return BacktestEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
