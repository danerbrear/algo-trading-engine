"""Shared pytest configuration for algo-trading-engine."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo import TreasuryRates, Universe

# Non-interactive backend before any test imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


def universe_from_frame(symbol: str, data: pd.DataFrame, schema=None) -> Universe:
    """Build a single-symbol universe from an in-memory OHLCV frame."""
    universe = Universe([symbol], schema=schema)
    universe.replace(symbol, _ensure_ohlcv(data))
    return universe


def load_strategy_frame(
    strategy: Strategy,
    data: pd.DataFrame,
    treasury_rates: Optional[TreasuryRates] = None,
    *,
    symbol: Optional[str] = None,
) -> None:
    """Store ``data`` on the strategy's instance universe."""
    universe = strategy.universe
    if universe is None:
        raise ValueError("Strategy has no universe")
    key = symbol if symbol is not None else universe.symbols[0]
    universe.replace(key, _ensure_ohlcv(data))
    strategy.set_treasury_rates(treasury_rates)


def _ensure_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing OHLCV columns from Close so schema validation can run."""
    if "Close" not in data.columns:
        return data
    frame = data.copy()
    if "Open" not in frame.columns:
        frame["Open"] = frame["Close"]
    if "High" not in frame.columns:
        frame["High"] = frame["Close"]
    if "Low" not in frame.columns:
        frame["Low"] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0
    return frame
