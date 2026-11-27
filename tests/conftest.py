"""Shared pytest configuration for algo-trading-engine."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from pandera.pandas import DataFrameSchema

from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo import TreasuryRates, Universe
from algo_trading_engine.vo.universe import identity_post_load

# Non-interactive backend before any test imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


def universe_from_frame(symbol: str, data: pd.DataFrame, schema=None) -> Universe:
    """Build a single-symbol universe from an in-memory OHLCV frame."""
    universe = Universe([symbol], schema=schema)
    _store_fixture_frame(universe, symbol, data)
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
    _store_fixture_frame(universe, key, data)
    strategy.set_treasury_rates(treasury_rates)


def _store_fixture_frame(universe: Universe, symbol: str, data: pd.DataFrame) -> None:
    """Store a test frame without re-running enrich or rejecting extra columns."""
    original_post_load = universe.post_load
    original_schema = universe.schema
    universe.post_load = identity_post_load
    universe.schema = DataFrameSchema(original_schema.columns, strict=False, coerce=True)
    try:
        universe.replace(symbol, _ensure_ohlcv(data))
    finally:
        universe.post_load = original_post_load
        universe.schema = original_schema


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
