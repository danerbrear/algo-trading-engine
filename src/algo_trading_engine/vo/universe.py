"""Universe: symbol selection, validated market-data frames, and optional post-load enrichment."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd
from pandera import Column, DataFrameSchema

from algo_trading_engine._internal.util.market_clock import market_index, market_time
from algo_trading_engine.enums import BarTimeInterval

if TYPE_CHECKING:
    from algo_trading_engine.data_retriever import DataRetriever

UniverseSet = list[str]
PostLoadCallback = Callable[[str, pd.DataFrame], pd.DataFrame]


def identity_post_load(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` unchanged (default ``post_load``)."""
    del symbol
    return data


def default_ohlcv_schema() -> DataFrameSchema:
    """Pandera schema for yfinance OHLCV bars (additional columns allowed after ``post_load``)."""
    return DataFrameSchema(
        {
            "Open": Column(float, nullable=True),
            "High": Column(float, nullable=True),
            "Low": Column(float, nullable=True),
            "Close": Column(float, nullable=True),
            "Volume": Column(float, nullable=True),
        },
        strict=False,
        coerce=True,
    )


class Universe:
    """A set of symbols and their validated market-data frames."""

    def __init__(
        self,
        universe_set: UniverseSet,
        *,
        schema: Optional[DataFrameSchema] = None,
        post_load: Optional[PostLoadCallback] = None,
    ):
        if not universe_set:
            raise ValueError("Universe requires at least one symbol")
        self._symbols = tuple(s.upper() for s in universe_set)
        self.schema = schema if schema is not None else default_ohlcv_schema()
        self.post_load = post_load if post_load is not None else identity_post_load
        self._frames: dict[str, pd.DataFrame] = {}

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def load(
        self,
        retriever: DataRetriever,
        bar_interval: BarTimeInterval,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> None:
        """Fetch, enrich, validate, and store a frame for every symbol."""
        self._frames.clear()
        for symbol in self._symbols:
            raw = retriever.fetch_data_for_period(symbol, bar_interval, start_date, end_date)
            self.replace(symbol, raw)

    def replace(self, symbol: str, data: pd.DataFrame) -> None:
        """Run ``post_load``, validate, and store a frame for ``symbol``."""
        key = symbol.upper()
        if key not in self._symbols:
            raise ValueError(f"Symbol {symbol!r} is not in this universe")
        enriched = self.post_load(key, data)
        self._frames[key] = self.schema.validate(enriched)

    def frame(self, symbol: str) -> pd.DataFrame:
        """Return the stored frame for ``symbol``."""
        key = symbol.upper()
        if key not in self._frames:
            raise KeyError(f"No loaded frame for {symbol!r}; call load() first")
        return self._frames[key]

    def timestamps(self) -> pd.DatetimeIndex:
        """Sorted union of bar timestamps across all stored frames."""
        if not self._frames:
            return pd.DatetimeIndex([])
        indices = [market_index(frame) for frame in self._frames.values()]
        union = indices[0]
        for index in indices[1:]:
            union = union.union(index)
        return pd.DatetimeIndex(union.sort_values())

    def has_bar(self, symbol: str, when: datetime) -> bool:
        """Whether ``symbol`` has a bar at ``when``."""
        try:
            frame = self.frame(symbol)
        except KeyError:
            return False
        return when in market_index(frame)

    def bar_timestamp_at_or_before(self, symbol: str, when: datetime) -> Optional[datetime]:
        """Latest bar time for ``symbol`` at or before ``when``."""
        frame = self.frame(symbol)
        index = market_index(frame)
        candidates = index[index <= when]
        if len(candidates) == 0:
            return None
        return market_time(candidates[-1])
