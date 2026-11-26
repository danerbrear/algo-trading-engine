"""Tests for Universe validation and post_load hooks."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from algo_trading_engine.vo.universe import Universe


def _ohlcv_frame(rows: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * rows,
            "High": [101.0] * rows,
            "Low": [99.0] * rows,
            "Close": [100.5] * rows,
            "Volume": [1_000_000.0] * rows,
        },
        index=index,
    )


class TestUniverse:
    def test_empty_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one symbol"):
            Universe([])

    def test_schema_failure_propagates(self) -> None:
        universe = Universe(["SPY"])
        bad = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-02", periods=2, freq="B"))
        with pytest.raises(SchemaError):
            universe.replace("SPY", bad)

    def test_post_load_adds_columns_before_validation(self) -> None:
        def add_derived(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
            del symbol
            out = data.copy()
            out["Derived"] = out["Close"] * 2
            return out

        universe = Universe(["SPY"], post_load=add_derived)
        universe.replace("SPY", _ohlcv_frame())
        frame = universe.frame("SPY")
        assert "Derived" in frame.columns
        assert frame["Derived"].iloc[0] == pytest.approx(201.0)

    def test_timestamps_union(self) -> None:
        universe = Universe(["A", "B"])
        universe.replace("A", _ohlcv_frame(3))
        later = _ohlcv_frame(3)
        later.index = pd.date_range("2024-01-10", periods=3, freq="B")
        universe.replace("B", later)
        stamps = universe.timestamps()
        assert len(stamps) == 6
