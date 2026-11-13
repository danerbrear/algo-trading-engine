"""Tests for live quote overlay onto OHLCV frames."""

from datetime import datetime

import pandas as pd

from algo_trading_engine._internal.trade.live_bar import apply_live_quote
from algo_trading_engine.enums import BarTimeInterval


def _sample_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [datetime(2026, 11, 11, 9, 30), datetime(2026, 11, 11, 10, 30)]
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_000_000],
        },
        index=idx,
    )


def test_apply_live_quote_updates_close_and_extremes():
    now = datetime(2026, 11, 11, 10, 45)
    result = apply_live_quote(_sample_frame(), 105.0, now, BarTimeInterval.HOUR)
    bar = datetime(2026, 11, 11, 10, 30)
    assert result.loc[bar, "Close"] == 105.0
    assert result.loc[bar, "High"] == 105.0
    assert result.loc[bar, "Low"] == 100.0
    assert result.loc[bar, "Open"] == 101.0


def test_apply_live_quote_appends_missing_bar():
    now = datetime(2026, 11, 11, 12, 15)
    result = apply_live_quote(_sample_frame(), 104.0, now, BarTimeInterval.HOUR)
    bar = datetime(2026, 11, 11, 11, 30)
    assert bar in result.index
    assert result.loc[bar, "Close"] == 104.0
    assert len(result) == 3
