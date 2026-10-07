"""Tests for backtest equity-curve frame construction."""

from datetime import datetime

from algo_trading_engine.backtest.equity import build_equity_curve_dataframe


def test_build_equity_curve_dataframe():
    positions = [
        {"exit_date": datetime(2024, 1, 10), "return_dollars": 100.0},
        {"exit_date": datetime(2024, 2, 1), "return_dollars": -50.0},
    ]
    frame = build_equity_curve_dataframe(positions, initial_capital=3000.0)
    assert list(frame.columns) == ["timestamp", "equity"]
    assert len(frame) == 3
    assert frame.iloc[0]["equity"] == 3000.0
    assert frame.iloc[-1]["equity"] == 3050.0
