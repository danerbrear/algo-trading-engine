"""Tests for BarTimeInterval.bar_start."""

from datetime import datetime

from algo_trading_engine.enums import BarTimeInterval


def test_bar_start_daily():
    ts = datetime(2026, 11, 11, 10, 45)
    assert BarTimeInterval.DAY.bar_start(ts) == datetime(2026, 11, 11, 0, 0, 0)


def test_bar_start_hourly_regular_session():
    ts = datetime(2026, 11, 11, 10, 45)
    assert BarTimeInterval.HOUR.bar_start(ts) == datetime(2026, 11, 11, 10, 30)


def test_bar_start_hourly_before_open():
    ts = datetime(2026, 11, 11, 8, 0)
    assert BarTimeInterval.HOUR.bar_start(ts) == datetime(2026, 11, 11, 9, 30)
