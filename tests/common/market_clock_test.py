"""Tests for US Eastern market clock helpers."""

from datetime import datetime
from unittest.mock import patch

from algo_trading_engine._internal.util.market_clock import market_now, market_today


def test_market_now_is_naive_without_subsecond_precision():
    result = market_now()
    assert result.tzinfo is None
    assert result.microsecond == 0


def test_market_today_midnight():
    with patch(
        "algo_trading_engine._internal.util.market_clock.market_now",
        return_value=datetime(2026, 11, 11, 14, 0, 0),
    ):
        assert market_today() == datetime(2026, 11, 11, 0, 0, 0)
