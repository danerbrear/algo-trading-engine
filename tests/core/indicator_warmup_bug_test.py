"""
Tests that validate the paper trading indicator warm-up bug.

Bug: In paper trading, on_new_date is called once for the current bar. Indicators
store a single value. When a strategy looks up an indicator value at a historical
bar date via get_value_at(), it returns None because only the latest bar was updated.

In backtesting, on_new_date is called for every bar in the date range, so indicators
accumulate values at every datetime and get_value_at() works for any historical bar.

These tests demonstrate that the bug exists by contrasting:
  - "backtest-style" iteration (every bar) -> historical lookups succeed
  - "paper-trading-style" single call (last bar only) -> historical lookups return None
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import pandas as pd

from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.core.indicators.indicator import Indicator
from algo_trading_engine.core.indicators.average_true_return_indicator import ATRIndicator


def _create_hourly_data(num_bars: int = 20) -> pd.DataFrame:
    """Create hourly OHLCV data simulating a single trading day with enough bars."""
    dates = pd.date_range(
        start="2026-05-19 09:30:00", periods=num_bars, freq="h"
    )
    return pd.DataFrame(
        {
            "Open": [500 + i for i in range(num_bars)],
            "High": [510 + i for i in range(num_bars)],
            "Low": [490 + i for i in range(num_bars)],
            "Close": [500 + i for i in range(num_bars)],
            "Volume": [1_000_000] * num_bars,
        },
        index=dates,
    )


def _create_daily_data(num_bars: int = 20) -> pd.DataFrame:
    """Create daily OHLCV data."""
    dates = pd.date_range(start="2026-04-20", periods=num_bars, freq="D")
    return pd.DataFrame(
        {
            "Open": [500 + i for i in range(num_bars)],
            "High": [510 + i for i in range(num_bars)],
            "Low": [490 + i for i in range(num_bars)],
            "Close": [500 + i for i in range(num_bars)],
            "Volume": [1_000_000] * num_bars,
        },
        index=dates,
    )


class _HistoricalLookupStrategy(Strategy):
    """Strategy that looks up indicator values at historical bar dates,
    mimicking strategies like uptrend-swing that check ATR K bars ago."""

    def __init__(self, atr: ATRIndicator, lookback_bars: int = 4):
        super().__init__()
        self.add_indicator(atr)
        self.lookback_bars = lookback_bars
        self.historical_atr_value = "NOT_CHECKED"
        self.current_atr_value = "NOT_CHECKED"

    def on_new_date(self, date, positions, add_position, remove_position):
        super().on_new_date(date, positions, add_position, remove_position)

        atr = self.indicators[0]
        self.current_atr_value = atr.get_value_at(date)

        bar_idx = self.data.index.get_loc(date)
        if bar_idx >= self.lookback_bars:
            historical_date = self.data.index[bar_idx - self.lookback_bars]
            self.historical_atr_value = atr.get_value_at(historical_date)

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return True


class TestIndicatorGetValueAtBug:
    """Demonstrate that get_value_at returns None for bars not yet updated."""

    def test_backtest_style_all_bars_updated_has_historical_values(self):
        """When update() is called for every bar (backtest), get_value_at()
        returns a value for any bar that has been processed."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)

        for date in data.index:
            atr.update(date, data)

        historical_date = data.index[4]
        assert atr.get_value_at(historical_date) is not None, (
            "ATR should have a stored value at bar 4 after backtest-style iteration"
        )

    def test_paper_trading_style_single_update_missing_historical_values(self):
        """BUG: When update() is called only for the latest bar (paper trading),
        get_value_at() returns None for every other bar."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)

        latest_bar = data.index[-1]
        atr.update(latest_bar, data)

        earlier_bar = data.index[4]
        result = atr.get_value_at(earlier_bar)
        assert result is None, (
            "BUG CONFIRMED: ATR returns None at an earlier bar when only "
            "the latest bar was updated (paper-trading scenario)"
        )

    def test_single_update_only_stores_one_value(self):
        """After a single update(), exactly one datetime has a stored value."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)

        latest_bar = data.index[-1]
        atr.update(latest_bar, data)

        stored = atr.get_values()
        assert len(stored) == 1, (
            f"Expected exactly 1 stored value, got {len(stored)}"
        )
        assert latest_bar in stored.index

    def test_all_bars_updated_stores_all_values(self):
        """After iterating every bar, every datetime has a stored value
        (at least for bars past the warm-up period)."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)

        for date in data.index:
            atr.update(date, data)

        stored = atr.get_values()
        warm_up = atr.warm_up_period  # period + 1 = 4
        bars_with_values = data.index[warm_up - 1:]
        for bar in bars_with_values:
            assert atr.get_value_at(bar) is not None, (
                f"Expected ATR value at {bar} after full iteration"
            )


class TestStrategyHistoricalLookupBug:
    """Demonstrate the bug at the Strategy level: a strategy that looks up
    indicator values at historical bar dates gets None in paper-trading mode."""

    def test_backtest_iteration_historical_lookup_succeeds(self):
        """Simulating backtest: calling on_new_date for every bar means the
        strategy can look up ATR at a bar K bars ago."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)
        strategy = _HistoricalLookupStrategy(atr, lookback_bars=4)
        strategy.set_data(data)

        mock_add = Mock()
        mock_remove = Mock()

        for date in data.index:
            strategy.on_new_date(date, (), mock_add, mock_remove)

        assert strategy.current_atr_value is not None, (
            "Current ATR should be available after backtest-style iteration"
        )
        assert strategy.historical_atr_value is not None, (
            "Historical ATR (K bars ago) should be available after "
            "backtest-style iteration"
        )

    def test_paper_trading_single_call_historical_lookup_returns_none(self):
        """BUG: Calling on_new_date only for the latest bar means the strategy
        cannot look up ATR at a bar K bars ago — it returns None."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)
        strategy = _HistoricalLookupStrategy(atr, lookback_bars=4)
        strategy.set_data(data)

        mock_add = Mock()
        mock_remove = Mock()

        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), mock_add, mock_remove)

        assert strategy.current_atr_value is not None, (
            "Current ATR should be available even with a single on_new_date call"
        )
        assert strategy.historical_atr_value is None, (
            "BUG CONFIRMED: Historical ATR (K bars ago) is None when "
            "on_new_date is called only once (paper-trading scenario)"
        )


class TestStrategyHasWarmUpMethod:
    """Verify that Strategy now exposes the warm_up_indicators method (post-fix)."""

    def test_strategy_has_warm_up_indicators_method(self):
        """Strategy should have warm_up_indicators after the fix is applied."""
        assert hasattr(Strategy, "warm_up_indicators"), (
            "Strategy must have warm_up_indicators after the fix"
        )
        assert callable(Strategy.warm_up_indicators)


class TestWarmUpWouldFixBug:
    """Show that manually replaying historical bars through indicators (the
    proposed warm_up_indicators approach) resolves the lookup issue."""

    def test_manual_warmup_then_single_on_new_date_has_historical_values(self):
        """If we manually iterate all bars through the indicator before
        calling on_new_date once, historical lookups succeed."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)
        strategy = _HistoricalLookupStrategy(atr, lookback_bars=4)
        strategy.set_data(data)

        for date in data.index:
            for indicator in strategy.indicators:
                indicator.update(date, data)

        mock_add = Mock()
        mock_remove = Mock()
        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), mock_add, mock_remove)

        assert strategy.current_atr_value is not None
        assert strategy.historical_atr_value is not None, (
            "After manual warm-up, historical ATR lookup should succeed"
        )

    def test_without_warmup_historical_lookup_fails(self):
        """Without warm-up, the same scenario fails — confirming the gap."""
        atr = ATRIndicator(period=3)
        data = _create_hourly_data(10)
        strategy = _HistoricalLookupStrategy(atr, lookback_bars=4)
        strategy.set_data(data)

        mock_add = Mock()
        mock_remove = Mock()
        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), mock_add, mock_remove)

        assert strategy.historical_atr_value is None, (
            "Without warm-up, historical ATR should be None"
        )


class TestDailyBarsWarmUpBug:
    """Same bug reproduced with daily bar data to ensure it's not
    frequency-specific."""

    def test_daily_backtest_iteration_has_historical_values(self):
        """Backtest-style iteration with daily bars stores values at every bar."""
        atr = ATRIndicator(period=5)
        data = _create_daily_data(20)

        for date in data.index:
            atr.update(date, data)

        historical_date = data.index[10]
        assert atr.get_value_at(historical_date) is not None

    def test_daily_single_update_missing_historical_values(self):
        """BUG: Single update with daily bars also fails historical lookups."""
        atr = ATRIndicator(period=5)
        data = _create_daily_data(20)

        latest_bar = data.index[-1]
        atr.update(latest_bar, data)

        historical_date = data.index[10]
        result = atr.get_value_at(historical_date)
        assert result is None, (
            "BUG CONFIRMED: Daily bar single-update also returns None for "
            "historical lookups"
        )

    def test_daily_strategy_paper_trading_scenario(self):
        """Full strategy-level test with daily bars in paper-trading mode."""
        atr = ATRIndicator(period=5)
        data = _create_daily_data(20)
        strategy = _HistoricalLookupStrategy(atr, lookback_bars=4)
        strategy.set_data(data)

        mock_add = Mock()
        mock_remove = Mock()
        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), mock_add, mock_remove)

        assert strategy.current_atr_value is not None
        assert strategy.historical_atr_value is None, (
            "BUG CONFIRMED: Daily bar strategy historical lookup returns None "
            "in paper-trading mode"
        )
