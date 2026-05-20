"""
Tests that assert the desired behavior AFTER the warm-up fix is applied.

These tests FAIL on the current codebase (Strategy.warm_up_indicators does not
exist yet) and will PASS once the fix from fix-paper-trading-indicator-warmup.md
is implemented.

The fix adds Strategy.warm_up_indicators() which replays all historical bars
through each indicator so that get_value_at(historical_date) returns a value
for every bar — matching backtest behavior in paper-trading contexts.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.core.indicators.indicator import Indicator
from algo_trading_engine.core.indicators.average_true_return_indicator import ATRIndicator
from algo_trading_engine.core.indicators.sma_indicator import SMAIndicator
from algo_trading_engine.enums import BarTimeInterval


def _make_hourly_data(num_bars: int = 20) -> pd.DataFrame:
    dates = pd.date_range(start="2026-05-19 09:30:00", periods=num_bars, freq="h")
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


def _make_daily_data(num_bars: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start="2026-04-01", periods=num_bars, freq="D")
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


class _SimpleStrategy(Strategy):
    """Minimal concrete Strategy for testing warm_up_indicators."""

    def on_new_date(self, date, positions, add_position, remove_position):
        super().on_new_date(date, positions, add_position, remove_position)

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return data is not None and not data.empty


class TestWarmUpIndicatorsExists:
    """Strategy must expose a public warm_up_indicators() method."""

    def test_warm_up_indicators_is_callable(self):
        strategy = _SimpleStrategy()
        assert hasattr(strategy, "warm_up_indicators"), (
            "Strategy must have a warm_up_indicators method"
        )
        assert callable(strategy.warm_up_indicators)


class TestWarmUpIndicatorsATR:
    """After warm_up_indicators(), ATR should have stored values at every
    bar past its warm-up period."""

    def test_atr_has_values_at_all_bars_after_warmup(self):
        atr = ATRIndicator(period=3)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        bars_after_warmup = data.index[atr.warm_up_period - 1:]
        for bar in bars_after_warmup:
            assert atr.get_value_at(bar) is not None, (
                f"ATR should have a value at {bar} after warm-up"
            )

    def test_atr_historical_lookup_succeeds_after_warmup_then_single_on_new_date(self):
        """Paper-trading scenario: warm_up_indicators + one on_new_date call."""
        atr = ATRIndicator(period=3)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), Mock(), Mock())

        earlier_bar = data.index[5]
        assert atr.get_value_at(earlier_bar) is not None, (
            "Historical ATR lookup must succeed after warm-up"
        )

    def test_atr_daily_warmup_populates_all_bars(self):
        atr = ATRIndicator(period=5)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        data = _make_daily_data(20)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        bars_after_warmup = data.index[atr.warm_up_period - 1:]
        for bar in bars_after_warmup:
            assert atr.get_value_at(bar) is not None, (
                f"Daily ATR should have a value at {bar} after warm-up"
            )


class TestWarmUpIndicatorsSMA:
    """After warm_up_indicators(), SMA should have stored values at every
    bar past its warm-up period."""

    def test_sma_has_values_at_all_bars_after_warmup(self):
        sma = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        strategy = _SimpleStrategy()
        strategy.add_indicator(sma)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        bars_after_warmup = data.index[sma.warm_up_period - 1:]
        for bar in bars_after_warmup:
            assert sma.get_value_at(bar) is not None, (
                f"SMA should have a value at {bar} after warm-up"
            )

    def test_sma_historical_lookup_succeeds_after_warmup_then_single_on_new_date(self):
        sma = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        strategy = _SimpleStrategy()
        strategy.add_indicator(sma)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        latest_bar = data.index[-1]
        strategy.on_new_date(latest_bar, (), Mock(), Mock())

        earlier_bar = data.index[6]
        assert sma.get_value_at(earlier_bar) is not None, (
            "Historical SMA lookup must succeed after warm-up"
        )


class TestWarmUpIndicatorsMultiple:
    """warm_up_indicators must process all attached indicators."""

    def test_multiple_indicators_all_warmed_up(self):
        atr = ATRIndicator(period=3)
        sma = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        strategy.add_indicator(sma)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()

        test_bar = data.index[6]
        assert atr.get_value_at(test_bar) is not None, (
            "ATR must have a value after warm-up"
        )
        assert sma.get_value_at(test_bar) is not None, (
            "SMA must have a value after warm-up"
        )


class TestWarmUpIndicatorsNoOp:
    """warm_up_indicators should be a safe no-op in degenerate cases."""

    def test_no_op_when_no_data(self):
        atr = ATRIndicator(period=3)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)

        strategy.warm_up_indicators()

        assert atr.value is None

    def test_no_op_when_empty_data(self):
        atr = ATRIndicator(period=3)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        strategy.set_data(pd.DataFrame())

        strategy.warm_up_indicators()

        assert atr.value is None

    def test_no_op_when_no_indicators(self):
        strategy = _SimpleStrategy()
        data = _make_hourly_data(10)
        strategy.set_data(data)

        strategy.warm_up_indicators()


class TestWarmUpIndicatorsLogging:
    """warm_up_indicators should log a completion message."""

    def test_logs_warmup_completion(self):
        atr = ATRIndicator(period=3)
        strategy = _SimpleStrategy()
        strategy.add_indicator(atr)
        data = _make_hourly_data(10)
        strategy.set_data(data)

        with patch("algo_trading_engine.core.strategy.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            strategy.warm_up_indicators()

            mock_logger.info.assert_any_call(
                "Indicator warm-up complete: 1 indicator(s), 10 bar(s)"
            )


class TestWarmUpIndicatorsBacktestParity:
    """Values produced by warm_up_indicators + single on_new_date must match
    values produced by iterating on_new_date over every bar (backtest style)."""

    def test_atr_values_match_backtest_iteration(self):
        data = _make_hourly_data(10)

        # Backtest path: iterate every bar
        atr_backtest = ATRIndicator(period=3)
        backtest_strategy = _SimpleStrategy()
        backtest_strategy.add_indicator(atr_backtest)
        backtest_strategy.set_data(data)
        for date in data.index:
            backtest_strategy.on_new_date(date, (), Mock(), Mock())

        # Paper-trading path: warm-up then single call
        atr_paper = ATRIndicator(period=3)
        paper_strategy = _SimpleStrategy()
        paper_strategy.add_indicator(atr_paper)
        paper_strategy.set_data(data)
        paper_strategy.warm_up_indicators()
        paper_strategy.on_new_date(data.index[-1], (), Mock(), Mock())

        bars_after_warmup = data.index[atr_backtest.warm_up_period - 1:]
        for bar in bars_after_warmup:
            backtest_val = atr_backtest.get_value_at(bar)
            paper_val = atr_paper.get_value_at(bar)
            assert backtest_val == pytest.approx(paper_val), (
                f"ATR mismatch at {bar}: backtest={backtest_val}, paper={paper_val}"
            )

    def test_sma_values_match_backtest_iteration(self):
        data = _make_hourly_data(10)

        sma_backtest = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        backtest_strategy = _SimpleStrategy()
        backtest_strategy.add_indicator(sma_backtest)
        backtest_strategy.set_data(data)
        for date in data.index:
            backtest_strategy.on_new_date(date, (), Mock(), Mock())

        sma_paper = SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)
        paper_strategy = _SimpleStrategy()
        paper_strategy.add_indicator(sma_paper)
        paper_strategy.set_data(data)
        paper_strategy.warm_up_indicators()
        paper_strategy.on_new_date(data.index[-1], (), Mock(), Mock())

        bars_after_warmup = data.index[sma_backtest.warm_up_period - 1:]
        for bar in bars_after_warmup:
            backtest_val = sma_backtest.get_value_at(bar)
            paper_val = sma_paper.get_value_at(bar)
            assert backtest_val == pytest.approx(paper_val), (
                f"SMA mismatch at {bar}: backtest={backtest_val}, paper={paper_val}"
            )
