"""Unit tests for near-real-time option pricing wiring (make_rt_option_bar + Strategy.is_live)."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest

from algo_trading_engine.core.engine import make_rt_option_bar
from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.dto import OptionBarDTO
from algo_trading_engine.enums import BarTimeInterval


def _sample_bar(ticker: str = "O:SPY211119C00045000") -> OptionBarDTO:
    return OptionBarDTO(
        ticker=ticker,
        timestamp=datetime.now(),
        open_price=Decimal("1.0"),
        high_price=Decimal("1.1"),
        low_price=Decimal("0.9"),
        close_price=Decimal("1.05"),
        volume=10,
        volume_weighted_avg_price=Decimal("1.02"),
        number_of_transactions=1,
        adjusted=True,
    )


@pytest.fixture
def contract():
    return Mock(ticker="O:SPY211119C00045000")


class _MinimalStrategy(Strategy):
    """Concrete Strategy for exercising base-class helpers."""

    def on_new_date(self, date, positions, add_position, remove_position):
        return None

    def on_end(self, positions, remove_position):
        return None

    def validate_data(self, data: pd.DataFrame) -> bool:
        return True


class TestMakeRtOptionBar:
    def test_returns_snapshot_result(self, contract):
        handler = Mock()
        handler.get_option_snapshot = Mock(return_value=_sample_bar())
        handler.get_option_bar = Mock()

        rt = make_rt_option_bar(handler)
        bar = rt(contract)

        assert bar is not None
        handler.get_option_snapshot.assert_called_once_with(contract)
        handler.get_option_bar.assert_not_called()

    def test_returns_none_without_aggs_fallback(self, contract):
        handler = Mock()
        handler.get_option_snapshot = Mock(return_value=None)
        handler.get_option_bar = Mock(return_value=_sample_bar())

        rt = make_rt_option_bar(handler)
        bar = rt(contract)

        assert bar is None
        handler.get_option_snapshot.assert_called_once_with(contract)
        handler.get_option_bar.assert_not_called()


class TestStrategyIsLive:
    def test_is_live_false_by_default(self):
        strategy = _MinimalStrategy()
        assert strategy.is_live is False

    def test_is_live_true_when_rt_set(self):
        strategy = _MinimalStrategy()
        strategy.get_rt_option_bar = lambda contract: _sample_bar()
        assert strategy.is_live is True


class TestStrategyGetCurrentOptionBar:
    def test_live_uses_rt_snapshot(self, contract):
        strategy = _MinimalStrategy()
        rt_bar = _sample_bar()
        strategy.get_rt_option_bar = Mock(return_value=rt_bar)
        strategy.get_option_bar = Mock(return_value=_sample_bar("other"))

        result = strategy.get_current_option_bar(contract, datetime.now(), timespan=BarTimeInterval.HOUR)

        assert result is rt_bar
        strategy.get_rt_option_bar.assert_called_once_with(contract)
        strategy.get_option_bar.assert_not_called()

    def test_backtest_uses_historical_aggs(self, contract):
        strategy = _MinimalStrategy()
        hist_bar = _sample_bar()
        strategy.get_rt_option_bar = None
        strategy.get_option_bar = Mock(return_value=hist_bar)
        date = datetime(2025, 1, 2)

        result = strategy.get_current_option_bar(contract, date, timespan=BarTimeInterval.DAY)

        assert result is hist_bar
        strategy.get_option_bar.assert_called_once_with(contract, date, timespan=BarTimeInterval.DAY)
