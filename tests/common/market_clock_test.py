"""Tests for the market clock: the engine's single time zone boundary."""

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from algo_trading_engine._internal.util.market_clock import (
    bars_through,
    from_epoch_ms,
    market_index,
    market_now,
    market_time,
    market_today,
    session_dates,
    to_epoch_ms,
    to_market_dates,
    to_market_index,
    to_market_time,
    to_system_time,
)

EASTERN = ZoneInfo("America/New_York")


def _bars(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"Close": range(len(index))}, index=index)


class TestCurrentTime:
    def test_market_now_is_naive_without_subsecond_precision(self):
        result = market_now()
        assert result.tzinfo is None
        assert result.microsecond == 0

    def test_market_now_is_eastern_wall_time(self):
        before = datetime.now(EASTERN).replace(tzinfo=None, microsecond=0)
        result = market_now()
        after = datetime.now(EASTERN).replace(tzinfo=None)
        assert before <= result <= after

    def test_market_today_midnight(self):
        with patch(
            "algo_trading_engine._internal.util.market_clock.market_now",
            return_value=datetime(2026, 11, 11, 14, 0, 0),
        ):
            assert market_today() == datetime(2026, 11, 11, 0, 0, 0)


class TestEpochMilliseconds:
    def test_from_epoch_ms_returns_naive_eastern(self):
        opening_bell = datetime(2021, 11, 19, 9, 30, tzinfo=EASTERN)
        result = from_epoch_ms(opening_bell.timestamp() * 1000)
        assert result == datetime(2021, 11, 19, 9, 30)
        assert result.tzinfo is None

    def test_from_epoch_ms_follows_daylight_saving(self):
        # 14:30 UTC is 09:30 EST in winter and 10:30 EDT in summer.
        winter = datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc)
        summer = datetime(2024, 7, 16, 14, 30, tzinfo=timezone.utc)
        assert from_epoch_ms(winter.timestamp() * 1000) == datetime(2024, 1, 16, 9, 30)
        assert from_epoch_ms(summer.timestamp() * 1000) == datetime(2024, 7, 16, 10, 30)

    def test_to_epoch_ms_reads_naive_datetime_as_eastern(self):
        expected = int(datetime(2021, 11, 19, 16, 0, tzinfo=EASTERN).timestamp() * 1000)
        assert to_epoch_ms(datetime(2021, 11, 19, 16, 0)) == expected

    def test_epoch_ms_round_trip(self):
        bar = datetime(2024, 7, 16, 15, 0)
        assert from_epoch_ms(to_epoch_ms(bar)) == bar

    def test_to_epoch_ms_rejects_tz_aware(self):
        with pytest.raises(ValueError, match="naive US/Eastern"):
            to_epoch_ms(datetime(2021, 11, 19, 16, 0, tzinfo=timezone.utc))


class TestStoredInstants:
    def test_to_market_time_converts_utc_instant(self):
        instant = datetime(2025, 12, 9, 1, 30, tzinfo=timezone.utc)
        result = to_market_time(instant)
        assert result == datetime(2025, 12, 8, 20, 30)
        assert result.tzinfo is None

    def test_to_market_time_rejects_naive(self):
        with pytest.raises(ValueError, match="tz-aware instant"):
            to_market_time(datetime(2025, 12, 9, 1, 30))


class TestSystemTime:
    def test_to_system_time_is_aware_local(self):
        result = to_system_time(datetime(2024, 1, 16, 9, 30))
        assert result.tzinfo is not None
        assert result.astimezone(EASTERN).replace(tzinfo=None) == datetime(2024, 1, 16, 9, 30)

    def test_to_system_time_rejects_tz_aware(self):
        with pytest.raises(ValueError, match="naive US/Eastern"):
            to_system_time(datetime(2024, 1, 16, 9, 30, tzinfo=timezone.utc))


class TestProviderIndex:
    def test_to_market_index_converts_intraday_bars_to_eastern(self):
        chicago = pd.DatetimeIndex(["2024-01-16 08:30", "2024-01-16 09:30"], tz="America/Chicago")
        result = to_market_index(chicago)
        assert result.tz is None
        assert list(result) == [pd.Timestamp("2024-01-16 09:30"), pd.Timestamp("2024-01-16 10:30")]

    def test_to_market_dates_keeps_the_session_date(self):
        # Daily bars are stamped at exchange-local midnight (^VIX trades in Chicago).
        # Converting the instant would move them to 01:00 and break joins on date.
        chicago = pd.DatetimeIndex(["2024-01-16", "2024-01-17"], tz="America/Chicago")
        result = to_market_dates(chicago)
        assert result.tz is None
        assert list(result) == [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]

    @pytest.mark.parametrize("convert", [to_market_index, to_market_dates])
    def test_rejects_naive_provider_index(self, convert):
        with pytest.raises(ValueError, match="must be tz-aware"):
            convert(pd.date_range("2024-01-16", periods=2, freq="D"))

    @pytest.mark.parametrize("convert", [to_market_index, to_market_dates])
    def test_rejects_non_datetime_provider_index(self, convert):
        with pytest.raises(TypeError, match="Expected a DatetimeIndex"):
            convert(pd.RangeIndex(2))


class TestMarketIndex:
    def test_returns_naive_datetime_index(self):
        index = pd.date_range("2024-01-16 09:30", periods=3, freq="h")
        assert market_index(_bars(index)) is index

    def test_accepts_series(self):
        index = pd.date_range("2024-01-16", periods=3, freq="D")
        assert market_index(pd.Series([1.0, 2.0, 3.0], index=index)) is index

    def test_rejects_non_datetime_index(self):
        with pytest.raises(TypeError, match="Expected a DatetimeIndex, got RangeIndex"):
            market_index(_bars(pd.RangeIndex(3)))

    def test_rejects_tz_aware_index(self):
        aware = pd.date_range("2024-01-16", periods=3, freq="D", tz="UTC")
        with pytest.raises(ValueError, match="naive US/Eastern index"):
            market_index(_bars(aware))


class TestMarketTime:
    def test_timestamp_becomes_plain_datetime(self):
        result = market_time(pd.Timestamp("2024-01-16 09:30"))
        assert result == datetime(2024, 1, 16, 9, 30)
        assert type(result) is datetime

    def test_datetime_is_returned_unchanged(self):
        value = datetime(2024, 1, 16, 9, 30)
        assert market_time(value) is value

    @pytest.mark.parametrize("value", [pd.NaT, "2024-01-16", 20240116, None])
    def test_rejects_values_that_are_not_datetimes(self, value):
        with pytest.raises(TypeError, match="Expected a datetime"):
            market_time(value)

    def test_rejects_tz_aware(self):
        with pytest.raises(ValueError, match="naive US/Eastern datetime"):
            market_time(pd.Timestamp("2024-01-16 09:30", tz="UTC"))


class TestBarSelection:
    def test_bars_through_is_inclusive(self):
        data = _bars(pd.date_range("2024-01-16 09:30", periods=4, freq="h"))
        result = bars_through(data, datetime(2024, 1, 16, 11, 30))
        assert list(result.index) == list(data.index[:3])

    def test_bars_through_rejects_non_datetime_index(self):
        with pytest.raises(TypeError, match="Expected a DatetimeIndex"):
            bars_through(_bars(pd.RangeIndex(3)), datetime(2024, 1, 16))

    def test_session_dates_puts_intraday_bars_on_one_day(self):
        index = pd.DatetimeIndex(["2024-01-16 09:30", "2024-01-16 15:30", "2024-01-17 09:30"])
        assert list(session_dates(index)) == [
            pd.Timestamp("2024-01-16"),
            pd.Timestamp("2024-01-16"),
            pd.Timestamp("2024-01-17"),
        ]
