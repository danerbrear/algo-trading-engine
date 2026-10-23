"""Tests for FedEventsProvider and Finance Calendar integration."""

import calendar
import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from algo_trading_engine.fed_events_provider import (
    _USER_AGENT,
    FedEventsProvider,
    NextFedEvent,
    _http_get,
)

_FROZEN_TODAY = date(2024, 7, 15)
_SERIES = ("fomc", "cpi", "ppi")


def _six_months_before(today: date) -> date:
    month = today.month - 6
    year = today.year
    if month <= 0:
        month += 12
        year -= 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _history_windows(first_bar: date, last_bar: date, floor: date) -> list[tuple[date, date]]:
    if last_bar < floor:
        return []
    window_from = max(first_bar, floor)
    windows: list[tuple[date, date]] = []
    while window_from <= last_bar:
        end_month = window_from.month + 2
        end_year = window_from.year + (end_month - 1) // 12
        end_month = (end_month - 1) % 12 + 1
        end_day = min(window_from.day, calendar.monthrange(end_year, end_month)[1])
        window_to = date(end_year, end_month, end_day)
        windows.append((window_from, window_to))
        window_from = window_to + timedelta(days=1)
    return windows


@pytest.fixture
def provider():
    return FedEventsProvider()


def _history_payload(events: list[dict]) -> dict:
    return {"events": events, "attribution": "Finance Calendar"}


def _next_payload(series: str, event_date: str, name: str) -> dict:
    return {
        "series": series,
        "next": {"date": event_date, "name": name},
        "attribution": "Finance Calendar",
    }


class TestHttpGet:
    def test_sends_user_agent(self):
        response = MagicMock()
        response.read.return_value = json.dumps({"events": []}).encode()
        response.__enter__.return_value = response

        with patch(
            "algo_trading_engine.fed_events_provider.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = _http_get("history", {"series": "fomc"})

        assert result == {"events": []}
        request = urlopen.call_args.args[0]
        assert request.get_header("User-agent") == _USER_AGENT


class TestAddFedEventsDataIndex:
    """add_fed_events_data requires a DatetimeIndex before calling the API."""

    def test_rejects_non_datetime_index(self, provider):
        data = pd.DataFrame({"Close": [1.0]})

        with patch.object(provider, "_fetch_history") as fetch_history:
            result, success = provider.add_fed_events_data(data)

        assert result is data
        assert success is False
        fetch_history.assert_not_called()

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_accepts_datetime_index_with_empty_history(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        data = pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2024-06-01", periods=1))

        def fake_history(series, *, window_from, window_to):
            return _history_payload([])

        with patch.object(provider, "_fetch_history", side_effect=fake_history) as fetch_history:
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert pd.isna(result["Days Until Fed Event"].iloc[0])
        assert result["Fed Event Description"].iloc[0] is None
        assert fetch_history.call_count == len(_SERIES)

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_same_day_event_is_zero(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        data = pd.DataFrame(
            {"Close": [1.0, 1.0, 1.0]},
            index=pd.to_datetime(["2024-06-01", "2024-06-03", "2024-06-04"]),
        )

        def fake_history(series, *, window_from, window_to):
            if series == "cpi":
                return _history_payload(
                    [
                        {"date": "2024-06-03", "name": "US CPI"},
                        {"date": "2024-06-10", "name": "US PPI"},
                    ]
                )
            if series == "ppi":
                return _history_payload([{"date": "2024-06-10", "name": "US PPI"}])
            return _history_payload([])

        with patch.object(provider, "_fetch_history", side_effect=fake_history):
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert list(result["Days Until Fed Event"]) == [2, 0, 6]
        assert list(result["Fed Event Description"]) == ["US CPI", "US CPI", "US PPI"]

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_no_event_within_two_months_is_none(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        data = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-06-01"]))

        with patch.object(
            provider,
            "_fetch_history",
            return_value=_history_payload([{"date": "2024-09-01", "name": "US CPI"}]),
        ):
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert pd.isna(result["Days Until Fed Event"].iloc[0])
        assert result["Fed Event Description"].iloc[0] is None

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_hourly_bars_share_daily_values(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        index = pd.to_datetime(
            ["2024-06-03 09:30:00", "2024-06-03 10:30:00", "2024-06-04 09:30:00"]
        )
        data = pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=index)

        def fake_history(series, *, window_from, window_to):
            if series == "cpi":
                return _history_payload([{"date": "2024-06-03", "name": "US CPI"}])
            return _history_payload([])

        with patch.object(provider, "_fetch_history", side_effect=fake_history) as fetch_history:
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert list(result["Days Until Fed Event"]) == [0, 0, pd.NA]
        assert list(result["Fed Event Description"]) == ["US CPI", "US CPI", None]
        history_from_dates = {
            call.kwargs["window_from"] for call in fetch_history.call_args_list
        }
        assert history_from_dates == {date(2024, 6, 3)}

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_history_from_clamped_to_six_month_floor(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        floor = _six_months_before(_FROZEN_TODAY)
        data = pd.DataFrame(
            {"Close": [1.0, 1.0]},
            index=pd.to_datetime(["2023-01-01", "2024-06-01"]),
        )

        with patch.object(provider, "_fetch_history", return_value=_history_payload([])) as fetch_history:
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert pd.isna(result["Days Until Fed Event"].iloc[0])
        assert pd.isna(result["Days Until Fed Event"].iloc[1])
        fetch_history.assert_called()
        first_window_from = fetch_history.call_args_list[0].kwargs["window_from"]
        assert first_window_from == floor

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_history_to_is_two_months_after_from(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        data = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-06-01"]))

        with patch.object(provider, "_fetch_history", return_value=_history_payload([])) as fetch_history:
            provider.add_fed_events_data(data)

        window_to = fetch_history.call_args.kwargs["window_to"]
        assert window_to == date(2024, 8, 1)

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_long_frame_steps_history_windows(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        floor = _six_months_before(_FROZEN_TODAY)
        data_long = pd.DataFrame(
            {"Close": [1.0, 1.0]},
            index=pd.to_datetime(["2024-01-15", "2024-04-20"]),
        )
        windows = _history_windows(date(2024, 1, 15), date(2024, 4, 20), floor)
        assert len(windows) >= 2
        assert windows[1][0] == windows[0][1] + timedelta(days=1)

        with patch.object(provider, "_fetch_history", return_value=_history_payload([])) as fetch_history:
            provider.add_fed_events_data(data_long)

        assert fetch_history.call_count == len(_SERIES) * len(windows)

    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_all_history_failures_return_original(self, mock_date, provider):
        mock_date.today.return_value = _FROZEN_TODAY
        data = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-06-01"]))

        with patch.object(provider, "_fetch_history", return_value=None):
            result, success = provider.add_fed_events_data(data)

        assert result is data
        assert success is False


class TestGetNextFedEvent:
    @patch("algo_trading_engine.fed_events_provider.date", side_effect=date)
    def test_returns_soonest_series(self, mock_date, provider):
        mock_date.today.return_value = date(2024, 7, 1)
        with patch.object(
            provider,
            "_fetch_next",
            side_effect=[
                _next_payload("fomc", "2024-08-01", "FOMC"),
                _next_payload("cpi", "2024-07-10", "US CPI"),
                None,
            ],
        ):
            event = provider.get_next_fed_event()

        assert event == NextFedEvent(days_until_event=9, event_desc="US CPI")

    def test_returns_none_when_all_fail(self, provider):
        with patch.object(provider, "_fetch_next", return_value=None):
            assert provider.get_next_fed_event() is None
