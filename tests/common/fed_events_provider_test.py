"""Tests for FedEventsProvider dataframe validation."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from algo_trading_engine._internal.common.fed_events_provider import FedEventsProvider

_TEST_CALENDAR_ID = "test-bls-calendar"


@pytest.fixture
def provider(monkeypatch):
    """Provider with credentials and a calendar id, independent of .env."""
    monkeypatch.setenv("BLS_CALENDAR_ID", _TEST_CALENDAR_ID)
    created = FedEventsProvider()
    created._creds = Mock()
    return created


class TestAddFedEventsDataIndex:
    """add_fed_events_data requires a DatetimeIndex before calling Calendar."""

    def test_rejects_non_datetime_index(self, provider):
        data = pd.DataFrame({"Close": [1.0]})

        with patch("algo_trading_engine._internal.common.fed_events_provider.build") as build:
            result, success = provider.add_fed_events_data(data)

        assert result is data
        assert success is False
        build.assert_not_called()

    def test_accepts_datetime_index(self, provider):
        data = pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
        service = Mock()
        service.events.return_value.list.return_value.execute.return_value = {"items": []}

        with patch(
            "algo_trading_engine._internal.common.fed_events_provider.build",
            return_value=service,
        ) as build:
            provider.add_fed_events_data(data)

        build.assert_called_once()

    def test_days_until_next_event_uses_event_timezone(self, provider):
        """A bar on an event date is 0 days until that event."""
        data = pd.DataFrame(
            {"Close": [1.0, 1.0, 1.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"]),
        )
        service = Mock()
        service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "summary": "Consumer Price Index",
                    "start": {"dateTime": "2024-01-03T15:00:00Z", "timeZone": "America/New_York"},
                },
                {
                    "summary": "Job Openings and Labor Turnover Survey",
                    "start": {"dateTime": "2024-01-05T15:00:00Z", "timeZone": "America/New_York"},
                },
                {
                    "summary": "Producer Price Index",
                    "start": {"dateTime": "2024-01-10T15:00:00Z", "timeZone": "America/New_York"},
                },
            ]
        }

        with patch("algo_trading_engine._internal.common.fed_events_provider.build", return_value=service):
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert list(result["Days Until Fed Event"]) == [2, 0, 6]
        assert list(result["Fed Event Description"]) == [
            "Consumer Price Index",
            "Consumer Price Index",
            "Producer Price Index",
        ]

    def test_follows_next_page_token(self, provider):
        data = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        service = Mock()
        service.events.return_value.list.return_value.execute.side_effect = [
            {
                "items": [
                    {
                        "summary": "Consumer Price Index",
                        "start": {"dateTime": "2024-01-02T15:00:00Z", "timeZone": "America/New_York"},
                    }
                ],
                "nextPageToken": "page-2",
            },
            {
                "items": [
                    {
                        "summary": "CPI Release",
                        "start": {"dateTime": "2024-01-05T15:00:00Z", "timeZone": "America/New_York"},
                    }
                ]
            },
        ]

        with patch("algo_trading_engine._internal.common.fed_events_provider.build", return_value=service):
            result, success = provider.add_fed_events_data(data)

        assert success is True
        assert int(result["Days Until Fed Event"].iloc[0]) == 1
        assert result["Fed Event Description"].iloc[0] == "Consumer Price Index"
        assert service.events.return_value.list.call_args_list[0].kwargs["calendarId"] == _TEST_CALENDAR_ID
        assert service.events.return_value.list.call_args_list[1].kwargs["pageToken"] == "page-2"
