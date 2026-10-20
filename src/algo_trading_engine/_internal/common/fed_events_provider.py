import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from algo_trading_engine.logging import get_logger

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
_FED_EVENT_SUMMARY_SUBSTRINGS = (
    "Consumer Price Index",
    "Producer Price Index"
)


def _event_start_date(event: dict) -> pd.Timestamp | None:
    """Calendar date of an event start, in the event's time zone."""
    start = event.get("start") or {}
    if start.get("date"):
        return pd.Timestamp(start["date"])

    raw_start = start.get("dateTime")
    if not raw_start:
        return None

    timestamp = pd.Timestamp(raw_start)
    if timestamp.tzinfo is not None:
        zone = start.get("timeZone")
        if zone:
            timestamp = timestamp.tz_convert(zone)
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _days_until_next_event(
    index: pd.DatetimeIndex, events: pd.DataFrame
) -> tuple[pd.Series, pd.Series] | None:
    """Days until the next event on or after each bar, and that event's summary."""
    if events.empty or "start" not in events.columns:
        return None

    scheduled = []
    for event in events.to_dict("records"):
        start = _event_start_date(event)
        if start is not None:
            scheduled.append((start, event.get("summary") or ""))
    scheduled.sort(key=lambda item: item[0])

    event_dates: list[pd.Timestamp] = []
    summaries: list[str] = []
    for start, summary in scheduled:
        if event_dates and start == event_dates[-1]:
            continue
        event_dates.append(start)
        summaries.append(summary)
    if not event_dates:
        return None

    dates = pd.DatetimeIndex(event_dates)
    bar_dates = pd.DatetimeIndex(index)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_localize(None)
    bar_dates = pd.DatetimeIndex(pd.Timestamp(value).normalize() for value in bar_dates)

    positions = dates.searchsorted(bar_dates, side="left")
    in_range = positions < len(dates)
    next_dates = dates.take(positions.clip(max=len(dates) - 1))
    days = pd.Series((next_dates - bar_dates).days, index=index, dtype="Int64")
    descriptions = pd.Series(
        pd.Index(summaries).take(positions.clip(max=len(summaries) - 1)),
        index=index,
        dtype="string",
    )
    return days.mask(~in_range), descriptions.mask(~in_range)


class FedEventsProvider:
    """Fetches and handles federal reserve events from the data source"""

    _creds: Credentials | None
    _bls_cal_id: str | None


    def __init__(self) -> None:
        self._creds = None

        bls_cal_id = os.getenv("BLS_CALENDAR_ID")
        if not bls_cal_id:
            get_logger().warning("Could not create FedEventsProvider. Missing BLS_CALENDAR_ID.")
            return
        self._bls_cal_id = bls_cal_id

        creds_path = os.getenv("GOOGLE_CREDS_PATH")
        if not creds_path:
            get_logger().warning("Could not create FedEventsProvider. Missing GOOGLE_CREDS_PATH.")
            return

        service_account_key = Path(creds_path)
        if not service_account_key.is_file():
            relative_key = Path(creds_path.lstrip("/"))
            if relative_key.is_file():
                service_account_key = relative_key
        if not service_account_key.is_file():
            get_logger().warning(
                f"Could not create FedEventsProvider. Missing service account key at {service_account_key}."
            )
            return

        with service_account_key.open(encoding="utf-8") as key_file:
            creds_info = json.load(key_file)
        self._creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)


    def add_fed_events_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Fetch federal reserve events and attach next event date and type.
        
        Arguments:
        - data (pd.DataFrame): A datetime-indexed dataframe

        Returns:
        - A tuple of:
            - The copy of the initial DataFrame plus the next fed event date and type
            - Successful - Will return original DataFrame when unsuccessful
        """

        if not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
            get_logger().warning(
                "Could not add fed events. Data must be a DataFrame with a DatetimeIndex."
            )
            return data, False

        if not self._creds:
            return data, False

        service = build("calendar", "v3", credentials=self._creds)

        events: list[dict] = []
        page_token = None
        try:
            while True:
                events_result = (
                    service.events()  # pylint: disable=no-member
                    .list(
                        calendarId=self._bls_cal_id,
                        singleEvents=True,
                        orderBy="startTime",
                        pageToken=page_token,
                    )
                    .execute()
                )
                events.extend(events_result.get("items") or [])
                page_token = events_result.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:
            get_logger().warning(f"Failed to fetch events from calendar {self._bls_cal_id}. {e}")
            return data, False

        # Filter for relevant events
        summary_pattern = re.compile(
            "|".join(re.escape(part) for part in _FED_EVENT_SUMMARY_SUBSTRINGS)
        )
        events[:] = [
            event for event in events
            if summary_pattern.search(event.get("summary") or "")
        ]

        result = data.copy()
        counted = _days_until_next_event(result.index, pd.DataFrame(events))
        if counted is None:
            get_logger().warning(f"Failed to fetch events from calendar {self._bls_cal_id}")
            return data, False
        days_until, descriptions = counted

        event_lines = []
        for event in events:
            start = _event_start_date(event)
            start_text = start.strftime("%Y-%m-%d") if start is not None else "unknown"
            event_lines.append(f"\t\t\t\t\tStart: {start_text} | {event.get('summary', '')} | Status: {event.get('status')}")
        get_logger().debug("Fed events:\n" + "\n".join(event_lines))

        get_logger().info(f"Fetched federal reserve events for {len(events)} calendar rows")
        result["Days Until Fed Event"] = days_until
        result["Fed Event Description"] = descriptions
        return result, True
