"""Federal reserve macro events via Finance Calendar (https://www.financecalendar.com/api/)."""

from __future__ import annotations

import calendar
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from algo_trading_engine.logging import get_logger

_BASE_URL = "https://www.financecalendar.com/wp-json/fc/v1"
_FED_EVENT_SERIES = ("fomc", "cpi", "ppi")
_ATTRIBUTION_URL = "https://www.financecalendar.com"
_DAYS_COLUMN = "Days Until Fed Event"
_DESCRIPTION_COLUMN = "Fed Event Description"
_HISTORY_LIMIT = "500"
_HTTP_TIMEOUT_SECONDS = 30
_HISTORY_FLOOR_MONTHS = 6
_HORIZON_MONTHS = 2
_USER_AGENT = "algo-trading-engine (+https://www.financecalendar.com)"


@dataclass(frozen=True)
class NextFedEvent:
    """Soonest upcoming release among fomc, cpi, and ppi."""

    days_until_event: int
    event_desc: str


def _http_get(path: str, params: dict[str, str]) -> dict | None:
    query = urllib.parse.urlencode(params)
    url = f"{_BASE_URL}/{path.lstrip('/')}?{query}"
    # Cloudflare rejects Python-urllib's default User-Agent with HTTP 403 (error 1010).
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            body = response.read().decode()
            return json.loads(body)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        get_logger().warning(f"Finance Calendar request failed ({error.code}): {url}")
        return None
    except Exception as error:  # pylint: disable=broad-exception-caught
        get_logger().warning(f"Finance Calendar request failed: {url} ({error})")
        return None


def _shift_months(start: date, months: int) -> date:
    """Same calendar day shifted by `months`, clamped to the end of that month."""

    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _calendar_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive midnight timestamps for each bar, so intraday rows share a calendar day."""

    bar_dates = index.tz_localize(None) if index.tz is not None else index
    return pd.DatetimeIndex(pd.Timestamp(value).normalize() for value in bar_dates)


def _history_windows(first_bar: date, last_bar: date, floor: date) -> list[tuple[date, date]]:
    """Two-month history ranges covering bars on or after `floor`.

    Each window ends two months after it starts, which may fall after `last_bar`.
    """

    if last_bar < floor:
        return []

    windows: list[tuple[date, date]] = []
    window_from = max(first_bar, floor)
    while window_from <= last_bar:
        window_to = _shift_months(window_from, _HORIZON_MONTHS)
        windows.append((window_from, window_to))
        window_from = window_to + timedelta(days=1)
    return windows


def _event_description(event: dict) -> str:
    name = event.get("name")
    if name:
        return str(name)
    title = event.get("title")
    return str(title) if title else ""


def _events_from_payload(payload: dict) -> list[tuple[date, str]]:
    events: list[tuple[date, str]] = []
    for event in payload.get("events") or []:
        raw_date = event.get("date")
        if not raw_date:
            continue
        events.append((pd.Timestamp(raw_date).date(), _event_description(event)))
    return events


def _with_blank_fed_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result[_DAYS_COLUMN] = pd.NA
    result[_DESCRIPTION_COLUMN] = None
    return result


def _next_event_for_bar(
    bar_date: date,
    events: list[tuple[date, str]],
    floor: date,
) -> tuple[int | None, str | None]:
    """Days until the soonest release on or after `bar_date`, within the horizon."""

    if bar_date < floor:
        return None, None

    horizon_end = _shift_months(bar_date, _HORIZON_MONTHS)
    upcoming = [
        (event_date, description)
        for event_date, description in events
        if bar_date <= event_date <= horizon_end
    ]
    if not upcoming:
        return None, None
    event_date, description = min(upcoming, key=lambda release: release[0])
    return (event_date - bar_date).days, description


def _days_and_descriptions(
    calendar_dates: pd.DatetimeIndex,
    events: list[tuple[date, str]],
    floor: date,
) -> tuple[list[int | None], list[str | None]]:
    by_date = {
        normalized: _next_event_for_bar(normalized.date(), events, floor)
        for normalized in calendar_dates.unique()
    }
    days: list[int | None] = []
    descriptions: list[str | None] = []
    for normalized in calendar_dates:
        day_count, description = by_date[normalized]
        days.append(day_count)
        descriptions.append(description)
    return days, descriptions


class FedEventsProvider:
    """Fetches macro release dates from Finance Calendar."""

    def __init__(self) -> None:
        self._logged_attribution = False

    def _log_attribution_once(self, payload: dict | None) -> None:
        if self._logged_attribution or not payload:
            return
        attribution = payload.get("attribution")
        if attribution:
            get_logger().info(f"{attribution} ({_ATTRIBUTION_URL})")
        else:
            get_logger().info(f"Macro calendar data from {_ATTRIBUTION_URL}")
        self._logged_attribution = True

    def _fetch_history(
        self,
        series: str,
        *,
        window_from: date,
        window_to: date,
    ) -> dict | None:
        return _http_get(
            "history",
            {
                "series": series,
                "from": window_from.isoformat(),
                "to": window_to.isoformat(),
                "limit": _HISTORY_LIMIT,
            },
        )

    def _fetch_next(self, series: str) -> dict | None:
        return _http_get("next", {"series": series})

    def _collect_history_events(
        self,
        windows: list[tuple[date, date]],
    ) -> tuple[list[tuple[date, str]], bool]:
        events: list[tuple[date, str]] = []
        fetched = False
        for window_from, window_to in windows:
            for series in _FED_EVENT_SERIES:
                payload = self._fetch_history(
                    series,
                    window_from=window_from,
                    window_to=window_to,
                )
                if payload is None:
                    continue
                fetched = True
                self._log_attribution_once(payload)
                events.extend(_events_from_payload(payload))
        return events, fetched

    def add_fed_events_data(  # pylint: disable=too-many-return-statements
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, bool]:
        """Attach days-until and description columns using Finance Calendar /history."""

        if not isinstance(data, pd.DataFrame) or not isinstance(data.index, pd.DatetimeIndex):
            get_logger().warning(
                "Could not add fed events. Data must be a DataFrame with a DatetimeIndex."
            )
            return data, False

        calendar_dates = _calendar_dates(data.index)
        if calendar_dates.empty:
            return data.copy(), True

        floor = _shift_months(date.today(), -_HISTORY_FLOOR_MONTHS)
        windows = _history_windows(
            calendar_dates.min().date(),
            calendar_dates.max().date(),
            floor,
        )
        if not windows:
            return _with_blank_fed_columns(data), True

        events, fetched = self._collect_history_events(windows)
        if not fetched:
            return data, False

        days_values, description_values = _days_and_descriptions(calendar_dates, events, floor)
        result = data.copy()
        result[_DAYS_COLUMN] = pd.Series(days_values, index=data.index, dtype="Int64")
        result[_DESCRIPTION_COLUMN] = pd.Series(
            description_values, index=data.index, dtype="object"
        )

        get_logger().info(
            f"Attached fed event columns using {len(events)} history releases "
            f"across {len(windows)} two-month window(s)"
        )
        return result, True

    def get_next_fed_event(self) -> NextFedEvent | None:
        """Return the soonest upcoming fomc, cpi, or ppi release via /next."""

        today = date.today()
        candidates: list[NextFedEvent] = []
        for series in _FED_EVENT_SERIES:
            payload = self._fetch_next(series)
            if not payload:
                continue
            self._log_attribution_once(payload)
            next_event = payload.get("next")
            if not isinstance(next_event, dict):
                continue
            raw_date = next_event.get("date")
            if not raw_date:
                continue
            event_date = pd.Timestamp(raw_date).date()
            candidates.append(
                NextFedEvent(
                    days_until_event=(event_date - today).days,
                    event_desc=_event_description(next_event),
                )
            )

        if not candidates:
            return None
        return min(candidates, key=lambda item: item.days_until_event)
