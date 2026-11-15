"""Market time: the one place that converts time zones.

Convention: every datetime, every bar index, and every ``data_cache`` entry in
the engine is **naive US/Eastern** (``America/New_York`` wall time, ``tzinfo``
of ``None``). Time zones are converted only here, at the edges:

- provider -> engine: ``to_market_index``, ``to_market_dates``, ``from_epoch_ms``
- stored instant -> engine: ``to_market_time`` (decision records are UTC instants)
- engine -> provider: ``to_epoch_ms``
- engine -> display: ``to_system_time`` (local wall clock; never write this back)
- current time: ``market_now``, ``market_today`` (never ``datetime.now()`` or
  ``date.today()``, which read the system time zone)

Inside the engine, read a frame's index through ``market_index``,
``market_time``, ``bars_through``, and ``session_dates``. They check the
convention at runtime and return types pyright can follow, which pandas'
dynamically attached ``DatetimeIndex`` members do not.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def market_now() -> datetime:
    """Current time as naive US/Eastern, without sub-second precision."""
    return datetime.now(MARKET_TZ).replace(tzinfo=None, microsecond=0)


def market_today() -> datetime:
    """Today's calendar date at midnight US/Eastern, naive."""
    now = market_now()
    return datetime(now.year, now.month, now.day)


def from_epoch_ms(epoch_ms: float) -> datetime:
    """Provider epoch milliseconds -> naive Eastern."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=MARKET_TZ).replace(tzinfo=None)


def to_market_time(instant: datetime) -> datetime:
    """A tz-aware instant (e.g. a UTC decision timestamp) as naive Eastern."""
    if instant.tzinfo is None:
        raise ValueError("Expected a tz-aware instant; naive datetimes are already US/Eastern")
    return instant.astimezone(MARKET_TZ).replace(tzinfo=None)


def to_epoch_ms(when: datetime) -> int:
    """Naive Eastern -> provider epoch milliseconds."""
    return int(market_time(when).replace(tzinfo=MARKET_TZ).timestamp() * 1000)


def to_system_time(when: datetime) -> datetime:
    """Naive Eastern -> aware datetime in the local system (wall) time zone.

    For display only. Never feed the result back into the engine or ``data_cache``.
    """
    return market_time(when).replace(tzinfo=MARKET_TZ).astimezone()


def to_market_index(index: pd.Index) -> pd.DatetimeIndex:
    """Intraday provider index (tz-aware instants) -> naive Eastern."""
    return _require_aware(index).tz_convert(MARKET_TZ).tz_localize(None)


def to_market_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Daily provider index (tz-aware) -> naive midnight of each session date.

    A daily bar is labelled with its session date at the exchange's local
    midnight. It is a calendar date, not an instant, so the date is kept as-is
    rather than shifted into Eastern.
    """
    return session_dates(_require_aware(index).tz_localize(None))


def market_index(data: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """Return ``data.index`` as a naive Eastern ``DatetimeIndex``, or raise."""
    index = _require_datetime_index(data.index)
    if index.tz is not None:
        raise ValueError(f"Expected a naive US/Eastern index, got tz={index.tz}")
    return index


def market_time(value: object) -> datetime:
    """Return an index label or timestamp as a naive Eastern ``datetime``, or raise."""
    # NaT is a datetime subclass; reject it before the isinstance branches below.
    if value is pd.NaT or (isinstance(value, pd.Timestamp) and pd.isna(value)):
        raise TypeError(f"Expected a datetime, got {value!r}")
    # Check Timestamp before datetime: Timestamp is a datetime subclass, and pyright
    # only exposes ``to_pydatetime`` on the Timestamp type.
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            raise ValueError(f"Expected a naive US/Eastern datetime, got tzinfo={value.tzinfo}")
        return value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise ValueError(f"Expected a naive US/Eastern datetime, got tzinfo={value.tzinfo}")
        return value
    raise TypeError(f"Expected a datetime, got {value!r}")


def bars_through(data: pd.DataFrame, when: datetime) -> pd.DataFrame:
    """Bars at or before ``when`` (naive Eastern)."""
    return data.loc[market_index(data) <= when]


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive midnight of each bar's calendar date, so intraday bars share a day."""
    # pandas attaches ``normalize`` to DatetimeIndex dynamically, so pyright cannot see it.
    return index.normalize()  # pyright: ignore[reportAttributeAccessIssue]


def _require_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(index).__name__}")
    return index


def _require_aware(index: pd.Index) -> pd.DatetimeIndex:
    datetime_index = _require_datetime_index(index)
    if datetime_index.tz is None:
        raise ValueError("Provider index must be tz-aware so it can be converted to US/Eastern")
    return datetime_index
