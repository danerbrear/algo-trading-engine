"""US equity session clock aligned with naive Eastern bar timestamps."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_US_EASTERN = ZoneInfo("America/New_York")


def market_now() -> datetime:
    """Current time as naive US/Eastern (matches yfinance bar index after tz strip)."""
    now = datetime.now(_US_EASTERN).replace(tzinfo=None, microsecond=0)
    return now


def market_today() -> datetime:
    """Today's calendar date at midnight US/Eastern, naive."""
    n = market_now()
    return datetime(n.year, n.month, n.day)
