from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import Enum


class BarTimeInterval(Enum):
    """Enum for bar time intervals."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"

    def bar_start(self, timestamp: datetime) -> datetime:
        """Start of the bar containing *timestamp* (US equity session, naive Eastern)."""
        if self == BarTimeInterval.DAY:
            return datetime(timestamp.year, timestamp.month, timestamp.day)

        if self == BarTimeInterval.HOUR:
            session_open = time(9, 30)
            day_start = datetime(timestamp.year, timestamp.month, timestamp.day)
            if timestamp.time() < session_open:
                return day_start.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes = (timestamp.hour * 60 + timestamp.minute) - (9 * 60 + 30)
            if minutes < 0:
                return day_start.replace(hour=9, minute=30, second=0, microsecond=0)
            hour_slots = minutes // 60
            return day_start.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(
                hours=hour_slots
            )

        if self == BarTimeInterval.MINUTE:
            return timestamp.replace(second=0, microsecond=0)

        raise ValueError(f"Unsupported bar interval: {self}")
