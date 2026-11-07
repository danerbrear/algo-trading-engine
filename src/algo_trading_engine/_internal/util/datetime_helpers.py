"""Helper functions for formatting and manipulating datetime objects."""

from datetime import datetime

EXPIRATION_DATE_FORMAT = "%Y-%m-%d"

def to_readable_str(timestamp: datetime) -> str:
    """Returns a readable date time string"""
    return timestamp.strftime("%Y-%m-%d %H:%M")

def to_readable_expiration_str(timestamp: datetime) -> str:
    """Returns a readable date time string for a contract expiration date"""
    return timestamp.strftime("%Y-%m-%d")
