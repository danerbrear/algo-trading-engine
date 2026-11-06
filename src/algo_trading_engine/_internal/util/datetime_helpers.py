"""Helper functions for formatting and manipulating datetime objects."""

from datetime import datetime

def to_readable_str(timestamp: datetime) -> str:
    """Returns a readable date time string"""
    return timestamp.strftime("%Y-%m-%d %H:%M")

def to_readable_expiration_str(timestamp: datetime) -> str:
    """Returns a readable date time string for a contract expiration date"""
    return timestamp.strftime('%Y-%m-%d')