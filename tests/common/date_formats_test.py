"""Tests for DateFormats."""

from datetime import datetime

from algo_trading_engine._internal.util.date_formats import DateFormats


def test_date_only_attributes_share_pattern():
    assert DateFormats.EXPIRATION == DateFormats.ISO_DATE == DateFormats.CACHE_DATE == "%Y-%m-%d"


def test_strftime_accepts_all_patterns():
    dt = datetime(2024, 1, 2, 9, 30)
    assert dt.strftime(DateFormats.EXPIRATION) == "2024-01-02"
    assert dt.strftime(DateFormats.ISO_DATE) == "2024-01-02"
    assert dt.strftime(DateFormats.CACHE_DATE) == "2024-01-02"
    assert dt.strftime(DateFormats.PRINTABLE) == "2024-01-02 09:30"
    assert dt.strftime(DateFormats.CACHE_TIME) == "0930"
    assert dt.strftime(DateFormats.COMPACT_DATE) == "20240102"
