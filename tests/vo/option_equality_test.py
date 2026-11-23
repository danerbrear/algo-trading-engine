"""Option equality identifies the contract, not the quote time."""

from datetime import datetime

from algo_trading_engine.enums import OptionType
from algo_trading_engine.vo import Option


def _option(timestamp: datetime, strike: float = 100.0) -> Option:
    return Option(
        ticker="O:SPY251121P00680000",
        timestamp=timestamp,
        symbol="SPY",
        strike=strike,
        expiration="2025-11-21",
        option_type=OptionType.PUT,
        last_price=4.72,
    )


def test_equality_ignores_timestamp():
    opened = _option(datetime(2026, 11, 6, 15, 30, 50))
    refreshed = _option(datetime(2026, 11, 12, 14, 30, 38))
    assert opened == refreshed


def test_equality_requires_the_same_contract():
    opened = _option(datetime(2026, 11, 6, 15, 30, 50), strike=680.0)
    other_strike = _option(datetime(2026, 11, 6, 15, 30, 50), strike=685.0)
    assert opened != other_strike
