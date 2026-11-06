"""Tests for Option.get_black_scholes_implied_vol."""

from datetime import datetime

import pytest
from vollib.black_scholes import black_scholes

from algo_trading_engine.enums import OptionType
from algo_trading_engine.vo import Option

_UNDERLYING_PRICE = 100.0
_TREASURY_RATE = 0.01
_VOLATILITY = 0.20


def _option_priced_at_volatility(
    option_type: OptionType,
    timestamp: datetime,
    expiration: str,
    strike: float,
) -> Option:
    """Build an option whose last price is the Black-Scholes price at a known vol."""
    probe = Option(
        ticker="O:SPY",
        timestamp=timestamp,
        symbol="SPY",
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        last_price=1.0,
    )
    flag = "c" if option_type == OptionType.CALL else "p"
    price = black_scholes(
        flag,
        _UNDERLYING_PRICE,
        strike,
        probe.time_to_expiration,
        _TREASURY_RATE,
        _VOLATILITY,
    )
    return Option(
        ticker="O:SPY",
        timestamp=timestamp,
        symbol="SPY",
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        last_price=price,
        mid_price=price + 1.0,
    )


class TestGetBlackScholesImpliedVol:
    """Black-Scholes implied vol recovered from an option quote."""

    def test_call_recovers_volatility_from_last_price(self):
        """A one-year call priced at 20% vol should imply 20% vol from last price."""
        option = _option_priced_at_volatility(
            OptionType.CALL,
            datetime(2025, 1, 1),
            "2026-01-01",
            strike=100.0,
        )

        assert option.time_to_expiration == pytest.approx(1.0)
        implied = option.get_black_scholes_implied_vol(_UNDERLYING_PRICE, _TREASURY_RATE)

        assert implied == pytest.approx(_VOLATILITY, abs=1e-8)

    def test_put_recovers_volatility_using_year_fraction(self):
        """A put uses the put flag and calendar days to expiration divided by 365."""
        option = _option_priced_at_volatility(
            OptionType.PUT,
            datetime(2025, 1, 1),
            "2025-07-02",
            strike=110.0,
        )

        assert option.time_to_expiration == pytest.approx(182 / 365)
        implied = option.get_black_scholes_implied_vol(_UNDERLYING_PRICE, _TREASURY_RATE)

        assert implied == pytest.approx(_VOLATILITY, abs=1e-8)
