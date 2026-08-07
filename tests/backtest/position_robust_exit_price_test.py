"""
Unit tests for robust spread exit price calculation.

Validates that spread exit prices are bounded to [0, width] and that garbage or
missing leg data is substituted with intrinsic values when the underlying is
deeply ITM or OTM.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from algo_trading_engine.common.models import Option, OptionChain, OptionType, StrategyType
from algo_trading_engine.dto import OptionBarDTO
from algo_trading_engine.vo import create_position


def _make_bar(ticker: str, close_price: float) -> OptionBarDTO:
    close_decimal = Decimal(str(close_price))
    return OptionBarDTO(
        ticker=ticker,
        timestamp=datetime(2025, 6, 17),
        open_price=close_decimal,
        high_price=close_decimal,
        low_price=close_decimal,
        close_price=close_decimal,
        volume=10,
        volume_weighted_avg_price=close_decimal,
        number_of_transactions=1,
    )


def _make_call_option(ticker: str, strike: float, last_price: float) -> Option:
    return Option(
        ticker=ticker,
        symbol="SPY",
        strike=strike,
        expiration="2025-06-30",
        option_type=OptionType.CALL,
        last_price=last_price,
    )


def _make_put_option(ticker: str, strike: float, last_price: float) -> Option:
    return Option(
        ticker=ticker,
        symbol="SPY",
        strike=strike,
        expiration="2025-06-30",
        option_type=OptionType.PUT,
        last_price=last_price,
    )


class TestRobustSpreadExitPrice:
    def test_valid_in_range_value_passes_through_call_debit(self):
        long_leg = _make_call_option("O:SPY250630C00100000", 100.0, 4.0)
        short_leg = _make_call_option("O:SPY250630C00105000", 105.0, 1.0)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=2.50,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 4.0),
            _make_bar(short_leg.ticker, 1.0),
            underlying_price=103.0,
        )

        assert exit_price == pytest.approx(3.0)

    def test_valid_in_range_value_passes_through_call_credit(self):
        short_leg = _make_call_option("O:SPY250630C00100000", 100.0, 4.0)
        long_leg = _make_call_option("O:SPY250630C00105000", 105.0, 1.0)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=2.50,
            spread_options=[short_leg, long_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(short_leg.ticker, 4.0),
            _make_bar(long_leg.ticker, 1.0),
            underlying_price=103.0,
        )

        assert exit_price == pytest.approx(3.0)

    def test_garbage_deep_itm_call_debit_returns_capped_width(self):
        long_leg = _make_call_option("O:SPY250630C00737000", 737.0, 16.31)
        short_leg = _make_call_option("O:SPY250630C00739000", 739.0, 19.54)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=737.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=0.88,
            spread_options=[long_leg, short_leg],
        )
        position.set_quantity(1)

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 16.31),
            _make_bar(short_leg.ticker, 19.54),
            underlying_price=756.0,
        )

        assert exit_price == pytest.approx(1.8)
        assert position.get_return_dollars(exit_price) == pytest.approx(92.0)

    def test_missing_bar_deep_otm_call_debit_returns_zero(self):
        long_leg = _make_call_option("O:SPY250630C00100000", 100.0, 3.0)
        short_leg = _make_call_option("O:SPY250630C00105000", 105.0, 0.5)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=2.50,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            None,
            None,
            underlying_price=88.0,
        )

        assert exit_price == pytest.approx(0.0)

    def test_garbage_near_money_returns_none(self):
        long_leg = _make_call_option("O:SPY250630C00100000", 100.0, 1.0)
        short_leg = _make_call_option("O:SPY250630C00105000", 105.0, 3.0)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=2.50,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 1.0),
            _make_bar(short_leg.ticker, 3.0),
            underlying_price=102.0,
        )

        assert exit_price is None

    def test_garbage_without_underlying_returns_none(self):
        long_leg = _make_call_option("O:SPY250630C00737000", 737.0, 16.31)
        short_leg = _make_call_option("O:SPY250630C00739000", 739.0, 19.54)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=737.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=0.88,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 16.31),
            _make_bar(short_leg.ticker, 19.54),
            underlying_price=None,
        )

        assert exit_price is None

    def test_put_spread_deep_itm_returns_capped_width(self):
        long_leg = _make_put_option("O:SPY250630P00100000", 100.0, 8.0)
        short_leg = _make_put_option("O:SPY250630P00095000", 95.0, 3.0)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=4.50,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 20.0),
            _make_bar(short_leg.ticker, 25.0),
            underlying_price=88.0,
        )

        assert exit_price == pytest.approx(4.5)

    def test_put_spread_deep_otm_returns_zero(self):
        long_leg = _make_put_option("O:SPY250630P00100000", 100.0, 1.0)
        short_leg = _make_put_option("O:SPY250630P00095000", 95.0, 0.5)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=0.50,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 10.0),
            _make_bar(short_leg.ticker, 12.0),
            underlying_price=110.0,
        )

        assert exit_price == pytest.approx(0.0)

    def test_calculate_exit_price_from_option_chain_deep_itm(self):
        long_leg = _make_call_option("O:SPY250630C00737000", 737.0, 16.31)
        short_leg = _make_call_option("O:SPY250630C00739000", 739.0, 19.54)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=737.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=0.88,
            spread_options=[long_leg, short_leg],
        )
        option_chain = OptionChain(calls=(long_leg, short_leg))

        exit_price = position.calculate_exit_price(option_chain, underlying_price=756.0)

        assert exit_price == pytest.approx(1.8)
