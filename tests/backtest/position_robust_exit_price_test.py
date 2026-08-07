"""
Unit tests for robust spread exit price calculation.

Validates that spread exit prices are bounded to [0, width] and that garbage or
missing leg data is substituted with intrinsic values when the underlying is
deeply ITM or OTM.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

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


class TestSpreadIntrinsicFloor:
    """Paths through _resolve_spread_value after spread-intrinsic validation."""

    def test_in_range_below_intrinsic_deep_itm_substitutes_capped_width_aug5_regression(self):
        """Path C: in-range leg marks below spread intrinsic, deep ITM -> 90% of width."""
        long_leg = _make_call_option("O:SPY260821C00739000", 739.0, 35.34)
        short_leg = _make_call_option("O:SPY260821C00741000", 741.0, 34.67)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2026, 8, 21),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=739.0,
            entry_date=datetime(2026, 7, 30),
            entry_price=1.15,
            spread_options=[long_leg, short_leg],
        )
        position.set_quantity(1)

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 35.34),
            _make_bar(short_leg.ticker, 34.67),
            underlying_price=771.29,
        )

        assert exit_price == pytest.approx(1.8)
        assert position.get_return_dollars(exit_price) == pytest.approx(65.0)

    def test_in_range_below_intrinsic_near_money_returns_none(self):
        """Path E: below intrinsic but not deep ITM/OTM -> cannot safely mark."""
        long_leg = _make_call_option("O:SPY250630C00100000", 100.0, 2.50)
        short_leg = _make_call_option("O:SPY250630C00105000", 105.0, 2.20)
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
            _make_bar(long_leg.ticker, 2.50),
            _make_bar(short_leg.ticker, 2.20),
            underlying_price=101.0,
        )

        assert exit_price is None

    def test_in_range_slightly_below_intrinsic_deep_itm_uses_capped_width_not_raw(self):
        """Path G: raw 1.70 vs intrinsic 2.00 deep ITM -> 1.80, not 1.70."""
        long_leg = _make_call_option("O:SPY260821C00739000", 739.0, 35.50)
        short_leg = _make_call_option("O:SPY260821C00741000", 741.0, 33.80)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2026, 8, 21),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=739.0,
            entry_date=datetime(2026, 7, 30),
            entry_price=1.15,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 35.50),
            _make_bar(short_leg.ticker, 33.80),
            underlying_price=771.0,
        )

        assert exit_price == pytest.approx(1.8)
        assert exit_price != pytest.approx(1.7)

    def test_in_range_at_intrinsic_deep_itm_passes_through_raw(self):
        """Path G': mark at intrinsic passes through without substitution."""
        long_leg = _make_call_option("O:SPY260821C00739000", 739.0, 35.50)
        short_leg = _make_call_option("O:SPY260821C00741000", 741.0, 33.50)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2026, 8, 21),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=739.0,
            entry_date=datetime(2026, 7, 30),
            entry_price=1.15,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 35.50),
            _make_bar(short_leg.ticker, 33.50),
            underlying_price=771.0,
        )

        assert exit_price == pytest.approx(2.0)

    def test_in_range_below_intrinsic_without_underlying_returns_raw_lenient(self):
        """Path H: no underlying -> accept in-range raw (lenient policy)."""
        long_leg = _make_call_option("O:SPY260821C00739000", 739.0, 35.34)
        short_leg = _make_call_option("O:SPY260821C00741000", 741.0, 34.67)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2026, 8, 21),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=739.0,
            entry_date=datetime(2026, 7, 30),
            entry_price=1.15,
            spread_options=[long_leg, short_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(long_leg.ticker, 35.34),
            _make_bar(short_leg.ticker, 34.67),
            underlying_price=None,
        )

        assert exit_price == pytest.approx(0.67)

    def test_put_debit_in_range_below_intrinsic_deep_itm_substitutes_capped_width(self):
        """Put analogue of Path C: below-intrinsic mark, deep ITM -> 90% of width."""
        long_leg = _make_put_option("O:SPY250630P00100000", 100.0, 13.50)
        short_leg = _make_put_option("O:SPY250630P00095000", 95.0, 12.00)
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
            _make_bar(long_leg.ticker, 13.50),
            _make_bar(short_leg.ticker, 12.00),
            underlying_price=88.0,
        )

        assert exit_price == pytest.approx(4.5)

    def test_call_credit_spread_shares_intrinsic_floor(self):
        """Credit spread uses same resolver; below-intrinsic deep ITM -> capped width."""
        short_leg = _make_call_option("O:SPY260821C00739000", 739.0, 35.34)
        long_leg = _make_call_option("O:SPY260821C00741000", 741.0, 34.67)
        position = create_position(
            symbol="SPY",
            expiration_date=datetime(2026, 8, 21),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=739.0,
            entry_date=datetime(2026, 7, 30),
            entry_price=0.50,
            spread_options=[short_leg, long_leg],
        )

        exit_price = position.calculate_exit_price_from_bars(
            _make_bar(short_leg.ticker, 35.34),
            _make_bar(long_leg.ticker, 34.67),
            underlying_price=771.29,
        )

        assert exit_price == pytest.approx(1.8)


class TestSpreadIntrinsicHelper:
    def _call_debit_position(self):
        long_leg = _make_call_option("O:SPY250630C00100000", 100.0, 4.0)
        short_leg = _make_call_option("O:SPY250630C00105000", 105.0, 1.0)
        return create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.CALL_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=2.50,
            spread_options=[long_leg, short_leg],
        )

    def _put_debit_position(self):
        long_leg = _make_put_option("O:SPY250630P00100000", 100.0, 8.0)
        short_leg = _make_put_option("O:SPY250630P00095000", 95.0, 3.0)
        return create_position(
            symbol="SPY",
            expiration_date=datetime(2025, 6, 30),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2025, 6, 10),
            entry_price=4.50,
            spread_options=[long_leg, short_leg],
        )

    def test_call_spread_intrinsic_below_low_strike_is_zero(self):
        position = self._call_debit_position()
        assert position._spread_intrinsic(98.0) == pytest.approx(0.0)

    def test_call_spread_intrinsic_between_strikes(self):
        position = self._call_debit_position()
        assert position._spread_intrinsic(102.0) == pytest.approx(2.0)

    def test_call_spread_intrinsic_above_high_strike_is_width(self):
        position = self._call_debit_position()
        assert position._spread_intrinsic(110.0) == pytest.approx(5.0)

    def test_put_spread_intrinsic_above_high_strike_is_zero(self):
        position = self._put_debit_position()
        assert position._spread_intrinsic(110.0) == pytest.approx(0.0)

    def test_put_spread_intrinsic_between_strikes(self):
        position = self._put_debit_position()
        # S=97: long 100P intrinsic 3, short 95P intrinsic 0
        assert position._spread_intrinsic(97.0) == pytest.approx(3.0)

    def test_put_spread_intrinsic_below_low_strike_is_width(self):
        position = self._put_debit_position()
        assert position._spread_intrinsic(88.0) == pytest.approx(5.0)

    def test_unsupported_option_type_raises(self):
        position = self._call_debit_position()
        bad_leg = Mock()
        bad_leg.strike = 100.0
        bad_leg.option_type = object()
        position.spread_options[0] = bad_leg
        with pytest.raises(ValueError, match="Unsupported option type"):
            position._spread_intrinsic(100.0)
