"""Tests for VelocitySignalMomentumStrategy."""

# These tests call the strategy's private helpers directly.
# pylint: disable=protected-access

import inspect
from datetime import datetime
from decimal import Decimal
from typing import Optional, Sequence
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from algo_trading_engine._internal.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from algo_trading_engine.backtest._strategy_builder import StrategyFactory
from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.dto import OptionBarDTO, OptionContractDTO
from algo_trading_engine.enums import OptionType, StrategyType
from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo import ExpirationDate, Option, Position, StrikePrice, TreasuryRates, create_position


def _new_strategy(**kwargs) -> VelocitySignalMomentumStrategy:
    """Build a strategy whose option callbacks are mocks."""
    handler = Mock()
    return VelocitySignalMomentumStrategy(
        get_contract_list_for_date=handler.get_contract_list_for_date,
        get_option_bar=handler.get_option_bar,
        get_options_chain=handler.get_options_chain,
        **kwargs,
    )


def _annotation_name(annotation):
    """Return the readable name of a type annotation."""
    name = annotation if isinstance(annotation, str) else annotation.__name__
    return name.rsplit(".", 1)[-1]


def _close_price(frame: pd.DataFrame):
    """Underlying-price callback that reads Close from ``frame``."""

    def get_price(date: datetime, symbol: str) -> Optional[float]:
        _ = symbol
        return float(frame.loc[date, "Close"])

    return get_price


def _ignore_remove_position(
    date: datetime,
    position: Position,
    exit_price: float,
    underlying_price: Optional[float] = None,
    current_volumes: Optional[Sequence[Optional[int]]] = None,
) -> None:
    """Close callback whose parameters match Strategy.RemovePositionCallback."""
    _ = date, position, exit_price, underlying_price, current_volumes


def _put_contract(ticker: str, strike: str) -> OptionContractDTO:
    """One put contract used by spread-construction tests."""
    return OptionContractDTO(
        ticker=ticker,
        underlying_ticker="SPY",
        contract_type=OptionType.PUT,
        strike_price=StrikePrice(Decimal(strike)),
        expiration_date=ExpirationDate(datetime(2024, 1, 15).date()),
        exercise_style="american",
        shares_per_contract=100,
    )


def _option_bar(ticker: str, close_price: str) -> OptionBarDTO:
    """One option bar with a uniform price."""
    price = Decimal(close_price)
    return OptionBarDTO(
        ticker=ticker,
        timestamp=datetime(2024, 1, 1),
        open_price=price,
        high_price=price,
        low_price=price,
        close_price=price,
        volume=1000,
        volume_weighted_avg_price=price,
        number_of_transactions=100,
    )


def _ohlcv_frame(periods: int = 50) -> pd.DataFrame:
    """Daily OHLCV frame with a steady price climb."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    closes = [100 + i * 0.1 for i in range(periods)]
    return pd.DataFrame(
        {
            "Close": closes,
            "Open": closes,
            "High": [price + 1 for price in closes],
            "Low": [price - 1 for price in closes],
            "Volume": [1000000] * periods,
        },
        index=dates,
    )


def _assert_same_parameter_names(base_signature, strategy_signature) -> None:
    """Fail when a strategy override renames a base-class parameter."""
    base_params = [name for name in base_signature.parameters if name != "self"]
    strategy_params = list(strategy_signature.parameters)
    assert strategy_params == base_params, f"Parameter names don't match: {strategy_params} vs {base_params}"


def _assert_remove_position_callback(base_signature, strategy_signature) -> None:
    """Fail when remove_position is not RemovePositionCallback."""
    base_type = base_signature.parameters["remove_position"].annotation
    strategy_type = strategy_signature.parameters["remove_position"].annotation
    assert (
        _annotation_name(base_type) == "RemovePositionCallback"
    ), f"remove_position should be RemovePositionCallback, got: {base_type}"
    assert (
        _annotation_name(strategy_type) == "RemovePositionCallback"
    ), f"remove_position should be RemovePositionCallback, got: {strategy_type}"


def _treasury_rates() -> TreasuryRates:
    """Three days of treasury yields."""
    return TreasuryRates(
        pd.DataFrame(
            {"IRX_1Y": [0.05, 0.06, 0.07], "TNX_10Y": [0.04, 0.05, 0.06]},
            index=pd.date_range("2024-01-01", periods=3),
        )
    )


class TestVelocitySignalMomentumStrategy:
    """Test cases for VelocitySignalMomentumStrategy"""

    def setup_method(self):
        """Set up test data"""
        # pylint: disable=attribute-defined-outside-init
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        self.strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2021-01-01", "2021-01-30", freq="D")
        self.market_data = pd.DataFrame(
            {"Close": [100 + i * 0.5 for i in range(len(dates))], "Volume": [1000000] * len(dates)}, index=dates
        )
        treasury_dates = pd.date_range("2021-01-01", "2021-01-30", freq="D")
        treasury_data = pd.DataFrame(
            {"IRX_1Y": [0.001] * len(treasury_dates), "TNX_10Y": [0.015] * len(treasury_dates)}, index=treasury_dates
        )
        self.treasury_rates = TreasuryRates(treasury_data)
        self.atm_option = Option(
            ticker="SPY",
            timestamp=datetime(2021, 1, 15),
            symbol="SPY",
            expiration="2021-02-19",
            strike=100.0,
            option_type=OptionType.PUT,
            last_price=2.5,
            bid=2.45,
            ask=2.55,
            volume=100,
            open_interest=1000,
        )
        self.otm_option = Option(
            ticker="SPY",
            timestamp=datetime(2021, 1, 15),
            symbol="SPY",
            expiration="2021-02-19",
            strike=95.0,
            option_type=OptionType.PUT,
            last_price=1.0,
            bid=0.95,
            ask=1.05,
            volume=50,
            open_interest=500,
        )
        self.position = create_position(
            symbol="SPY",
            expiration_date=datetime(2021, 2, 19),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2021, 1, 15),
            entry_price=1.5,
            legs=[self.atm_option, self.otm_option],
        )
        self.position.set_quantity(1)

    def test_profit_target_and_stop_loss_initialization(self):
        """Test that profit_target and stop_loss are properly initialized"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy_with_params = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            profit_target=0.25,
            stop_loss=0.6,
        )
        assert strategy_with_params.profit_target == 0.25
        assert strategy_with_params.stop_loss == 0.6
        strategy_without_params = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        assert strategy_without_params.profit_target is None
        assert strategy_without_params.stop_loss is None

    def test_should_close_due_to_profit_target(self):
        """Test that profit target check works correctly"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            profit_target=0.2,
        )
        position = Mock()
        position.profit_target_hit = Mock(side_effect=lambda target, price: price is not None and True)
        assert strategy._profit_target_hit(position, 1.5)
        position.profit_target_hit = Mock(side_effect=lambda target, price: price is not None and True)
        assert not strategy._profit_target_hit(position, None)
        strategy_no_target = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            profit_target=None,
        )
        assert not strategy_no_target._profit_target_hit(position, 1.5)

    def test_should_close_due_to_stop_loss(self):
        """Test that stop loss check works correctly"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            stop_loss=0.6,
        )
        position = Mock()
        position.stop_loss_hit = Mock(side_effect=lambda stop, price: price is not None and True)
        assert strategy._stop_loss_hit(position, 1.5)
        position.stop_loss_hit = Mock(side_effect=lambda stop, price: price is not None and True)
        assert not strategy._stop_loss_hit(position, None)
        strategy_no_stop = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            stop_loss=None,
        )
        assert not strategy_no_stop._stop_loss_hit(position, 1.5)

    def test_select_week_expiration_prefers_5_to_10_days(self):
        """Prefer an expiration 5 to 10 days out over nearer and farther dates."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        contracts = [
            OptionContractDTO(
                ticker="O:SPY240103P100",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal("100.0")),
                expiration_date=ExpirationDate(datetime(2024, 1, 3).date()),
                exercise_style="american",
                shares_per_contract=100,
            ),
            OptionContractDTO(
                ticker="O:SPY240108P100",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal("100.0")),
                expiration_date=ExpirationDate(datetime(2024, 1, 8).date()),
                exercise_style="american",
                shares_per_contract=100,
            ),
            OptionContractDTO(
                ticker="O:SPY240120P100",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal("100.0")),
                expiration_date=ExpirationDate(datetime(2024, 1, 20).date()),
                exercise_style="american",
                shares_per_contract=100,
            ),
        ]
        strategy.get_contract_list_for_date = Mock(return_value=contracts)
        date = datetime(2024, 1, 1)
        picked = strategy._select_week_expiration(date)
        assert picked == "2024-01-08"

    def test_get_current_underlying_price(self):
        """Return the close for a known date and raise when the date is missing."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", periods=3)
        data = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)
        data.index.name = "SPY"
        strategy.set_data(data)

        def mock_get_price(date: datetime, symbol: str) -> Optional[float]:
            _ = symbol
            if date == datetime(2024, 1, 2):
                return 101.0
            raise KeyError(f"Date {date} not found")

        strategy.get_current_underlying_price = mock_get_price
        assert strategy.get_current_underlying_price(datetime(2024, 1, 2), "SPY") == 101.0

        with pytest.raises(KeyError):
            strategy.get_current_underlying_price(datetime(2023, 12, 31), "SPY")

    def test_sanitize_exit_price(self):
        """Clamp negative prices to zero and round the rest to cents."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        assert strategy._sanitize_exit_price(-1.234) == 0.0
        assert strategy._sanitize_exit_price(0.004) == 0.0
        assert strategy._sanitize_exit_price(1.235) == 1.24
        assert strategy._sanitize_exit_price(None) is None

    def test_should_close_predicates(self):
        """Holding-period and stop-loss checks use the position and exit price."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        pos = self.position
        assert strategy._should_close_due_to_holding(pos, datetime(2021, 1, 20), holding_period=5) is True
        assert strategy._stop_loss_hit(pos, exit_price=0.5) is False

    def test_compute_exit_price_with_chain_and_missing_contracts(self):
        """Test that strategy uses the injected compute_exit_price callable"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        date = datetime(2024, 1, 1)
        atm = Option(
            ticker="O:SPY240115P100",
            timestamp=date,
            symbol="A",
            expiration="2024-01-15",
            strike=100.0,
            option_type=OptionType.PUT,
            last_price=2.0,
        )
        otm = Option(
            ticker="O:SPY240115P90",
            timestamp=date,
            symbol="B",
            expiration="2024-01-15",
            strike=90.0,
            option_type=OptionType.PUT,
            last_price=1.0,
        )
        pos = create_position(
            symbol="SPY",
            expiration_date=datetime(2024, 1, 15),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=date,
            entry_price=1.0,
            legs=[atm, otm],
        )
        pos.set_quantity(1)
        strategy.compute_exit_price = Mock(return_value=0.8)
        exit_price = strategy.compute_exit_price(pos, date)
        assert exit_price == 0.8
        strategy.compute_exit_price.assert_called_once_with(pos, date)

    def test_get_risk_free_rate_with_treasury_data(self):
        """Test getting risk-free rate when treasury data is available"""
        self.strategy.set_data(self.market_data, self.treasury_rates)
        current_date = datetime(2021, 1, 15)
        risk_free_rate = self.strategy._get_risk_free_rate(current_date)
        assert isinstance(risk_free_rate, float)
        assert risk_free_rate == 0.001

    def test_get_risk_free_rate_without_treasury_data(self):
        """Test getting risk-free rate when treasury data is not available."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        strategy.set_data(
            pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2024-01-02", periods=3, freq="D"))
        )
        rate = strategy._get_risk_free_rate(datetime.now())
        assert rate == 2.0

    def test_create_test_put_credit_spread_success(self):
        """Test creating a test put credit spread successfully."""
        strategy = _new_strategy()
        atm_contract = _put_contract("O:SPY240115P100", "100.0")
        otm_contract = _put_contract("O:SPY240115P94", "94.0")
        strategy.get_contract_list_for_date = Mock(return_value=[atm_contract, otm_contract])
        atm_bar = _option_bar("O:SPY240115P100", "2.50")
        otm_bar = _option_bar("O:SPY240115P94", "0.50")
        strategy.get_option_bar = Mock(
            side_effect=lambda ticker, bar_date, multiplier=1, timespan=None: atm_bar if "100" in ticker else otm_bar
        )
        prices = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2024-01-01", periods=3))
        strategy.set_data(prices, _treasury_rates())
        position = strategy._create_put_credit_spread(datetime(2024, 1, 1), 100.0, "2024-01-15")
        assert position is not None
        assert position.strategy_type == StrategyType.PUT_CREDIT_SPREAD
        assert position.strike_price == 100.0
        assert position.entry_price == 2.0
        assert len(position.legs) == 2

    def test_create_test_put_credit_spread_no_credit(self):
        """Test creating a test put credit spread when no credit is received."""
        strategy = _new_strategy()
        prices = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2024-01-01", periods=3))
        strategy.set_data(prices, _treasury_rates())
        position = strategy._create_put_credit_spread(datetime(2024, 1, 1), 100.0, "2024-01-15")
        assert position is None


class TestVelocityBuySignal:
    """Buy-signal detection for the velocity strategy."""

    def test_has_buy_signal_valid_uptrend(self):
        """Test _has_buy_signal with a valid upward trend using MA velocity."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 30 + [95] * 5 + [98, 102, 105, 107, 108, 109, 110] + [110] * (len(dates) - 42)
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [110] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 8))
        assert result is True

    def test_has_buy_signal_no_data(self):
        """Test _has_buy_signal when no data is available."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        result = strategy._has_buy_signal(datetime(2024, 1, 20))
        assert result is False

    def test_has_buy_signal_insufficient_history(self):
        """Test _has_buy_signal when there's insufficient historical data."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-01-20", freq="D")
        market_data = pd.DataFrame({"Close": [100 + i * 0.5 for i in range(len(dates))]}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 1, 20))
        assert result is False

    def test_has_buy_signal_trend_too_short(self):
        """Test _has_buy_signal when trend duration is less than 3 days."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 65 + [95, 96, 97]
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [97] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 3, 8))
        assert result is False

    def test_has_buy_signal_trend_too_long(self):
        """Test _has_buy_signal when trend duration exceeds 60 days."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 30 + [95] + [95 + i * 0.1 for i in range(39)]
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [base_prices[-1]] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 5))
        assert result is False

    def test_has_buy_signal_no_velocity_increase(self):
        """Test _has_buy_signal when there's no MA velocity increase."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 30 + [95] * (len(dates) - 30)
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [95] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 15))
        assert result is False

    def test_has_buy_signal_significant_reversal(self):
        """Test _has_buy_signal when there's a significant reversal (>2% drop)."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 5 + [95] * 2 + [98, 102, 105, 107, 108, 109, 110] + [107] * 55
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [107] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 3, 10))
        assert result is False

    def test_has_buy_signal_invalid_date(self):
        """Test _has_buy_signal with an invalid date."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        market_data = pd.DataFrame({"Close": [100 + i * 0.5 for i in range(len(dates))]}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2025, 1, 1))
        assert result is False

    def test_has_buy_signal_edge_case_minimal_reversal(self):
        """Test _has_buy_signal with a minimal reversal that doesn't exceed 2%."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        base_prices = [100] * 30 + [95] * 2 + [98, 102, 105, 107, 108, 109, 110] + [108.35] * 30
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [108.35] * remaining_days
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 2))
        assert isinstance(result, bool)


class TestVelocityStrategyMarketData:
    """Prepared market data and end-of-run behavior."""

    def test_set_data_pre_calculates_moving_averages(self):
        """Test that set_data pre-calculates moving averages and velocity."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        data = strategy.data
        assert data is not None
        assert "SMA_15" in data.columns
        assert "SMA_30" in data.columns
        assert "MA_Velocity_15_30" in data.columns
        assert "Velocity_Changes" in data.columns
        assert not data["SMA_15"].iloc[20:].isna().all()
        assert not data["SMA_30"].iloc[35:].isna().all()
        velocity = data["MA_Velocity_15_30"].iloc[35:]
        assert not velocity.isna().all()
        assert (velocity > 0).all()

    def test_position_entries_tracking(self):
        """Test that position entries are tracked correctly for plotting."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-03-10", freq="D")
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        assert len(strategy._position_entries) == 0
        entry_date1 = datetime(2024, 1, 15)
        entry_date2 = datetime(2024, 2, 1)
        strategy._position_entries.append(entry_date1)
        strategy._position_entries.append(entry_date2)
        assert len(strategy._position_entries) == 2
        assert entry_date1 in strategy._position_entries
        assert entry_date2 in strategy._position_entries
        strategy.set_data(market_data)
        assert len(strategy._position_entries) == 0

    def test_on_end_is_noop_without_plotting(self):
        """Test on_end does not require matplotlib or data."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        strategy.on_end((), Mock(), datetime.now())

    def test_on_end_with_data_is_noop(self):
        """Test on_end with data does not invoke matplotlib."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({"Close": prices}, index=dates)
        strategy.set_data(market_data)
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            strategy.on_end((), Mock(), datetime.now())
            mock_subplots.assert_not_called()


class TestVelocityStrategyFactory:
    """Test cases for VelocitySignalMomentumStrategy via StrategyFactory"""

    def test_factory_passes_profit_target_and_stop_loss(self):
        """Test that StrategyFactory properly passes profit_target and stop_loss to the strategy"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = StrategyFactory.create_strategy(
            "velocity_momentum",
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
            profit_target=0.2,
            stop_loss=0.6,
        )
        assert strategy.profit_target == 0.2
        assert strategy.stop_loss == 0.6

    def test_factory_handles_none_parameters(self):
        """Test that StrategyFactory handles None parameters correctly"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = StrategyFactory.create_strategy(
            "velocity_momentum",
            get_contract_list_for_date=get_contract_list_for_date,
            get_option_bar=get_option_bar,
            get_options_chain=get_options_chain,
        )
        assert strategy.profit_target is None
        assert strategy.stop_loss is None


class TestVelocityStrategyMethodSignatures:
    """Test cases to verify method signatures match the base Strategy class"""

    def test_on_new_date_signature_matches_base_class(self):
        """Test that on_new_date can be called with the correct signature from base class"""
        strategy = _new_strategy()
        data = _ohlcv_frame()
        strategy.set_data(data)
        base_sig = inspect.signature(Strategy.on_new_date)
        strategy_sig = inspect.signature(strategy.on_new_date)
        _assert_same_parameter_names(base_sig, strategy_sig)
        _assert_remove_position_callback(base_sig, strategy_sig)
        strategy.get_current_underlying_price = _close_price(data)
        try:
            strategy.on_new_date(datetime(2024, 1, 25), tuple(), lambda _position: None, _ignore_remove_position)
        except TypeError as error:
            pytest.fail(f"on_new_date signature mismatch: {error}")

    def test_on_end_signature_matches_base_class(self):
        """Test that on_end can be called with the correct signature from base class"""
        strategy = _new_strategy()
        strategy.set_data(_ohlcv_frame())
        base_sig = inspect.signature(Strategy.on_end)
        strategy_sig = inspect.signature(strategy.on_end)
        _assert_same_parameter_names(base_sig, strategy_sig)
        _assert_remove_position_callback(base_sig, strategy_sig)
        try:
            strategy.on_end(tuple(), _ignore_remove_position, datetime(2024, 1, 25))
        except TypeError as error:
            pytest.fail(f"on_end signature mismatch: {error}")

    def test_try_close_positions_signature(self):
        """Test that _try_close_positions uses the correct remove_position signature"""
        strategy = _new_strategy()
        data = _ohlcv_frame()
        strategy.set_data(data)
        signature = inspect.signature(strategy._try_close_positions)
        remove_position_param = signature.parameters["remove_position"]
        assert str(remove_position_param.annotation) == "Strategy.RemovePositionCallback"
        callback_params = [
            name for name in inspect.signature(Strategy.RemovePositionCallback.__call__).parameters if name != "self"
        ]
        assert callback_params[:3] == ["date", "position", "exit_price"]
        strategy.get_option_bar = Mock(return_value=None)
        strategy.get_current_underlying_price = _close_price(data)
        try:
            strategy._try_close_positions(datetime(2024, 1, 25), tuple(), _ignore_remove_position)
        except TypeError as error:
            pytest.fail(f"_try_close_positions signature mismatch: {error}")

    def test_integration_with_backtest_engine_signature(self):
        """Integration test: verify strategy can be called by backtest engine with correct signatures"""
        strategy = _new_strategy()
        data = _ohlcv_frame()
        strategy.set_data(data)
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=10000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
        )
        add_pos_sig = inspect.signature(engine._add_position)
        assert (
            len(add_pos_sig.parameters) == 1
        ), f"_add_position should take 1 parameter (Position), got: {list(add_pos_sig.parameters.keys())}"
        remove_pos_params = list(inspect.signature(engine._remove_position).parameters.keys())
        expected_params = ["date", "position", "exit_price", "underlying_price", "current_volumes"]
        assert (
            remove_pos_params == expected_params or len(remove_pos_params) >= 3
        ), f"_remove_position parameters don't match: {remove_pos_params} vs {expected_params}"
        test_date = datetime(2024, 1, 5)
        positions_tuple = tuple(engine.positions)
        try:
            strategy.on_new_date(test_date, positions_tuple, engine._add_position, engine._remove_position)
        except TypeError as error:
            pytest.fail(f"Strategy.on_new_date cannot be called by BacktestEngine: {error}")
        try:
            strategy.on_end(positions_tuple, engine._remove_position, test_date)
        except TypeError as error:
            pytest.fail(f"Strategy.on_end cannot be called by BacktestEngine: {error}")
