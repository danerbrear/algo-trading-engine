"""Unit tests for OptionsHandler snapshot-based live option pricing."""

import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from algo_trading_engine.common.options_handler import OptionsHandler
from algo_trading_engine.dto import OptionContractDTO
from algo_trading_engine.vo import ExpirationDate, StrikePrice
from algo_trading_engine.common.models import OptionType


@pytest.fixture
def options_handler(tmp_path):
    return OptionsHandler("SPY", api_key="test_key", cache_dir=str(tmp_path), use_cache=False)


@pytest.fixture
def sample_contract():
    future_date = date.today() + timedelta(days=30)
    return OptionContractDTO(
        ticker="O:SPY211119C00045000",
        underlying_ticker="SPY",
        contract_type=OptionType.CALL,
        strike_price=StrikePrice(Decimal("450.0")),
        expiration_date=ExpirationDate(future_date),
        exercise_style="american",
        shares_per_contract=100,
        primary_exchange="BATO",
        cfi="OCASPS",
        additional_underlyings=None,
    )


def _fresh_sip_timestamp_ns() -> int:
    return int(time.time() * 1_000_000_000)


def _stale_sip_timestamp_ns(hours: int = 2) -> int:
    return int((time.time() - hours * 3600) * 1_000_000_000)


def _make_snapshot(
    *,
    midpoint=None,
    bid=None,
    ask=None,
    trade_price=None,
    trade_sip_timestamp_ns=None,
    day_close=None,
    day_open=10.0,
    day_high=11.0,
    day_low=9.5,
    day_volume=42,
    day_vwap=10.25,
    include_day=False,
):
    last_quote = None
    if any(v is not None for v in (midpoint, bid, ask)):
        last_quote = Mock(
            midpoint=midpoint,
            bid=bid,
            ask=ask,
        )
    last_trade = None
    if trade_price is not None:
        sip_ts = trade_sip_timestamp_ns
        if sip_ts is None:
            sip_ts = _fresh_sip_timestamp_ns()
        last_trade = Mock(price=trade_price, sip_timestamp=sip_ts)
    day = None
    if include_day or day_close is not None:
        day = Mock(
            open=day_open,
            high=day_high,
            low=day_low,
            close=day_close,
            volume=day_volume,
            vwap=day_vwap,
        )
    return Mock(last_quote=last_quote, last_trade=last_trade, day=day)


class TestSnapshotClosePrice:
    def test_uses_midpoint(self):
        snapshot = _make_snapshot(midpoint=21.075, day_close=20.0)
        assert OptionsHandler._snapshot_close_price(
            snapshot.last_quote, snapshot.last_trade, snapshot.day
        ) == pytest.approx(21.075)

    def test_falls_back_to_bid_ask(self):
        snapshot = _make_snapshot(bid=20.9, ask=21.25, day_close=20.0)
        assert OptionsHandler._snapshot_close_price(
            snapshot.last_quote, snapshot.last_trade, snapshot.day
        ) == pytest.approx(21.075)

    def test_falls_back_to_last_trade_when_fresh(self):
        snapshot = _make_snapshot(trade_price=19.5, day_close=20.0, include_day=True)
        assert OptionsHandler._snapshot_close_price(
            snapshot.last_quote, snapshot.last_trade, snapshot.day
        ) == pytest.approx(19.5)

    def test_falls_back_to_day_close_when_no_quote_or_trade(self):
        snapshot = _make_snapshot(day_close=18.75, include_day=True)
        with patch("algo_trading_engine.common.options_handler.get_logger") as mock_logger:
            price = OptionsHandler._snapshot_close_price(
                snapshot.last_quote,
                snapshot.last_trade,
                snapshot.day,
                ticker="O:SPY211119C00045000",
            )
        assert price == pytest.approx(18.75)
        mock_logger.return_value.info.assert_called_once()

    def test_stale_last_trade_falls_back_to_day_close(self):
        snapshot = _make_snapshot(
            trade_price=19.5,
            trade_sip_timestamp_ns=_stale_sip_timestamp_ns(),
            day_close=18.75,
            include_day=True,
        )
        with patch("algo_trading_engine.common.options_handler.get_logger") as mock_logger:
            price = OptionsHandler._snapshot_close_price(
                snapshot.last_quote,
                snapshot.last_trade,
                snapshot.day,
                ticker="O:SPY211119C00045000",
            )
        assert price == pytest.approx(18.75)
        mock_logger.return_value.info.assert_called_once()

    def test_last_trade_without_timestamp_falls_back_to_day_close(self):
        snapshot = _make_snapshot(day_close=18.75, include_day=True)
        snapshot.last_trade = Mock(price=19.5, sip_timestamp=None)
        with patch("algo_trading_engine.common.options_handler.get_logger"):
            price = OptionsHandler._snapshot_close_price(
                snapshot.last_quote,
                snapshot.last_trade,
                snapshot.day,
            )
        assert price == pytest.approx(18.75)

    def test_returns_none_when_no_price(self):
        snapshot = _make_snapshot()
        assert (
            OptionsHandler._snapshot_close_price(
                snapshot.last_quote, snapshot.last_trade, snapshot.day
            )
            is None
        )


class TestConvertSnapshotToDto:
    def test_maps_full_snapshot(self, options_handler, sample_contract):
        snapshot = _make_snapshot(midpoint=21.075, include_day=True, day_close=21.0)
        bar = options_handler._convert_snapshot_to_dto(sample_contract.ticker, snapshot)

        assert bar is not None
        assert bar.ticker == sample_contract.ticker
        assert bar.close_price == Decimal("21.075")
        assert bar.open_price == Decimal("10.0")
        assert bar.high_price == Decimal("11.0")
        assert bar.low_price == Decimal("9.5")
        assert bar.volume == 42
        assert bar.volume_weighted_avg_price == Decimal("10.25")
        assert bar.number_of_transactions == 1

    def test_uses_close_for_ohlc_when_day_missing(self, options_handler, sample_contract):
        snapshot = _make_snapshot(trade_price=5.5, trade_sip_timestamp_ns=_fresh_sip_timestamp_ns())
        bar = options_handler._convert_snapshot_to_dto(sample_contract.ticker, snapshot)

        assert bar is not None
        assert bar.close_price == Decimal("5.5")
        assert bar.open_price == Decimal("5.5")
        assert bar.high_price == Decimal("5.5")
        assert bar.low_price == Decimal("5.5")
        assert bar.volume == 0

    def test_returns_none_when_no_price(self, options_handler, sample_contract):
        snapshot = _make_snapshot()
        assert options_handler._convert_snapshot_to_dto(sample_contract.ticker, snapshot) is None


class TestGetOptionSnapshot:
    @patch.object(OptionsHandler, "_fetch_snapshot_from_api")
    def test_returns_bar_from_snapshot(self, mock_fetch, options_handler, sample_contract):
        mock_fetch.return_value = _make_snapshot(midpoint=3.25)

        bar = options_handler.get_option_snapshot(sample_contract)

        assert bar is not None
        assert bar.close_price == Decimal("3.25")
        mock_fetch.assert_called_once_with(sample_contract)

    @patch.object(OptionsHandler, "_fetch_snapshot_from_api", return_value=None)
    def test_returns_none_when_api_returns_nothing(self, _mock_fetch, options_handler, sample_contract):
        assert options_handler.get_option_snapshot(sample_contract) is None

    @patch.object(OptionsHandler, "_fetch_snapshot_from_api")
    def test_returns_none_when_conversion_fails(self, mock_fetch, options_handler, sample_contract):
        mock_fetch.return_value = _make_snapshot()
        assert options_handler.get_option_snapshot(sample_contract) is None
