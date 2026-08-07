"""Unit tests for OptionsHandler snapshot-based live option pricing."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from algo_trading_engine.common.options_handler import OptionsHandler
from algo_trading_engine.dto import OptionBarDTO, OptionContractDTO
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


def _make_snapshot(
    *,
    day_close=None,
    day_open=10.0,
    day_high=11.0,
    day_low=9.5,
    day_volume=42,
    day_vwap=10.25,
    include_day=False,
    underlying_price=None,
    strike_price=450.0,
    contract_type="call",
):
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
    details = Mock(
        strike_price=strike_price,
        contract_type=contract_type,
    )
    underlying_asset = None
    if underlying_price is not None:
        underlying_asset = Mock(price=underlying_price)
    return Mock(day=day, details=details, underlying_asset=underlying_asset)


class TestFromSnapshot:
    def test_maps_full_snapshot(self, sample_contract):
        snapshot = _make_snapshot(day_close=21.0)
        bar = OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot)

        assert bar is not None
        assert bar.ticker == sample_contract.ticker
        assert bar.close_price == Decimal("21.0")
        assert bar.open_price == Decimal("10.0")
        assert bar.high_price == Decimal("11.0")
        assert bar.low_price == Decimal("9.5")
        assert bar.volume == 42
        assert bar.volume_weighted_avg_price == Decimal("10.25")
        assert bar.number_of_transactions == 1

    def test_uses_close_for_ohlc_when_day_fields_missing(self, sample_contract):
        snapshot = _make_snapshot(
            day_close=5.5,
            day_open=None,
            day_high=None,
            day_low=None,
            day_volume=0,
            day_vwap=None,
        )
        bar = OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot)

        assert bar is not None
        assert bar.close_price == Decimal("5.5")
        assert bar.open_price == Decimal("5.5")
        assert bar.high_price == Decimal("5.5")
        assert bar.low_price == Decimal("5.5")
        assert bar.volume_weighted_avg_price == Decimal("5.5")
        assert bar.volume == 0

    def test_returns_none_when_day_missing(self, sample_contract):
        snapshot = _make_snapshot()
        assert OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot) is None

    def test_returns_none_when_day_close_missing(self, sample_contract):
        snapshot = _make_snapshot(include_day=True, day_close=None)
        assert OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot) is None

    def test_close_below_intrinsic_uses_vwap(self, sample_contract):
        # 500 spot, 450 call -> intrinsic 50; close 40 is impossible, vwap 51 is ok
        snapshot = _make_snapshot(
            day_close=40.0,
            day_vwap=51.0,
            underlying_price=500.0,
            strike_price=450.0,
            contract_type="call",
        )
        bar = OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot)
        assert bar is not None
        assert bar.close_price == Decimal("51.0")

    def test_close_below_intrinsic_and_vwap_rejects_leg(self, sample_contract):
        snapshot = _make_snapshot(
            day_close=40.0,
            day_vwap=45.0,
            underlying_price=500.0,
        )
        assert OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot) is None

    def test_close_above_intrinsic_uses_close(self, sample_contract):
        snapshot = _make_snapshot(
            day_close=55.0,
            day_vwap=52.0,
            underlying_price=500.0,
        )
        bar = OptionBarDTO.from_snapshot(sample_contract.ticker, snapshot)
        assert bar is not None
        assert bar.close_price == Decimal("55.0")


class TestGetOptionSnapshot:
    @patch.object(OptionsHandler, "_fetch_snapshot_from_api")
    def test_returns_bar_from_snapshot(self, mock_fetch, options_handler, sample_contract):
        mock_fetch.return_value = _make_snapshot(day_close=3.25)

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
