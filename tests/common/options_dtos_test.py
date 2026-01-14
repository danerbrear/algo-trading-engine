"""
Unit tests for Options DTOs and VOs.

Tests all DTOs and VOs created for the refactored OptionsHandler.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from dataclasses import FrozenInstanceError

from algo_trading_engine.vo import StrikePrice, ExpirationDate
from algo_trading_engine.dto import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    OptionsChainDTO
)
from algo_trading_engine.common.models import OptionType


class TestStrikePrice:
    """Test cases for StrikePrice Value Object."""
    
    def test_valid_creation(self):
        """Test valid strike price creation."""
        strike = StrikePrice(Decimal('100.50'))
        assert strike.value == Decimal('100.50')
        assert str(strike) == "$100.50"
        assert repr(strike) == "StrikePrice(100.50)"
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike prices raise ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            StrikePrice(Decimal('-10'))
    
    def test_zero_strike_raises_error(self):
        """Test that zero strike prices raise ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            StrikePrice(Decimal('0'))
    
    def test_excessive_strike_raises_error(self):
        """Test that excessive strike prices raise ValueError."""
        with pytest.raises(ValueError, match="Strike price cannot exceed"):
            StrikePrice(Decimal('10001'))
    
    def test_is_atm(self):
        """Test at-the-money detection."""
        strike = StrikePrice(Decimal('100'))
        current_price = Decimal('100')
        
        assert strike.is_atm(current_price) is True
        assert strike.is_atm(current_price, Decimal('0.01')) is True
        # Test with a strike that's slightly off
        strike_off = StrikePrice(Decimal('100.01'))
        assert strike_off.is_atm(current_price, Decimal('0.001')) is False
    
    def test_is_itm_call(self):
        """Test in-the-money detection for calls."""
        strike = StrikePrice(Decimal('95'))  # ITM call
        current_price = Decimal('100')
        
        assert strike.is_itm(current_price, OptionType.CALL) is True
        assert strike.is_otm(current_price, OptionType.CALL) is False
    
    def test_is_itm_put(self):
        """Test in-the-money detection for puts."""
        strike = StrikePrice(Decimal('105'))  # ITM put
        current_price = Decimal('100')
        
        assert strike.is_itm(current_price, OptionType.PUT) is True
        assert strike.is_otm(current_price, OptionType.PUT) is False
    
    def test_is_otm_call(self):
        """Test out-of-the-money detection for calls."""
        strike = StrikePrice(Decimal('105'))  # OTM call
        current_price = Decimal('100')
        
        assert strike.is_otm(current_price, OptionType.CALL) is True
        assert strike.is_itm(current_price, OptionType.CALL) is False
    
    def test_is_otm_put(self):
        """Test out-of-the-money detection for puts."""
        strike = StrikePrice(Decimal('95'))  # OTM put
        current_price = Decimal('100')
        
        assert strike.is_otm(current_price, OptionType.PUT) is True
        assert strike.is_itm(current_price, OptionType.PUT) is False
    
    def test_immutability(self):
        """Test that StrikePrice is immutable."""
        strike = StrikePrice(Decimal('100'))
        with pytest.raises(FrozenInstanceError):
            strike.value = Decimal('200')


class TestExpirationDate:
    """Test cases for ExpirationDate Value Object."""
    
    def test_valid_creation(self):
        """Test valid expiration date creation."""
        future_date = date.today() + timedelta(days=30)
        exp_date = ExpirationDate(future_date)
        assert exp_date.date == future_date
        assert str(exp_date) == future_date.strftime('%Y-%m-%d')
    
    def test_days_to_expiration(self):
        """Test days to expiration calculation."""
        future_date = date.today() + timedelta(days=30)
        exp_date = ExpirationDate(future_date)
        
        assert exp_date.days_to_expiration() == 30
        assert exp_date.days_to_expiration(date.today() + timedelta(days=10)) == 20
    
    def test_is_weekly(self):
        """Test weekly expiration detection."""
        # Find next Friday
        today = date.today()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        friday = today + timedelta(days=days_ahead)
        
        exp_date = ExpirationDate(friday)
        assert exp_date.is_weekly() is True
    
    def test_is_monthly(self):
        """Test monthly expiration detection."""
        # Create a third Friday of a future month
        today = date.today()
        year = today.year + 1  # Use next year to ensure it's in the future
        month = 1
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        exp_date = ExpirationDate(third_friday)
        assert exp_date.is_monthly() is True
    
    def test_immutability(self):
        """Test that ExpirationDate is immutable."""
        future_date = date.today() + timedelta(days=30)
        exp_date = ExpirationDate(future_date)
        with pytest.raises(FrozenInstanceError):
            exp_date.date = date.today()


class TestOptionContractDTO:
    """Test cases for OptionContractDTO."""
    
    def test_valid_creation(self):
        """Test valid contract creation."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contract = OptionContractDTO(
            ticker="O:AAPL211119C00085000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        assert contract.ticker == "O:AAPL211119C00085000"
        assert contract.underlying_ticker == "AAPL"
        assert contract.contract_type == OptionType.CALL
        assert contract.is_call() is True
        assert contract.is_put() is False
    
    def test_invalid_ticker_format_raises_error(self):
        """Test that invalid ticker format raises ValueError."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        with pytest.raises(ValueError, match="Invalid ticker format"):
            OptionContractDTO(
                ticker="INVALID_TICKER",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=exp_date
            )
    
    def test_invalid_underlying_ticker_raises_error(self):
        """Test that invalid underlying ticker raises ValueError."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        with pytest.raises(ValueError, match="Invalid underlying ticker"):
            OptionContractDTO(
                ticker="O:AAPL211119C00085000",
                underlying_ticker="invalid-ticker",
                contract_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=exp_date
            )
    
    def test_invalid_exercise_style_raises_error(self):
        """Test that invalid exercise style raises ValueError."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        with pytest.raises(ValueError, match="Invalid exercise style"):
            OptionContractDTO(
                ticker="O:AAPL211119C00085000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=exp_date,
                exercise_style="invalid"
            )
    
    def test_negative_shares_per_contract_raises_error(self):
        """Test that negative shares per contract raises ValueError."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        with pytest.raises(ValueError, match="Shares per contract must be positive"):
            OptionContractDTO(
                ticker="O:AAPL211119C00085000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=exp_date,
                shares_per_contract=-1
            )
    
    def test_days_to_expiration(self):
        """Test days to expiration calculation."""
        strike = StrikePrice(Decimal('100'))
        future_date = date.today() + timedelta(days=30)
        exp_date = ExpirationDate(future_date)
        
        contract = OptionContractDTO(
            ticker="O:AAPL211119C00085000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        assert contract.days_to_expiration() == 30
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        original = OptionContractDTO(
            ticker="O:AAPL211119C00085000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date,
            primary_exchange="BATO",
            cfi="OCASPS"
        )
        
        data = original.to_dict()
        restored = OptionContractDTO.from_dict(data)
        
        assert restored.ticker == original.ticker
        assert restored.underlying_ticker == original.underlying_ticker
        assert restored.contract_type == original.contract_type
        assert restored.strike_price.value == original.strike_price.value
        assert restored.expiration_date.date == original.expiration_date.date
        assert restored.primary_exchange == original.primary_exchange
        assert restored.cfi == original.cfi


class TestOptionBarDTO:
    """Test cases for OptionBarDTO."""
    
    def test_valid_creation(self):
        """Test valid bar creation."""
        bar = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        assert bar.ticker == "O:AAPL211119C00085000"
        assert bar.open_price == Decimal('25.50')
        assert bar.close_price == Decimal('26.00')
        assert bar.volume == 100
    
    def test_negative_price_raises_error(self):
        """Test that negative prices raise ValueError."""
        with pytest.raises(ValueError, match="open price cannot be negative"):
            OptionBarDTO(
                ticker="O:AAPL211119C00085000",
                timestamp=datetime(2021, 11, 19, 16, 0, 0),
                open_price=Decimal('-25.50'),
                high_price=Decimal('26.20'),
                low_price=Decimal('25.20'),
                close_price=Decimal('26.00'),
                volume=100,
                volume_weighted_avg_price=Decimal('25.80'),
                number_of_transactions=5
            )
    
    def test_high_less_than_low_raises_error(self):
        """Test that high < low raises ValueError."""
        with pytest.raises(ValueError, match="High price cannot be less than low price"):
            OptionBarDTO(
                ticker="O:AAPL211119C00085000",
                timestamp=datetime(2021, 11, 19, 16, 0, 0),
                open_price=Decimal('25.50'),
                high_price=Decimal('25.20'),  # High < Low
                low_price=Decimal('26.00'),
                close_price=Decimal('26.00'),
                volume=100,
                volume_weighted_avg_price=Decimal('25.80'),
                number_of_transactions=5
            )
    
    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            OptionBarDTO(
                ticker="O:AAPL211119C00085000",
                timestamp=datetime(2021, 11, 19, 16, 0, 0),
                open_price=Decimal('25.50'),
                high_price=Decimal('26.20'),
                low_price=Decimal('25.20'),
                close_price=Decimal('26.00'),
                volume=-100,  # Negative volume
                volume_weighted_avg_price=Decimal('25.80'),
                number_of_transactions=5
            )
    
    def test_price_range(self):
        """Test price range calculation."""
        bar = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        assert bar.price_range() == Decimal('1.00')  # 26.20 - 25.20
    
    def test_candle_types(self):
        """Test candle type detection."""
        # Green candle
        green_bar = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),  # Close > Open
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        assert green_bar.is_green_candle() is True
        assert green_bar.is_red_candle() is False
        assert green_bar.is_doji() is False
        
        # Red candle
        red_bar = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('26.00'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('25.50'),  # Close < Open
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        assert red_bar.is_green_candle() is False
        assert red_bar.is_red_candle() is True
        assert red_bar.is_doji() is False
        
        # Doji candle
        doji_bar = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('25.50'),  # Close = Open (exact doji)
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        assert doji_bar.is_green_candle() is False
        assert doji_bar.is_red_candle() is False
        assert doji_bar.is_doji() is True
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = OptionBarDTO(
            ticker="O:AAPL211119C00085000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        data = original.to_dict()
        restored = OptionBarDTO.from_dict(data)
        
        assert restored.ticker == original.ticker
        assert restored.timestamp == original.timestamp
        assert restored.open_price == original.open_price
        assert restored.high_price == original.high_price
        assert restored.low_price == original.low_price
        assert restored.close_price == original.close_price
        assert restored.volume == original.volume
        assert restored.volume_weighted_avg_price == original.volume_weighted_avg_price
        assert restored.number_of_transactions == original.number_of_transactions


class TestStrikeRangeDTO:
    """Test cases for StrikeRangeDTO."""
    
    def test_valid_creation(self):
        """Test valid strike range creation."""
        min_strike = StrikePrice(Decimal('90'))
        max_strike = StrikePrice(Decimal('110'))
        target_strike = StrikePrice(Decimal('100'))
        
        strike_range = StrikeRangeDTO(
            min_strike=min_strike,
            max_strike=max_strike,
            target_strike=target_strike,
            tolerance=Decimal('5')
        )
        
        assert strike_range.min_strike == min_strike
        assert strike_range.max_strike == max_strike
        assert strike_range.target_strike == target_strike
        assert strike_range.tolerance == Decimal('5')
    
    def test_min_greater_than_max_raises_error(self):
        """Test that min > max raises ValueError."""
        min_strike = StrikePrice(Decimal('110'))
        max_strike = StrikePrice(Decimal('90'))
        
        with pytest.raises(ValueError, match="Min strike cannot be greater than max strike"):
            StrikeRangeDTO(min_strike=min_strike, max_strike=max_strike)
    
    def test_negative_tolerance_raises_error(self):
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="Tolerance cannot be negative"):
            StrikeRangeDTO(tolerance=Decimal('-5'))
    
    def test_contains_strike(self):
        """Test strike containment logic."""
        min_strike = StrikePrice(Decimal('90'))
        max_strike = StrikePrice(Decimal('110'))
        strike_range = StrikeRangeDTO(min_strike=min_strike, max_strike=max_strike)
        
        # Within range
        assert strike_range.contains_strike(StrikePrice(Decimal('100'))) is True
        assert strike_range.contains_strike(StrikePrice(Decimal('90'))) is True
        assert strike_range.contains_strike(StrikePrice(Decimal('110'))) is True
        
        # Outside range
        assert strike_range.contains_strike(StrikePrice(Decimal('85'))) is False
        assert strike_range.contains_strike(StrikePrice(Decimal('115'))) is False
    
    def test_is_target_within_tolerance(self):
        """Test target tolerance logic."""
        target_strike = StrikePrice(Decimal('100'))
        strike_range = StrikeRangeDTO(target_strike=target_strike, tolerance=Decimal('5'))
        
        # Within tolerance
        assert strike_range.is_target_within_tolerance(StrikePrice(Decimal('100'))) is True
        assert strike_range.is_target_within_tolerance(StrikePrice(Decimal('105'))) is True
        assert strike_range.is_target_within_tolerance(StrikePrice(Decimal('95'))) is True
        
        # Outside tolerance
        assert strike_range.is_target_within_tolerance(StrikePrice(Decimal('106'))) is False
        assert strike_range.is_target_within_tolerance(StrikePrice(Decimal('94'))) is False


class TestExpirationRangeDTO:
    """Test cases for ExpirationRangeDTO."""
    
    def test_valid_creation(self):
        """Test valid expiration range creation."""
        exp_range = ExpirationRangeDTO(min_days=5, max_days=30)
        assert exp_range.min_days == 5
        assert exp_range.max_days == 30
    
    def test_min_greater_than_max_raises_error(self):
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="Min days cannot be greater than max days"):
            ExpirationRangeDTO(min_days=30, max_days=5)
    
    def test_negative_days_raises_error(self):
        """Test that negative days raise ValueError."""
        with pytest.raises(ValueError, match="Min days cannot be negative"):
            ExpirationRangeDTO(min_days=-5)
        
        with pytest.raises(ValueError, match="Max days cannot be negative"):
            ExpirationRangeDTO(max_days=-5)
    
    def test_contains_expiration(self):
        """Test expiration containment logic."""
        exp_range = ExpirationRangeDTO(min_days=5, max_days=30)
        
        # Create expiration dates
        today = date.today()
        exp_10_days = ExpirationDate(today + timedelta(days=10))
        exp_40_days = ExpirationDate(today + timedelta(days=40))
        
        # Within range
        assert exp_range.contains_expiration(exp_10_days) is True
        
        # Outside range
        assert exp_range.contains_expiration(exp_40_days) is False
    
    def test_contains_expiration_with_tolerance(self):
        """Test expiration tolerance logic."""
        current_date = date.today()
        target_date = ExpirationDate(current_date + timedelta(days=30))
        exp_range = ExpirationRangeDTO(target_date=target_date, current_date=current_date)
        
        # Within tolerance (1 day)
        exp_29_days = ExpirationDate(current_date + timedelta(days=29))
        exp_31_days = ExpirationDate(current_date + timedelta(days=31))
        
        assert exp_range.contains_expiration_with_tolerance(exp_29_days, current_date) is True
        assert exp_range.contains_expiration_with_tolerance(exp_31_days, current_date) is True
        
        # Outside tolerance
        exp_28_days = ExpirationDate(current_date + timedelta(days=28))
        exp_32_days = ExpirationDate(current_date + timedelta(days=32))
        
        assert exp_range.contains_expiration_with_tolerance(exp_28_days, current_date) is False
        assert exp_range.contains_expiration_with_tolerance(exp_32_days, current_date) is False


class TestOptionsChainDTO:
    """Test cases for OptionsChainDTO."""
    
    def test_valid_creation(self):
        """Test valid options chain creation."""
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today()
        )
        
        assert chain.underlying_symbol == "AAPL"
        assert chain.current_price == Decimal('150.00')
        assert chain.date == date.today()
        assert len(chain.contracts) == 0
        assert len(chain.bars) == 0
    
    def test_invalid_underlying_symbol_raises_error(self):
        """Test that invalid underlying symbol raises ValueError."""
        with pytest.raises(ValueError, match="Invalid underlying symbol"):
            OptionsChainDTO(
                underlying_symbol="invalid-symbol",
                current_price=Decimal('150.00'),
                date=date.today()
            )
    
    def test_negative_current_price_raises_error(self):
        """Test that negative current price raises ValueError."""
        with pytest.raises(ValueError, match="Current price must be positive"):
            OptionsChainDTO(
                underlying_symbol="AAPL",
                current_price=Decimal('-150.00'),
                date=date.today()
            )
    
    def test_get_calls_and_puts(self):
        """Test call and put filtering."""
        strike = StrikePrice(Decimal('150'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        call_contract = OptionContractDTO(
            ticker="O:AAPL211119C00015000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        put_contract = OptionContractDTO(
            ticker="O:AAPL211119P00015000",
            underlying_ticker="AAPL",
            contract_type=OptionType.PUT,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today(),
            contracts=[call_contract, put_contract]
        )
        
        calls = chain.get_calls()
        puts = chain.get_puts()
        
        assert len(calls) == 1
        assert len(puts) == 1
        assert calls[0].is_call() is True
        assert puts[0].is_put() is True
    
    def test_get_contracts_by_strike(self):
        """Test strike-based filtering."""
        strike_140 = StrikePrice(Decimal('140'))
        strike_150 = StrikePrice(Decimal('150'))
        strike_160 = StrikePrice(Decimal('160'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contracts = [
            OptionContractDTO(
                ticker="O:AAPL211119C00014000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_140,
                expiration_date=exp_date
            ),
            OptionContractDTO(
                ticker="O:AAPL211119C00015000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_150,
                expiration_date=exp_date
            ),
            OptionContractDTO(
                ticker="O:AAPL211119C00016000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_160,
                expiration_date=exp_date
            )
        ]
        
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today(),
            contracts=contracts
        )
        
        # Filter for strikes 140-150
        strike_range = StrikeRangeDTO(
            min_strike=strike_140,
            max_strike=strike_150
        )
        
        filtered_contracts = chain.get_contracts_by_strike(strike_range)
        assert len(filtered_contracts) == 2
        assert all(contract.strike_price.value in [140, 150] for contract in filtered_contracts)
    
    def test_get_atm_contracts(self):
        """Test at-the-money contract filtering."""
        strike_149 = StrikePrice(Decimal('149'))
        strike_150 = StrikePrice(Decimal('150'))
        strike_151 = StrikePrice(Decimal('151'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contracts = [
            OptionContractDTO(
                ticker="O:AAPL211119C00014900",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_149,
                expiration_date=exp_date
            ),
            OptionContractDTO(
                ticker="O:AAPL211119C00015000",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_150,
                expiration_date=exp_date
            ),
            OptionContractDTO(
                ticker="O:AAPL211119C00015100",
                underlying_ticker="AAPL",
                contract_type=OptionType.CALL,
                strike_price=strike_151,
                expiration_date=exp_date
            )
        ]
        
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today(),
            contracts=contracts
        )
        
        atm_contracts = chain.get_atm_contracts(Decimal('0.01'))
        assert len(atm_contracts) == 1
        assert atm_contracts[0].strike_price.value == 150
    
    def test_get_bar_for_contract(self):
        """Test bar retrieval for specific contract."""
        strike = StrikePrice(Decimal('150'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contract = OptionContractDTO(
            ticker="O:AAPL211119C00015000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        bar = OptionBarDTO(
            ticker="O:AAPL211119C00015000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today(),
            contracts=[contract],
            bars={contract.ticker: bar}
        )
        
        retrieved_bar = chain.get_bar_for_contract(contract)
        assert retrieved_bar is not None
        assert retrieved_bar.ticker == contract.ticker
        
        # Test non-existent contract
        non_existent_contract = OptionContractDTO(
            ticker="O:AAPL211119C00020000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('200')),
            expiration_date=exp_date
        )
        
        assert chain.get_bar_for_contract(non_existent_contract) is None
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        strike = StrikePrice(Decimal('150'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contract = OptionContractDTO(
            ticker="O:AAPL211119C00015000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        bar = OptionBarDTO(
            ticker="O:AAPL211119C00015000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        original = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today(),
            contracts=[contract],
            bars={contract.ticker: bar}
        )
        
        data = original.to_dict()
        restored = OptionsChainDTO.from_dict(data)
        
        assert restored.underlying_symbol == original.underlying_symbol
        assert restored.current_price == original.current_price
        assert restored.date == original.date
        assert len(restored.contracts) == len(original.contracts)
        assert len(restored.bars) == len(original.bars)
        assert restored.contracts[0].ticker == original.contracts[0].ticker
        assert restored.bars[contract.ticker].ticker == original.bars[contract.ticker].ticker


class TestImmutability:
    """Test that all DTOs and VOs are immutable."""
    
    def test_strike_price_immutability(self):
        """Test StrikePrice immutability."""
        strike = StrikePrice(Decimal('100'))
        with pytest.raises(FrozenInstanceError):
            strike.value = Decimal('200')
    
    def test_expiration_date_immutability(self):
        """Test ExpirationDate immutability."""
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        with pytest.raises(FrozenInstanceError):
            exp_date.date = date.today()
    
    def test_option_contract_dto_immutability(self):
        """Test OptionContractDTO immutability."""
        strike = StrikePrice(Decimal('100'))
        exp_date = ExpirationDate(date.today() + timedelta(days=30))
        
        contract = OptionContractDTO(
            ticker="O:AAPL211119C00015000",
            underlying_ticker="AAPL",
            contract_type=OptionType.CALL,
            strike_price=strike,
            expiration_date=exp_date
        )
        
        with pytest.raises(FrozenInstanceError):
            contract.ticker = "O:INVALID"
    
    def test_option_bar_dto_immutability(self):
        """Test OptionBarDTO immutability."""
        bar = OptionBarDTO(
            ticker="O:AAPL211119C00015000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('25.50'),
            high_price=Decimal('26.20'),
            low_price=Decimal('25.20'),
            close_price=Decimal('26.00'),
            volume=100,
            volume_weighted_avg_price=Decimal('25.80'),
            number_of_transactions=5
        )
        
        with pytest.raises(FrozenInstanceError):
            bar.volume = 200
    
    def test_strike_range_dto_immutability(self):
        """Test StrikeRangeDTO immutability."""
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal('90')),
            max_strike=StrikePrice(Decimal('110'))
        )
        
        with pytest.raises(FrozenInstanceError):
            strike_range.min_strike = StrikePrice(Decimal('80'))
    
    def test_expiration_range_dto_immutability(self):
        """Test ExpirationRangeDTO immutability."""
        exp_range = ExpirationRangeDTO(min_days=5, max_days=30)
        
        with pytest.raises(FrozenInstanceError):
            exp_range.min_days = 10
    
    def test_options_chain_dto_immutability(self):
        """Test OptionsChainDTO immutability."""
        chain = OptionsChainDTO(
            underlying_symbol="AAPL",
            current_price=Decimal('150.00'),
            date=date.today()
        )
        
        with pytest.raises(FrozenInstanceError):
            chain.underlying_symbol = "MSFT"
