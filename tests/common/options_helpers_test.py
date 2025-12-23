"""
Tests for Phase 4 OptionsRetrieverHelper strategy-specific methods.

This module tests the new strategy-specific helper methods added in Phase 4
of the OptionsHandler refactoring.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict

from src.common.options_helpers import OptionsRetrieverHelper
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikePrice, ExpirationDate
)
from src.common.models import OptionType, SignalType


class TestOptionsRetrieverHelperPhase4:
    """Test cases for Phase 4 strategy-specific helper methods."""
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts for testing."""
        # Use a date that's actually a Friday (for weekly expiration test)
        # Find next Friday
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # If today is Friday, use next Friday
        future_date = today + timedelta(days=days_until_friday)
        return [
            # Call options
            OptionContractDTO(
                ticker="O:SPY250929C00580000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(580.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929C00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929C00605000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(605.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            # Put options
            OptionContractDTO(
                ticker="O:SPY250929P00580000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(580.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929P00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929P00605000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(605.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample bar data for testing."""
        return {
            "O:SPY250929C00580000": OptionBarDTO(
                ticker="O:SPY250929C00580000",
                timestamp=datetime.now(),
                open_price=Decimal('15.50'),
                high_price=Decimal('16.00'),
                low_price=Decimal('15.25'),
                close_price=Decimal('15.75'),
                volume=150,
                volume_weighted_avg_price=Decimal('15.60'),
                number_of_transactions=25,
                adjusted=True
            ),
            "O:SPY250929C00600000": OptionBarDTO(
                ticker="O:SPY250929C00600000",
                timestamp=datetime.now(),
                open_price=Decimal('10.50'),
                high_price=Decimal('11.00'),
                low_price=Decimal('10.25'),
                close_price=Decimal('10.75'),
                volume=200,
                volume_weighted_avg_price=Decimal('10.60'),
                number_of_transactions=30,
                adjusted=True
            ),
            "O:SPY250929P00600000": OptionBarDTO(
                ticker="O:SPY250929P00600000",
                timestamp=datetime.now(),
                open_price=Decimal('8.50'),
                high_price=Decimal('9.00'),
                low_price=Decimal('8.25'),
                close_price=Decimal('8.75'),
                volume=50,
                volume_weighted_avg_price=Decimal('8.60'),
                number_of_transactions=15,
                adjusted=True
            )
        }
    
    def test_find_credit_spread_legs_call(self, sample_contracts):
        """Test finding call credit spread legs."""
        current_price = 600.0
        expiration_date = str(sample_contracts[0].expiration_date)
        
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            sample_contracts, current_price, expiration_date, OptionType.CALL, spread_width=5
        )
        
        assert short_leg is not None
        assert long_leg is not None
        assert short_leg.contract_type == OptionType.CALL
        assert long_leg.contract_type == OptionType.CALL
        assert short_leg.strike_price.value <= Decimal(str(current_price))
        assert long_leg.strike_price.value > short_leg.strike_price.value
    
    def test_find_credit_spread_legs_put(self, sample_contracts):
        """Test finding put credit spread legs."""
        current_price = 600.0
        expiration_date = str(sample_contracts[0].expiration_date)
        
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            sample_contracts, current_price, expiration_date, OptionType.PUT, spread_width=5
        )
        
        assert short_leg is not None
        assert long_leg is not None
        assert short_leg.contract_type == OptionType.PUT
        assert long_leg.contract_type == OptionType.PUT
        assert short_leg.strike_price.value >= Decimal(str(current_price))
        assert long_leg.strike_price.value < short_leg.strike_price.value
    
    def test_find_credit_spread_legs_insufficient_contracts(self):
        """Test finding credit spread legs with insufficient contracts."""
        contracts = []  # Empty list
        
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            contracts, 600.0, "2025-09-29", OptionType.CALL, spread_width=5
        )
        
        assert short_leg is None
        assert long_leg is None
    
    def test_calculate_credit_spread_premium(self, sample_contracts):
        """Test calculating credit spread premium."""
        short_leg = sample_contracts[1]  # 600 call
        long_leg = sample_contracts[2]   # 605 call
        short_premium = 2.50
        long_premium = 1.00
        
        net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
            short_leg, long_leg, short_premium, long_premium
        )
        
        assert net_credit == 1.50
    
    def test_find_optimal_expiration(self, sample_contracts):
        """Test finding optimal expiration date."""
        # The fixture uses a date that's 1-7 days away (next Friday)
        # So we need to use a range that includes that
        optimal_exp = OptionsRetrieverHelper.find_optimal_expiration(
            sample_contracts, min_days=1, max_days=10
        )
        
        assert optimal_exp is not None
        assert optimal_exp == str(sample_contracts[0].expiration_date)
    
    def test_find_optimal_expiration_no_match(self, sample_contracts):
        """Test finding optimal expiration with no matching dates."""
        # Calculate days to the fixture's expiration date
        from datetime import date
        today = date.today()
        fixture_exp_date = sample_contracts[0].expiration_date.date
        days_to_fixture = (fixture_exp_date - today).days
        
        # Use a range that definitely excludes the fixture's date
        # If fixture is within 1-5 days, use a range after it
        # If fixture is far away, use a range before it
        if 1 <= days_to_fixture <= 5:
            # Fixture is in the 1-5 range, so use a range after it
            min_days = days_to_fixture + 10
            max_days = days_to_fixture + 20
        else:
            # Fixture is outside 1-5, so use 1-5 range (shouldn't match)
            min_days = 1
            max_days = 5
        
        optimal_exp = OptionsRetrieverHelper.find_optimal_expiration(
            sample_contracts, min_days=min_days, max_days=max_days
        )
        
        assert optimal_exp is None
    
    def test_calculate_implied_volatility_rank(self, sample_contracts):
        """Test calculating implied volatility rank."""
        iv_ranks = OptionsRetrieverHelper.calculate_implied_volatility_rank(
            sample_contracts, current_price=600.0
        )
        
        assert len(iv_ranks) == len(sample_contracts)
        for ticker, rank in iv_ranks.items():
            assert 0.0 <= rank <= 100.0
            assert ticker in [c.ticker for c in sample_contracts]
    
    def test_find_high_volume_contracts(self, sample_contracts, sample_bars):
        """Test finding high volume contracts."""
        high_volume_contracts = OptionsRetrieverHelper.find_high_volume_contracts(
            sample_contracts, sample_bars, min_volume=100
        )
        
        assert len(high_volume_contracts) == 2  # Two contracts with volume >= 100
        tickers = [c.ticker for c in high_volume_contracts]
        assert "O:SPY250929C00580000" in tickers
        assert "O:SPY250929C00600000" in tickers
    
    def test_find_high_volume_contracts_no_matches(self, sample_contracts, sample_bars):
        """Test finding high volume contracts with high threshold."""
        high_volume_contracts = OptionsRetrieverHelper.find_high_volume_contracts(
            sample_contracts, sample_bars, min_volume=1000
        )
        
        assert len(high_volume_contracts) == 0
    
    def test_calculate_delta_exposure(self, sample_contracts, sample_bars):
        """Test calculating delta exposure."""
        delta_exposure = OptionsRetrieverHelper.calculate_delta_exposure(
            sample_contracts[:3], sample_bars, quantity=1
        )
        
        assert isinstance(delta_exposure, float)
        # Should be positive for calls, negative for puts
        assert delta_exposure > 0  # More calls than puts in first 3 contracts
    
    def test_find_iron_condor_legs(self, sample_contracts):
        """Test finding iron condor legs."""
        current_price = 600.0
        expiration_date = str(sample_contracts[0].expiration_date)
        
        put_long, put_short, call_short, call_long = OptionsRetrieverHelper.find_iron_condor_legs(
            sample_contracts, current_price, expiration_date, spread_width=5
        )
        
        assert all([put_long, put_short, call_short, call_long])
        assert put_long.contract_type == OptionType.PUT
        assert put_short.contract_type == OptionType.PUT
        assert call_short.contract_type == OptionType.CALL
        assert call_long.contract_type == OptionType.CALL
    
    def test_find_iron_condor_legs_insufficient_contracts(self):
        """Test finding iron condor legs with insufficient contracts."""
        contracts = []  # Empty list
        
        put_long, put_short, call_short, call_long = OptionsRetrieverHelper.find_iron_condor_legs(
            contracts, 600.0, "2025-09-29", spread_width=5
        )
        
        assert all([put_long is None, put_short is None, call_short is None, call_long is None])
    
    def test_calculate_breakeven_points_call(self, sample_contracts):
        """Test calculating breakeven points for call credit spread."""
        short_leg = sample_contracts[1]  # 600 call
        long_leg = sample_contracts[2]   # 605 call
        net_credit = 1.50
        
        lower_be, upper_be = OptionsRetrieverHelper.calculate_breakeven_points(
            short_leg, long_leg, net_credit, OptionType.CALL
        )
        
        assert lower_be == upper_be  # Call credit spread has single breakeven
        assert lower_be == 601.50  # 600 + 1.50
    
    def test_calculate_breakeven_points_put(self, sample_contracts):
        """Test calculating breakeven points for put credit spread."""
        short_leg = sample_contracts[4]  # 600 put
        long_leg = sample_contracts[3]   # 580 put
        net_credit = 1.50
        
        lower_be, upper_be = OptionsRetrieverHelper.calculate_breakeven_points(
            short_leg, long_leg, net_credit, OptionType.PUT
        )
        
        assert lower_be == upper_be  # Put credit spread has single breakeven
        assert lower_be == 598.50  # 600 - 1.50
    
    def test_find_weekly_expirations(self, sample_contracts):
        """Test finding weekly expirations."""
        weekly_expirations = OptionsRetrieverHelper.find_weekly_expirations(sample_contracts)
        
        # All contracts have the same expiration date, which should be a Friday
        # The fixture now uses a Friday date, so we should find it
        assert len(weekly_expirations) == 1
        assert weekly_expirations[0] == str(sample_contracts[0].expiration_date)
    
    def test_find_monthly_expirations(self):
        """Test finding monthly expirations."""
        # Create contracts with a third Friday date
        today = date.today()
        first_day = today.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        # If third Friday is in the past, use next month's third Friday
        if third_friday < today:
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            first_day = next_month
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
        
        monthly_contracts = [
            OptionContractDTO(
                ticker="O:SPY250929C00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(third_friday),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
        
        monthly_expirations = OptionsRetrieverHelper.find_monthly_expirations(monthly_contracts)
        
        # Should find the third Friday expiration
        assert len(monthly_expirations) == 1
        assert monthly_expirations[0] == str(monthly_contracts[0].expiration_date)
    

class TestOptionsRetrieverHelperIntegration:
    """Integration tests for OptionsRetrieverHelper with real-world scenarios."""
    
    def test_complete_credit_spread_analysis(self):
        """Test complete credit spread analysis workflow."""
        # Create realistic contract data
        future_date = date.today() + timedelta(days=30)
        contracts = [
            OptionContractDTO(
                ticker="O:SPY250929C00595000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(595.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929C00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929C00605000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(605.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
        
        current_price = 600.0
        expiration_date = str(future_date)
        
        # Find credit spread legs
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            contracts, current_price, expiration_date, OptionType.CALL, spread_width=5
        )
        
        assert short_leg is not None
        assert long_leg is not None
        
        # Calculate premiums (simulated)
        short_premium = 2.50
        long_premium = 1.00
        net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
            short_leg, long_leg, short_premium, long_premium
        )
        
        # Calculate breakeven
        lower_be, upper_be = OptionsRetrieverHelper.calculate_breakeven_points(
            short_leg, long_leg, net_credit, OptionType.CALL
        )
        
        # Verify all calculations are consistent
        assert net_credit == 1.50
        assert lower_be == upper_be == 601.50
        
        # Verify spread width (calculate manually since method doesn't exist)
        spread_width = abs(float(short_leg.strike_price.value) - float(long_leg.strike_price.value))
        assert spread_width == 5.0
