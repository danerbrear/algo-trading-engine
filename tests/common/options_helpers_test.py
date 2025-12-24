"""
Tests for Phase 4 OptionsRetrieverHelper strategy-specific methods.

This module tests the new strategy-specific helper methods added in Phase 4
of the OptionsHandler refactoring.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional

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


class TestFindBestCreditSpread:
    """Test cases for find_best_credit_spread helper method."""
    
    def test_find_best_credit_spread_selects_highest_ratio(self):
        """Test that find_best_credit_spread selects the spread with highest credit/width ratio."""
        from datetime import date as date_type
        
        # Create contracts for multiple spread widths
        # ATM at $100, OTM strikes at $96 (4pt), $94 (6pt), $92 (8pt)
        atm_contract = OptionContractDTO(
            ticker='O:SPY240115P100',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(100.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        otm_4pt = OptionContractDTO(
            ticker='O:SPY240115P96',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(96.0),  # 4-point spread
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        otm_6pt = OptionContractDTO(
            ticker='O:SPY240115P94',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(94.0),  # 6-point spread
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        otm_8pt = OptionContractDTO(
            ticker='O:SPY240115P92',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(92.0),  # 8-point spread
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        contracts = [atm_contract, otm_4pt, otm_6pt, otm_8pt]
        
        # Create bars with different credits to test ratio selection
        # 4pt: $2.00 credit / 4pt = 0.50 ratio (best)
        # 6pt: $1.50 credit / 6pt = 0.25 ratio
        # 8pt: $2.20 credit / 8pt = 0.275 ratio
        
        def get_bar(contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]:
            if '100' in contract.ticker:  # ATM
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('3.00'),
                    high_price=Decimal('3.10'),
                    low_price=Decimal('2.90'),
                    close_price=Decimal('3.00'),
                    volume=1000,
                    volume_weighted_avg_price=Decimal('3.00'),
                    number_of_transactions=100
                )
            elif '96' in contract.ticker:  # 4pt OTM
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('1.00'),
                    high_price=Decimal('1.10'),
                    low_price=Decimal('0.90'),
                    close_price=Decimal('1.00'),  # Credit: 3.00 - 1.00 = 2.00, Ratio: 2.00/4 = 0.50
                    volume=800,
                    volume_weighted_avg_price=Decimal('1.00'),
                    number_of_transactions=80
                )
            elif '94' in contract.ticker:  # 6pt OTM
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('1.50'),
                    high_price=Decimal('1.60'),
                    low_price=Decimal('1.40'),
                    close_price=Decimal('1.50'),  # Credit: 3.00 - 1.50 = 1.50, Ratio: 1.50/6 = 0.25
                    volume=800,
                    volume_weighted_avg_price=Decimal('1.50'),
                    number_of_transactions=80
                )
            elif '92' in contract.ticker:  # 8pt OTM
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('0.80'),
                    high_price=Decimal('0.90'),
                    low_price=Decimal('0.70'),
                    close_price=Decimal('0.80'),  # Credit: 3.00 - 0.80 = 2.20, Ratio: 2.20/8 = 0.275
                    volume=800,
                    volume_weighted_avg_price=Decimal('0.80'),
                    number_of_transactions=80
                )
            return None
        
        result = OptionsRetrieverHelper.find_best_credit_spread(
            contracts=contracts,
            current_price=100.0,
            expiration='2024-01-15',
            get_bar_fn=get_bar,
            date=datetime(2024, 1, 1),
            min_spread_width=4,
            max_spread_width=10,
            max_strike_difference=2.0,
            option_type=OptionType.PUT
        )
        
        # Should select 4-point spread (best credit/width ratio: 0.50)
        assert result is not None
        assert result['credit'] == 2.0  # 3.00 - 1.00
        assert result['width'] == 4.0  # 100 - 96
        assert result['credit_width_ratio'] == 0.5  # 2.00 / 4.0
        assert result['atm_contract'].ticker == 'O:SPY240115P100'
        assert result['otm_contract'].ticker == 'O:SPY240115P96'
    
    def test_find_best_credit_spread_rejects_strikes_too_far(self):
        """Test that spreads are rejected when OTM strike is more than max_strike_difference from target."""
        from datetime import date as date_type
        
        atm_contract = OptionContractDTO(
            ticker='O:SPY240115P100',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(100.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        # Only have OTM at $90 (10 points away)
        # For 4pt target ($96), $90 is 6 points away (rejected with max_difference=2.0)
        # For 6pt target ($94), $90 is 4 points away (rejected)
        # For 8pt target ($92), $90 is 2 points away (accepted)
        # For 10pt target ($90), $90 is 0 points away (accepted)
        otm_contract = OptionContractDTO(
            ticker='O:SPY240115P90',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(90.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        contracts = [atm_contract, otm_contract]
        
        def get_bar(contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]:
            if '100' in contract.ticker:
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('3.00'),
                    high_price=Decimal('3.10'),
                    low_price=Decimal('2.90'),
                    close_price=Decimal('3.00'),
                    volume=1000,
                    volume_weighted_avg_price=Decimal('3.00'),
                    number_of_transactions=100
                )
            else:  # OTM at 90
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('0.50'),
                    high_price=Decimal('0.60'),
                    low_price=Decimal('0.40'),
                    close_price=Decimal('0.50'),
                    volume=800,
                    volume_weighted_avg_price=Decimal('0.50'),
                    number_of_transactions=80
                )
        
        result = OptionsRetrieverHelper.find_best_credit_spread(
            contracts=contracts,
            current_price=100.0,
            expiration='2024-01-15',
            get_bar_fn=get_bar,
            date=datetime(2024, 1, 1),
            min_spread_width=4,
            max_spread_width=10,
            max_strike_difference=2.0,
            option_type=OptionType.PUT
        )
        
        # Should select 10pt spread (exact match, best ratio)
        assert result is not None
        assert result['width'] == 10.0  # 100 - 90
        assert result['credit'] == 2.5  # 3.00 - 0.50
    
    def test_find_best_credit_spread_no_valid_spreads(self):
        """Test that None is returned when no valid spreads are found."""
        from datetime import date as date_type
        
        # Only ATM contract, no OTM contracts
        atm_contract = OptionContractDTO(
            ticker='O:SPY240115P100',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(100.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        contracts = [atm_contract]
        
        def get_bar(contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]:
            return None
        
        result = OptionsRetrieverHelper.find_best_credit_spread(
            contracts=contracts,
            current_price=100.0,
            expiration='2024-01-15',
            get_bar_fn=get_bar,
            date=datetime(2024, 1, 1),
            min_spread_width=4,
            max_spread_width=10,
            max_strike_difference=2.0,
            option_type=OptionType.PUT
        )
        
        assert result is None
    
    def test_find_best_credit_spread_all_rejected_due_to_strike_distance(self):
        """Test that None is returned when all spreads are rejected due to strikes being too far."""
        from datetime import date as date_type
        
        atm_contract = OptionContractDTO(
            ticker='O:SPY240115P100',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(100.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        # OTM at $80 is way too far from any target (4pt=$96, 10pt=$90)
        otm_contract = OptionContractDTO(
            ticker='O:SPY240115P80',
            underlying_ticker='SPY',
            contract_type=OptionType.PUT,
            strike_price=StrikePrice(80.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        contracts = [atm_contract, otm_contract]
        
        def get_bar(contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]:
            if '100' in contract.ticker:
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('3.00'),
                    high_price=Decimal('3.10'),
                    low_price=Decimal('2.90'),
                    close_price=Decimal('3.00'),
                    volume=1000,
                    volume_weighted_avg_price=Decimal('3.00'),
                    number_of_transactions=100
                )
            else:
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('0.10'),
                    high_price=Decimal('0.20'),
                    low_price=Decimal('0.05'),
                    close_price=Decimal('0.10'),
                    volume=800,
                    volume_weighted_avg_price=Decimal('0.10'),
                    number_of_transactions=80
                )
        
        result = OptionsRetrieverHelper.find_best_credit_spread(
            contracts=contracts,
            current_price=100.0,
            expiration='2024-01-15',
            get_bar_fn=get_bar,
            date=datetime(2024, 1, 1),
            min_spread_width=4,
            max_spread_width=10,
            max_strike_difference=2.0,
            option_type=OptionType.PUT
        )
        
        # Should return None - all spreads rejected (strikes too far)
        assert result is None
    
    def test_find_best_credit_spread_works_with_calls(self):
        """Test that find_best_credit_spread works with call credit spreads."""
        from datetime import date as date_type
        
        # For call credit spreads, ATM is lower, OTM is higher
        atm_contract = OptionContractDTO(
            ticker='O:SPY240115C100',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(100.0),
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        otm_contract = OptionContractDTO(
            ticker='O:SPY240115C104',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(104.0),  # 4-point spread
            expiration_date=ExpirationDate(date_type(2024, 1, 15)),
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
        
        contracts = [atm_contract, otm_contract]
        
        def get_bar(contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]:
            if '100' in contract.ticker:  # ATM
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('3.00'),
                    high_price=Decimal('3.10'),
                    low_price=Decimal('2.90'),
                    close_price=Decimal('3.00'),
                    volume=1000,
                    volume_weighted_avg_price=Decimal('3.00'),
                    number_of_transactions=100
                )
            else:  # OTM at 104
                return OptionBarDTO(
                    ticker=contract.ticker,
                    timestamp=date,
                    open_price=Decimal('1.00'),
                    high_price=Decimal('1.10'),
                    low_price=Decimal('0.90'),
                    close_price=Decimal('1.00'),  # Credit: 3.00 - 1.00 = 2.00
                    volume=800,
                    volume_weighted_avg_price=Decimal('1.00'),
                    number_of_transactions=80
                )
        
        result = OptionsRetrieverHelper.find_best_credit_spread(
            contracts=contracts,
            current_price=100.0,
            expiration='2024-01-15',
            get_bar_fn=get_bar,
            date=datetime(2024, 1, 1),
            min_spread_width=4,
            max_spread_width=10,
            max_strike_difference=2.0,
            option_type=OptionType.CALL
        )
        
        # Should find the 4-point call credit spread
        assert result is not None
        assert result['width'] == 4.0  # 104 - 100
        assert result['credit'] == 2.0  # 3.00 - 1.00
        assert result['atm_contract'].ticker == 'O:SPY240115C100'
        assert result['otm_contract'].ticker == 'O:SPY240115C104'
