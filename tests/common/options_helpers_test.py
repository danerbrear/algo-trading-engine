"""
Tests for Phase 4 OptionsRetrieverHelper strategy-specific methods.

This module tests the new strategy-specific helper methods added in Phase 4
of the OptionsHandler refactoring.
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict

from algo_trading_engine.common.options_helpers import OptionsRetrieverHelper
from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO
from algo_trading_engine.vo import StrikePrice, ExpirationDate
from algo_trading_engine.common.models import OptionType, StrategyType
from algo_trading_engine.enums import BarTimeInterval


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


class TestCreditSpreadMaxCreditWidth:
    """Test cases for find_credit_spread_max_credit_width."""
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts with multiple strikes for spread testing."""
        future_date = date.today() + timedelta(days=30)
        contracts = []
        
        # Create PUT options with strikes from 595 to 610
        for strike in range(595, 611):
            contracts.append(
                OptionContractDTO(
                    ticker=f"O:SPY250929P00{strike}000",
                    underlying_ticker="SPY",
                    contract_type=OptionType.PUT,
                    strike_price=StrikePrice(float(strike)),
                    expiration_date=ExpirationDate(future_date),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS",
                    additional_underlyings=None
                )
            )
        
        # Create CALL options with strikes from 595 to 610
        for strike in range(595, 611):
            contracts.append(
                OptionContractDTO(
                    ticker=f"O:SPY250929C00{strike}000",
                    underlying_ticker="SPY",
                    contract_type=OptionType.CALL,
                    strike_price=StrikePrice(float(strike)),
                    expiration_date=ExpirationDate(future_date),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS",
                    additional_underlyings=None
                )
            )
        
        return contracts, str(future_date)
    
    def test_find_credit_spread_put_happy_path(self, sample_contracts):
        """Test finding best PUT credit spread with valid data."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Mock get_bar function that returns decreasing premiums for lower strikes
        def get_bar_fn(contract, date):
            strike = float(contract.strike_price.value)
            # ATM (600) has highest premium, decreases as we go OTM
            if contract.contract_type == OptionType.PUT:
                if strike == 600:
                    premium = Decimal('10.00')
                elif strike == 596:
                    premium = Decimal('7.00')
                elif strike == 595:
                    premium = Decimal('6.50')
                elif strike == 594:
                    premium = Decimal('6.00')
                else:
                    premium = Decimal('5.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.PUT
        )
        
        assert result is not None
        assert result['atm_contract'] is not None
        assert result['otm_contract'] is not None
        assert result['atm_contract'].contract_type == OptionType.PUT
        assert result['otm_contract'].contract_type == OptionType.PUT
        assert result['credit'] > 0
        assert result['width'] >= 4
        assert result['width'] <= 6
        assert result['credit_width_ratio'] > 0
        assert result['credit_width_ratio'] == result['credit'] / result['width']
    
    def test_find_credit_spread_call_happy_path(self, sample_contracts):
        """Test finding best CALL credit spread with valid data."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Mock get_bar function for CALL spreads
        def get_bar_fn(contract, date):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                if strike == 600:
                    premium = Decimal('10.00')
                elif strike == 604:
                    premium = Decimal('7.00')
                elif strike == 605:
                    premium = Decimal('6.50')
                elif strike == 606:
                    premium = Decimal('6.00')
                else:
                    premium = Decimal('5.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.CALL
        )
        
        assert result is not None
        assert result['atm_contract'].contract_type == OptionType.CALL
        assert result['otm_contract'].contract_type == OptionType.CALL
        assert result['credit'] > 0
        assert result['width'] >= 4
    
    def test_find_credit_spread_no_contracts(self):
        """Test with empty contract list."""
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            [], 600.0, "2025-09-29", lambda c, d: None, datetime.now()
        )
        
        assert result is None
    
    def test_find_credit_spread_no_bar_data(self, sample_contracts):
        """Test when bar data is unavailable."""
        contracts, expiration = sample_contracts
        
        # get_bar_fn always returns None
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            contracts, 600.0, expiration, lambda c, d: None, datetime.now()
        )
        
        assert result is None
    
    def test_find_credit_spread_negative_credit(self, sample_contracts):
        """Test when spread would result in negative credit (invalid)."""
        contracts, expiration = sample_contracts
        test_date = datetime.now()
        
        # Mock function that returns higher premium for OTM than ATM (invalid for credit spread)
        def get_bar_fn(contract, date):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.PUT:
                if strike == 600:  # ATM
                    premium = Decimal('5.00')
                else:  # OTM strikes have HIGHER premiums (unrealistic)
                    premium = Decimal('10.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            contracts, 600.0, expiration, get_bar_fn, test_date,
            option_type=OptionType.PUT
        )
        
        assert result is None
    
    def test_find_credit_spread_optimization(self, sample_contracts):
        """Test that the function selects the spread with highest credit/width ratio."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Create pricing that makes width=5 the optimal choice
        def get_bar_fn(contract, date):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.PUT:
                if strike == 600:
                    premium = Decimal('10.00')
                elif strike == 596:  # width=4, credit=10-7=3, ratio=0.75
                    premium = Decimal('7.00')
                elif strike == 595:  # width=5, credit=10-6=4, ratio=0.80 (best!)
                    premium = Decimal('6.00')
                elif strike == 594:  # width=6, credit=10-5.5=4.5, ratio=0.75
                    premium = Decimal('5.50')
                else:
                    premium = Decimal('5.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_credit_spread_max_credit_width(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.PUT
        )
        
        assert result is not None
        assert result['width'] == 5.0
        assert abs(result['credit_width_ratio'] - 0.80) < 0.01


class TestDebitSpreadMaxRewardRisk:
    """Test cases for find_debit_spread_max_reward_risk."""
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts with multiple strikes for spread testing."""
        future_date = date.today() + timedelta(days=30)
        contracts = []
        
        # Create CALL options with strikes from 595 to 610
        for strike in range(595, 611):
            contracts.append(
                OptionContractDTO(
                    ticker=f"O:SPY250929C00{strike}000",
                    underlying_ticker="SPY",
                    contract_type=OptionType.CALL,
                    strike_price=StrikePrice(float(strike)),
                    expiration_date=ExpirationDate(future_date),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS",
                    additional_underlyings=None
                )
            )
        
        # Create PUT options with strikes from 595 to 610
        for strike in range(595, 611):
            contracts.append(
                OptionContractDTO(
                    ticker=f"O:SPY250929P00{strike}000",
                    underlying_ticker="SPY",
                    contract_type=OptionType.PUT,
                    strike_price=StrikePrice(float(strike)),
                    expiration_date=ExpirationDate(future_date),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS",
                    additional_underlyings=None
                )
            )
        
        return contracts, str(future_date)
    
    def test_find_debit_spread_call_happy_path(self, sample_contracts):
        """Test finding best CALL debit spread with valid data."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Mock get_bar function - ITM/ATM has higher premium than OTM (accepts multiplier/timespan like get_option_bar)
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                if strike == 600:  # ATM (long)
                    premium = Decimal('10.00')
                elif strike == 604:  # OTM (short)
                    premium = Decimal('7.00')
                elif strike == 605:
                    premium = Decimal('6.00')
                elif strike == 606:
                    premium = Decimal('5.00')
                else:
                    premium = Decimal('4.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.CALL
        )
        
        assert result is not None
        # Returns a DebitSpreadPosition with spread_options from Option.from_contract_and_bar
        assert result.strategy_type == StrategyType.CALL_DEBIT_SPREAD
        assert result.entry_price > 0
        assert len(result.spread_options) == 2
        assert result.spread_options[0].symbol == "SPY"
        assert result.spread_options[1].symbol == "SPY"
        assert result.max_profit() > 0
        assert result.max_loss_per_share() == result.entry_price
        assert result.risk_reward_ratio() > 0
        assert result.spread_width() >= 4
        assert result.spread_width() <= 6
    
    def test_find_debit_spread_put_happy_path(self, sample_contracts):
        """Test finding best PUT debit spread with valid data."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Mock get_bar function for PUT debit spreads (accepts multiplier/timespan like get_option_bar)
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.PUT:
                if strike == 600:  # ATM (long)
                    premium = Decimal('10.00')
                elif strike == 596:  # OTM (short)
                    premium = Decimal('7.00')
                elif strike == 595:
                    premium = Decimal('6.00')
                elif strike == 594:
                    premium = Decimal('5.00')
                else:
                    premium = Decimal('4.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.PUT
        )
        
        assert result is not None
        assert result.strategy_type == StrategyType.PUT_DEBIT_SPREAD
        assert len(result.spread_options) == 2
        assert result.entry_price > 0
        assert result.max_profit() > 0

    def test_find_debit_spread_no_contracts(self):
        """Test with empty contract list."""
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            [], 600.0, "2025-09-29", lambda c, d: None, datetime.now()
        )
        
        assert result is None
    
    def test_find_debit_spread_no_bar_data(self, sample_contracts):
        """Test when bar data is unavailable."""
        contracts, expiration = sample_contracts

        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            return None

        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, 600.0, expiration, get_bar_fn, datetime.now()
        )
        
        assert result is None
    
    def test_find_debit_spread_negative_debit(self, sample_contracts):
        """Test when spread would result in negative debit (invalid)."""
        contracts, expiration = sample_contracts
        test_date = datetime.now()
        
        # Mock function that returns higher premium for OTM than ITM (invalid for debit spread)
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                # OTM has higher premium than ITM (unrealistic)
                premium = Decimal(str(strike))
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, 600.0, expiration, get_bar_fn, test_date,
            option_type=OptionType.CALL
        )
        
        assert result is None
    
    def test_find_debit_spread_zero_max_profit(self, sample_contracts):
        """Test when max profit would be zero or negative."""
        contracts, expiration = sample_contracts
        test_date = datetime.now()
        
        # Create scenario where debit >= spread width (no profit potential)
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                if strike == 600:
                    premium = Decimal('10.00')
                else:
                    # OTM options cost 6.00, making 4-point spread cost 4.00
                    # This leaves 0 max profit
                    premium = Decimal('6.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, 600.0, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=4, option_type=OptionType.CALL
        )
        
        assert result is None
    
    def test_find_debit_spread_optimization(self, sample_contracts):
        """Test that the function selects the spread with highest reward/risk ratio."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Create pricing that makes width=5 the optimal choice
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                if strike == 600:
                    premium = Decimal('10.00')
                elif strike == 604:  # width=4, debit=10-7=3, profit=1, ratio=0.33
                    premium = Decimal('7.00')
                elif strike == 605:  # width=5, debit=10-6=4, profit=1, ratio=0.25
                    premium = Decimal('6.00')
                elif strike == 606:  # width=6, debit=10-5=5, profit=1, ratio=0.20
                    premium = Decimal('5.00')
                else:
                    premium = Decimal('4.00')
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.CALL
        )
        
        assert result is not None
        # Should select width=4 because it has the best (lowest) risk/reward ratio
        # width=4: debit=3, max_profit=1, max_loss=3 -> risk_reward_ratio = max_loss/max_profit = 3.0
        assert result.spread_width() == 4.0
        assert abs(result.risk_reward_ratio() - 3.0) < 0.01

    def test_find_debit_spread_realistic_scenario(self, sample_contracts):
        """Test with realistic option pricing scenario."""
        contracts, expiration = sample_contracts
        current_price = 600.0
        test_date = datetime.now()
        
        # Realistic pricing with time decay
        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                # Intrinsic value + time value
                intrinsic = max(0, current_price - strike)
                time_value = 2.0 * (1.0 / (1.0 + abs(strike - current_price) * 0.1))
                premium = Decimal(str(intrinsic + time_value))
            else:
                return None
            
            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )
        
        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, current_price, expiration, get_bar_fn, test_date,
            min_spread_width=4, max_spread_width=10, option_type=OptionType.CALL
        )
        
        assert result is not None
        assert result.entry_price > 0
        assert result.max_profit() > 0
        assert result.max_loss_per_share() > 0
        assert result.risk_reward_ratio() > 0
        # Verify relationship: max_profit = width - debit
        assert abs(result.max_profit() - (result.spread_width() - result.entry_price)) < 0.01

    @pytest.mark.parametrize("bar_datetime,timespan", [
        (datetime(2025, 1, 10), BarTimeInterval.DAY),
        (datetime(2025, 1, 10, 9, 30), BarTimeInterval.HOUR),
        (datetime(2025, 1, 10, 9, 31), BarTimeInterval.MINUTE),
    ])
    def test_find_debit_spread_works_with_any_bar_interval(self, sample_contracts, bar_datetime, timespan):
        """Validate find_debit_spread_max_reward_risk returns a debit spread for bars from any time interval."""
        contracts, expiration = sample_contracts
        current_price = 600.0

        def get_bar_fn(contract, date, multiplier=1, timespan=None):
            strike = float(contract.strike_price.value)
            if contract.contract_type == OptionType.CALL:
                if strike == 600:
                    premium = Decimal('10.00')
                elif strike == 604:
                    premium = Decimal('7.00')
                elif strike == 605:
                    premium = Decimal('6.00')
                elif strike == 606:
                    premium = Decimal('5.00')
                else:
                    premium = Decimal('4.00')
            else:
                return None

            return OptionBarDTO(
                ticker=contract.ticker,
                timestamp=date,
                open_price=premium,
                high_price=premium,
                low_price=premium,
                close_price=premium,
                volume=100,
                volume_weighted_avg_price=premium,
                number_of_transactions=10,
                adjusted=True
            )

        result = OptionsRetrieverHelper.find_debit_spread_max_reward_risk(
            contracts, current_price, expiration, get_bar_fn, bar_datetime,
            min_spread_width=4, max_spread_width=6, option_type=OptionType.CALL,
            timespan=timespan
        )

        assert result is not None
        assert result.strategy_type == StrategyType.CALL_DEBIT_SPREAD
        assert result.entry_date == bar_datetime
        assert result.entry_price > 0
        assert len(result.spread_options) == 2
        assert result.max_profit() > 0
        assert result.max_loss_per_share() == result.entry_price
