"""
Unit tests for Phase 2 OptionsHandler refactoring.

Tests the new caching infrastructure, migration utility, and helper classes.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path

from src.common.options_handler import OptionsHandler
from src.common.cache.options_cache_manager import OptionsCacheManager
from src.common.options_cache_migration import OptionsCacheMigrator
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikePrice, ExpirationDate
)
from src.common.models import OptionType


class TestOptionsCacheManager:
    """Test cases for OptionsCacheManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_dir):
        """Create cache manager with temporary directory."""
        return OptionsCacheManager(temp_dir)
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts for testing."""
        # Use future dates to avoid validation errors
        future_date = date.today() + timedelta(days=30)
        return [
            OptionContractDTO(
                ticker="O:SPY211119C00045000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY211119P00045000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY211119C00046000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('460.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
    
    @pytest.fixture
    def sample_bar(self):
        """Create sample bar for testing."""
        return OptionBarDTO(
            ticker="O:SPY211119C00045000",
            timestamp=datetime(2021, 11, 19, 16, 0, 0),
            open_price=Decimal('5.50'),
            high_price=Decimal('5.75'),
            low_price=Decimal('5.25'),
            close_price=Decimal('5.60'),
            volume=1000,
            volume_weighted_avg_price=Decimal('5.55'),
            number_of_transactions=50,
            adjusted=True
        )
    
    def test_cache_paths(self, cache_manager):
        """Test cache path generation."""
        test_date = date(2021, 11, 19)
        
        contracts_path = cache_manager.get_contracts_cache_path("SPY", test_date)
        expected_contracts = Path(cache_manager.base_dir) / 'options' / 'SPY' / '2021-11-19' / 'contracts.pkl'
        assert contracts_path == expected_contracts
        
        bars_path = cache_manager.get_bars_cache_path("SPY", test_date, "O:SPY211119C00045000")
        expected_bars = Path(cache_manager.base_dir) / 'options' / 'SPY' / '2021-11-19' / 'bars' / 'O:SPY211119C00045000.pkl'
        assert bars_path == expected_bars
    
    def test_save_and_load_contracts(self, cache_manager, sample_contracts):
        """Test saving and loading contracts."""
        test_date = date(2021, 11, 19)
        
        # Save contracts
        cache_manager.save_contracts("SPY", test_date, sample_contracts)
        
        # Load contracts
        loaded_contracts = cache_manager.load_contracts("SPY", test_date)
        
        assert loaded_contracts is not None
        assert len(loaded_contracts) == 3
        assert loaded_contracts[0].ticker == "O:SPY211119C00045000"
        assert loaded_contracts[1].ticker == "O:SPY211119P00045000"
    
    def test_save_and_load_bar(self, cache_manager, sample_bar):
        """Test saving and loading bar data."""
        test_date = date(2021, 11, 19)
        ticker = "O:SPY211119C00045000"
        
        # Save bar
        cache_manager.save_bar("SPY", test_date, ticker, sample_bar)
        
        # Load bar
        loaded_bar = cache_manager.load_bar("SPY", test_date, ticker)
        
        assert loaded_bar is not None
        assert loaded_bar.ticker == ticker
        assert loaded_bar.close_price == Decimal('5.60')
        assert loaded_bar.volume == 1000
    
    def test_load_nonexistent_contracts(self, cache_manager):
        """Test loading non-existent contracts."""
        test_date = date(2021, 11, 19)
        contracts = cache_manager.load_contracts("SPY", test_date)
        assert contracts is None
    
    def test_load_nonexistent_bar(self, cache_manager):
        """Test loading non-existent bar."""
        test_date = date(2021, 11, 19)
        bar = cache_manager.load_bar("SPY", test_date, "O:SPY211119C00045000")
        assert bar is None
    
    def test_contract_exists(self, cache_manager, sample_contracts):
        """Test contract existence check."""
        test_date = date(2021, 11, 19)
        
        # Initially should not exist
        assert not cache_manager.contract_exists("SPY", test_date, "O:SPY211119C00045000")
        
        # Save contracts
        cache_manager.save_contracts("SPY", test_date, sample_contracts)
        
        # Should exist now
        assert cache_manager.contract_exists("SPY", test_date, "O:SPY211119C00045000")
        assert cache_manager.contract_exists("SPY", test_date, "O:SPY211119P00045000")
        assert cache_manager.contract_exists("SPY", test_date, "O:SPY211119C00046000")
    
    def test_bar_exists(self, cache_manager, sample_bar):
        """Test bar existence check."""
        test_date = date(2021, 11, 19)
        ticker = "O:SPY211119C00045000"
        
        # Initially should not exist
        assert not cache_manager.bar_exists("SPY", test_date, ticker)
        
        # Save bar
        cache_manager.save_bar("SPY", test_date, ticker, sample_bar)
        
        # Should exist now
        assert cache_manager.bar_exists("SPY", test_date, ticker)
    
    def test_get_cached_counts(self, cache_manager, sample_contracts, sample_bar):
        """Test getting cached counts."""
        test_date = date(2021, 11, 19)
        
        # Initially should be 0
        assert cache_manager.get_cached_contracts_count("SPY", test_date) == 0
        assert cache_manager.get_cached_bars_count("SPY", test_date) == 0
        
        # Save contracts
        cache_manager.save_contracts("SPY", test_date, sample_contracts)
        assert cache_manager.get_cached_contracts_count("SPY", test_date) == 3
        
        # Save bar
        cache_manager.save_bar("SPY", test_date, "O:SPY211119C00045000", sample_bar)
        assert cache_manager.get_cached_bars_count("SPY", test_date) == 1
    
    def test_list_available_dates(self, cache_manager, sample_contracts):
        """Test listing available dates."""
        # Initially should be empty
        dates = cache_manager.list_available_dates("SPY")
        assert dates == []
        
        # Save contracts for different dates
        date1 = date(2021, 11, 19)
        date2 = date(2021, 11, 20)
        
        cache_manager.save_contracts("SPY", date1, sample_contracts)
        cache_manager.save_contracts("SPY", date2, sample_contracts)
        
        # Should list both dates
        dates = cache_manager.list_available_dates("SPY")
        assert len(dates) == 2
        assert date1 in dates
        assert date2 in dates
        assert dates == sorted(dates)  # Should be sorted


class TestOptionsHandler:
    """Test cases for refactored OptionsHandler."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def options_handler(self, temp_dir):
        """Create options handler with temporary directory."""
        return OptionsHandler("SPY", cache_dir=temp_dir)
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts for testing."""
        # Use future dates to avoid validation errors
        future_date = date.today() + timedelta(days=30)
        return [
            OptionContractDTO(
                ticker="O:SPY211119C00045000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
    
    def test_initialization(self, temp_dir):
        """Test OptionsHandler initialization."""
        handler = OptionsHandler("SPY", cache_dir=temp_dir)
        assert handler.symbol == "SPY"
        assert handler.cache_manager.base_dir == Path(temp_dir)
    
    def test_initialization_without_api_key(self, temp_dir):
        """Test initialization without API key raises error."""
        import os
        # Temporarily unset the API key environment variable
        original_key = os.environ.get('POLYGON_API_KEY')
        if 'POLYGON_API_KEY' in os.environ:
            del os.environ['POLYGON_API_KEY']
        
        try:
            with pytest.raises(ValueError, match="Polygon.io API key is required"):
                OptionsHandler("SPY", api_key=None, cache_dir=temp_dir)
        finally:
            # Restore the original API key
            if original_key:
                os.environ['POLYGON_API_KEY'] = original_key
    
    def test_get_contract_list_for_date_empty(self, options_handler):
        """Test getting contract list when no cached data exists."""
        test_date = datetime(2021, 11, 19)
        contracts = options_handler.get_contract_list_for_date(test_date)
        assert contracts == []
    
    def test_get_contract_list_for_date_with_data(self, options_handler, sample_contracts):
        """Test getting contract list with cached data."""
        test_date = datetime(2021, 11, 19)
        
        # Cache contracts using private method (for testing purposes)
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Get contracts
        contracts = options_handler.get_contract_list_for_date(test_date)
        assert len(contracts) == 1
        assert contracts[0].ticker == "O:SPY211119C00045000"
    
    def test_get_option_bar_empty(self, options_handler, sample_contracts):
        """Test getting option bar when no cached data exists."""
        test_date = datetime(2021, 11, 19)
        contract = sample_contracts[0]
        
        bar = options_handler.get_option_bar(contract, test_date)
        assert bar is None
    
    def test_get_options_chain(self, options_handler, sample_contracts):
        """Test getting complete options chain."""
        test_date = datetime(2021, 11, 19)
        current_price = 450.0
        
        # Cache contracts using private method (for testing purposes)
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Get chain
        chain = options_handler.get_options_chain(test_date, current_price)
        
        assert chain.underlying_symbol == "SPY"
        assert chain.current_price == current_price
        assert chain.date == test_date.date()
        assert len(chain.contracts) == 1
        assert len(chain.bars) == 0  # No bars cached yet
    
    def test_cache_stats(self, options_handler, sample_contracts):
        """Test getting cache statistics."""
        test_date = datetime(2021, 11, 19)
        
        # Initially should be 0
        stats = options_handler._get_cache_stats(test_date)
        assert stats['contracts_count'] == 0
        assert stats['bars_count'] == 0
        
        # Cache contracts using private method (for testing purposes)
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Should show cached contracts
        stats = options_handler._get_cache_stats(test_date)
        assert stats['contracts_count'] == 1
        assert stats['bars_count'] == 0
    
    def test_private_method_enforcement(self, options_handler):
        """Test that private methods can be accessed during testing but are blocked in production."""
        test_date = datetime(2021, 11, 19)
        
        # During testing, private methods should be accessible (for test setup)
        # This allows tests to set up cache data for testing public methods
        try:
            options_handler._cache_contracts(test_date, [])
            options_handler._cache_bar(test_date, "test_ticker", None)
            options_handler._get_cache_stats(test_date)
            print("✅ Private methods accessible during testing (as expected)")
        except AttributeError as e:
            pytest.fail(f"Private methods should be accessible during testing: {e}")
        
        # Note: The actual enforcement happens in production code outside of test files
        # This is demonstrated in the demo_private_methods.py script


class TestOptionsRetrieverHelper:
    """Test cases for OptionsRetrieverHelper."""
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts for testing."""
        # Use future dates to avoid validation errors
        future_date = date.today() + timedelta(days=30)
        return [
            OptionContractDTO(
                ticker="O:SPY211119C00045000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY211119P00045000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY211119C00046000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(460.0),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
    
    def test_filter_contracts_by_strike(self, sample_contracts):
        """Test filtering contracts by strike."""
        filtered = OptionsRetrieverHelper.filter_contracts_by_strike(
            sample_contracts, 450.0, 5.0
        )
        assert len(filtered) == 2  # Two contracts at 450 strike
        
        filtered = OptionsRetrieverHelper.filter_contracts_by_strike(
            sample_contracts, 450.0, 0.1
        )
        assert len(filtered) == 2  # Two contracts at exactly 450 strike
    
    def test_find_atm_contracts(self, sample_contracts):
        """Test finding ATM contracts."""
        call_contract, put_contract = OptionsRetrieverHelper.find_atm_contracts(
            sample_contracts, 450.0
        )
        
        assert call_contract is not None
        assert put_contract is not None
        assert call_contract.contract_type == OptionType.CALL
        assert put_contract.contract_type == OptionType.PUT
        assert call_contract.strike_price.value == Decimal('450.0')
        assert put_contract.strike_price.value == Decimal('450.0')
    
    
    def test_find_itm_contracts(self, sample_contracts):
        """Test finding ITM contracts."""
        # With current price 455, 450 call should be ITM, 450 put should be OTM
        itm_contracts = OptionsRetrieverHelper.find_itm_contracts(sample_contracts, 455.0)
        
        assert len(itm_contracts) == 1
        assert itm_contracts[0].ticker == "O:SPY211119C00045000"
    
    def test_find_otm_contracts(self, sample_contracts):
        """Test finding OTM contracts."""
        # With current price 455, 450 put should be OTM, 460 call should be OTM
        otm_contracts = OptionsRetrieverHelper.find_otm_contracts(sample_contracts, 455.0)
        
        assert len(otm_contracts) == 2
        tickers = [c.ticker for c in otm_contracts]
        assert "O:SPY211119P00045000" in tickers
        assert "O:SPY211119C00046000" in tickers
    
    def test_sort_contracts_by_strike(self, sample_contracts):
        """Test sorting contracts by strike."""
        sorted_asc = OptionsRetrieverHelper.sort_contracts_by_strike(sample_contracts, True)
        sorted_desc = OptionsRetrieverHelper.sort_contracts_by_strike(sample_contracts, False)
        
        assert sorted_asc[0].strike_price.value == Decimal('450.0')
        assert sorted_asc[-1].strike_price.value == Decimal('460.0')
        
        assert sorted_desc[0].strike_price.value == Decimal('460.0')
        assert sorted_desc[-1].strike_price.value == Decimal('450.0')
    
    def test_find_closest_strike_contract(self, sample_contracts):
        """Test finding closest strike contract."""
        closest = OptionsRetrieverHelper.find_closest_strike_contract(sample_contracts, 455.0)
        assert closest is not None
        assert closest.strike_price.value == Decimal('450.0')
        
        closest = OptionsRetrieverHelper.find_closest_strike_contract(sample_contracts, 465.0)
        assert closest is not None
        assert closest.strike_price.value == Decimal('460.0')
    
    def test_find_closest_strike_contract_empty(self):
        """Test finding closest strike with empty list."""
        closest = OptionsRetrieverHelper.find_closest_strike_contract([], 450.0)
        assert closest is None
    
    def test_calculate_contract_statistics(self, sample_contracts):
        """Test calculating contract statistics."""
        stats = OptionsRetrieverHelper.calculate_contract_statistics(sample_contracts)
        
        assert stats['total_contracts'] == 3
        assert stats['calls'] == 2
        assert stats['puts'] == 1
        assert stats['unique_strikes'] == 2
        assert stats['unique_expirations'] == 1
        assert stats['strike_range'] == (450.0, 460.0)
    
    def test_calculate_contract_statistics_empty(self):
        """Test calculating statistics for empty list."""
        stats = OptionsRetrieverHelper.calculate_contract_statistics([])
        
        assert stats['total_contracts'] == 0
        assert stats['calls'] == 0
        assert stats['puts'] == 0
        assert stats['unique_strikes'] == 0
        assert stats['unique_expirations'] == 0
        assert stats['strike_range'] is None
    
    def test_validate_contract_data(self, sample_contracts):
        """Test contract data validation."""
        valid_contract = sample_contracts[0]
        issues = OptionsRetrieverHelper.validate_contract_data(valid_contract)
        assert len(issues) == 0
        
        # Test invalid ticker by creating a mock contract with invalid data
        # Since we can't create an invalid OptionContractDTO due to validation,
        # we'll test the validation logic directly
        from unittest.mock import Mock
        
        invalid_contract = Mock()
        invalid_contract.ticker = "INVALID_TICKER"
        invalid_contract.underlying_ticker = "SPY"
        invalid_contract.contract_type = OptionType.CALL
        invalid_contract.strike_price = Mock()
        invalid_contract.strike_price.value = 450.0
        invalid_contract.expiration_date = Mock()
        invalid_contract.expiration_date.date = date.today() + timedelta(days=30)
        invalid_contract.shares_per_contract = 100
        
        issues = OptionsRetrieverHelper.validate_contract_data(invalid_contract)
        assert len(issues) > 0
        assert any("Invalid ticker format" in issue for issue in issues)


class TestOptionsRetrieverHelperStrategy:
    """Test strategy-specific helper methods."""
    
    def _create_test_contracts(self):
        """Create test contracts for strategy testing."""
        # Create a proper monthly expiration date (third Friday of next month)
        today = date.today()
        # Get first day of next month
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        # Find third Friday of next month
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        future_date = third_friday
        return [
            OptionContractDTO(
                ticker="O:SPY250929C00580000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929C00585000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal('455.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929P00450000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal('450.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            ),
            OptionContractDTO(
                ticker="O:SPY250929P00445000",
                underlying_ticker="SPY",
                contract_type=OptionType.PUT,
                strike_price=StrikePrice(Decimal('445.0')),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
    
    def test_find_credit_spread_legs_call(self):
        """Test finding call credit spread legs."""
        contracts = self._create_test_contracts()
        current_price = 450.0
        # Use the same expiration date as the test contracts
        today = date.today()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expiration_date = third_friday.strftime('%Y-%m-%d')
        
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            contracts, current_price, expiration_date, OptionType.CALL, spread_width=5
        )
        
        assert short_leg is not None
        assert long_leg is not None
        assert short_leg.contract_type == OptionType.CALL
        assert long_leg.contract_type == OptionType.CALL
        assert short_leg.strike_price.value <= Decimal(str(current_price))
        assert long_leg.strike_price.value > short_leg.strike_price.value
    
    def test_find_credit_spread_legs_put(self):
        """Test finding put credit spread legs."""
        contracts = self._create_test_contracts()
        current_price = 450.0
        # Use the same expiration date as the test contracts
        today = date.today()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expiration_date = third_friday.strftime('%Y-%m-%d')
        
        short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
            contracts, current_price, expiration_date, OptionType.PUT, spread_width=5
        )
        
        assert short_leg is not None
        assert long_leg is not None
        assert short_leg.contract_type == OptionType.PUT
        assert long_leg.contract_type == OptionType.PUT
        assert short_leg.strike_price.value >= Decimal(str(current_price))
        assert long_leg.strike_price.value < short_leg.strike_price.value
    
    def test_calculate_credit_spread_premium(self):
        """Test calculating credit spread premium."""
        contracts = self._create_test_contracts()
        short_leg = contracts[0]
        long_leg = contracts[1]
        short_premium = 2.50
        long_premium = 1.00
        
        net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
            short_leg, long_leg, short_premium, long_premium
        )
        
        assert net_credit == 1.50
    
    def test_calculate_max_profit_loss(self):
        """Test calculating max profit and loss."""
        contracts = self._create_test_contracts()
        short_leg = contracts[0]
        long_leg = contracts[1]
        net_credit = 1.50
        
        max_profit, max_loss = OptionsRetrieverHelper.calculate_max_profit_loss(
            short_leg, long_leg, net_credit
        )
        
        assert max_profit == 1.50
        assert max_loss > 0
    
    def test_find_optimal_expiration(self):
        """Test finding optimal expiration date."""
        contracts = self._create_test_contracts()
        
        optimal_exp = OptionsRetrieverHelper.find_optimal_expiration(
            contracts, min_days=20, max_days=40
        )
        
        assert optimal_exp is not None
        # Should match the expiration date of our test contracts
        today = date.today()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expected_date = third_friday.strftime('%Y-%m-%d')
        assert optimal_exp == expected_date
    
    def test_calculate_implied_volatility_rank(self):
        """Test calculating IV rank."""
        contracts = self._create_test_contracts()
        current_price = 450.0
        
        iv_ranks = OptionsRetrieverHelper.calculate_implied_volatility_rank(
            contracts, current_price, lookback_days=30
        )
        
        assert len(iv_ranks) == len(contracts)
        for ticker, rank in iv_ranks.items():
            assert 0.0 <= rank <= 100.0
    
    def test_find_high_volume_contracts(self):
        """Test finding high volume contracts."""
        contracts = self._create_test_contracts()
        bars = self._create_test_bars()
        
        high_volume = OptionsRetrieverHelper.find_high_volume_contracts(
            contracts, bars, min_volume=100
        )
        
        assert len(high_volume) > 0
        for contract in high_volume:
            bar = bars.get(contract.ticker)
            assert bar.volume >= 100
    
    def test_calculate_delta_exposure(self):
        """Test calculating delta exposure."""
        contracts = self._create_test_contracts()
        bars = self._create_test_bars()
        
        total_delta = OptionsRetrieverHelper.calculate_delta_exposure(
            contracts, bars, quantity=1
        )
        
        assert isinstance(total_delta, float)
    
    def test_find_iron_condor_legs(self):
        """Test finding iron condor legs."""
        contracts = self._create_test_contracts()
        current_price = 450.0
        # Use the same expiration date as the test contracts
        today = date.today()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expiration_date = third_friday.strftime('%Y-%m-%d')
        
        put_long, put_short, call_short, call_long = OptionsRetrieverHelper.find_iron_condor_legs(
            contracts, current_price, expiration_date, spread_width=5
        )
        
        # Should find all four legs
        assert all([put_long, put_short, call_short, call_long])
        assert put_long.contract_type == OptionType.PUT
        assert put_short.contract_type == OptionType.PUT
        assert call_short.contract_type == OptionType.CALL
        assert call_long.contract_type == OptionType.CALL
    
    def test_calculate_breakeven_points(self):
        """Test calculating breakeven points."""
        contracts = self._create_test_contracts()
        short_leg = contracts[0]
        long_leg = contracts[1]
        net_credit = 1.50
        
        lower_be, upper_be = OptionsRetrieverHelper.calculate_breakeven_points(
            short_leg, long_leg, net_credit, OptionType.CALL
        )
        
        assert lower_be == upper_be  # For credit spreads, breakeven is single point
        assert lower_be > 0
    
    def test_find_weekly_expirations(self):
        """Test finding weekly expirations."""
        contracts = self._create_test_contracts()
        
        weekly_exps = OptionsRetrieverHelper.find_weekly_expirations(contracts)
        
        assert isinstance(weekly_exps, list)
        # Our test contracts are monthly (third Friday), which is also a Friday (weekly)
        # So it should be detected as both weekly and monthly
        assert len(weekly_exps) > 0
    
    def test_find_monthly_expirations(self):
        """Test finding monthly expirations."""
        contracts = self._create_test_contracts()
        
        monthly_exps = OptionsRetrieverHelper.find_monthly_expirations(contracts)
        
        assert isinstance(monthly_exps, list)
        assert len(monthly_exps) > 0
        # Should match the expiration date of our test contracts
        # The test contracts use third Friday of next month
        today = date.today()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_day = next_month
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expected_date = third_friday.strftime('%Y-%m-%d')
        assert expected_date in monthly_exps
    
    def test_calculate_probability_of_profit(self):
        """Test calculating probability of profit."""
        contracts = self._create_test_contracts()
        short_leg = contracts[0]
        long_leg = contracts[1]
        net_credit = 1.50
        current_price = 450.0
        days_to_expiration = 30
        
        pop = OptionsRetrieverHelper.calculate_probability_of_profit(
            short_leg, long_leg, net_credit, OptionType.CALL, 
            current_price, days_to_expiration
        )
        
        assert 0.0 <= pop <= 1.0
        assert pop > 0.0  # Should have some probability of profit
    
    def _create_test_bars(self):
        """Create test bar data."""
        return {
            "O:SPY250929C00580000": OptionBarDTO(
                ticker="O:SPY250929C00580000",
                timestamp=datetime.now(),
                open_price=Decimal("2.50"),
                high_price=Decimal("2.75"),
                low_price=Decimal("2.25"),
                close_price=Decimal("2.60"),
                volume=150,
                volume_weighted_avg_price=Decimal("2.55"),
                number_of_transactions=25,
                adjusted=True
            ),
            "O:SPY250929C00585000": OptionBarDTO(
                ticker="O:SPY250929C00585000",
                timestamp=datetime.now(),
                open_price=Decimal("1.80"),
                high_price=Decimal("2.00"),
                low_price=Decimal("1.60"),
                close_price=Decimal("1.90"),
                volume=200,
                volume_weighted_avg_price=Decimal("1.85"),
                number_of_transactions=30,
                adjusted=True
            )
        }


class TestOptionsCacheMigrator:
    """Test cases for OptionsCacheMigrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def migrator(self, temp_dir):
        """Create migrator with temporary directory."""
        return OptionsCacheMigrator(temp_dir)
    
    def test_discover_old_cache_files_empty(self, migrator):
        """Test discovering cache files when none exist."""
        files = migrator.discover_old_cache_files("SPY")
        assert files == {}
    
    def test_migrate_symbol_no_files(self, migrator):
        """Test migrating symbol with no files."""
        stats = migrator.migrate_symbol("SPY")
        assert stats['dates_processed'] == 0
        assert stats['contracts_migrated'] == 0
        assert stats['bars_migrated'] == 0
    
    def test_migrate_all_symbols_empty(self, migrator):
        """Test migrating all symbols when none exist."""
        stats = migrator.migrate_all_symbols()
        assert stats == {}
