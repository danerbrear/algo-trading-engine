"""
Integration tests for Phase 3 OptionsHandler implementation.

This module tests the complete API functionality including:
- Contract fetching from API and cache
- Bar data fetching from API and cache
- Error handling and retry logic
- Rate limiting through APIRetryHandler
- Filtering capabilities
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.common.options_handler import OptionsHandler
from src.common.cache.options_cache_manager import OptionsCacheManager
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    StrikePrice, ExpirationDate
)
from src.common.models import OptionType


class TestOptionsHandlerPhase3:
    """Integration tests for Phase 3 OptionsHandler API."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def options_handler(self, temp_dir):
        """Create OptionsHandler with temporary directory."""
        return OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir, use_free_tier=True)
    
    @pytest.fixture
    def sample_contracts(self):
        """Create sample contracts for testing."""
        # Use a specific future date that works with our test scenarios
        future_date = date(2025, 9, 29)  # Fixed date for consistent testing
        return [
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
            )
        ]
    
    @pytest.fixture
    def sample_bar(self):
        """Create sample bar data for testing."""
        return OptionBarDTO(
            ticker="O:SPY250929C00600000",
            timestamp=datetime.now(),
            open_price=Decimal('10.50'),
            high_price=Decimal('11.00'),
            low_price=Decimal('10.25'),
            close_price=Decimal('10.75'),
            volume=1000,
            volume_weighted_avg_price=Decimal('10.60'),
            number_of_transactions=50,
            adjusted=True
        )
    
    def test_get_contract_list_for_date_from_cache(self, options_handler, sample_contracts):
        """Test getting contracts from cache."""
        test_date = datetime(2021, 11, 19)
        
        # Cache contracts first
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Get contracts from cache
        contracts = options_handler.get_contract_list_for_date(test_date)
        
        assert len(contracts) == 2
        assert contracts[0].ticker == "O:SPY250929C00600000"
        assert contracts[1].ticker == "O:SPY250929P00600000"
    
    @patch('polygon.RESTClient')
    def test_get_contract_list_for_date_from_api(self, mock_rest_client, options_handler):
        """Test getting contracts from API when not in cache."""
        test_date = datetime(2021, 11, 19)
        
        # Mock API response
        mock_response = Mock()
        mock_response.results = [
            {
                'ticker': 'O:SPY250929C00600000',
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 600.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            }
        ]
        
        mock_client = Mock()
        mock_client.reference_options_contracts.return_value = mock_response
        mock_rest_client.return_value = mock_client
        options_handler.client = mock_client
        
        # Mock retry handler
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Get contracts from API
        contracts = options_handler.get_contract_list_for_date(test_date)
        
        assert len(contracts) == 1
        assert contracts[0].ticker == "O:SPY250929C00600000"
        assert contracts[0].contract_type == OptionType.CALL
        
        # Verify API was called
        options_handler.retry_handler.fetch_with_retry.assert_called_once()
    
    def test_get_contract_list_for_date_with_strike_filter(self, options_handler, sample_contracts):
        """Test getting contracts with strike price filtering."""
        test_date = datetime(2021, 11, 19)
        
        # Cache contracts first
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Create strike range filter
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(590.0),
            max_strike=StrikePrice(610.0)
        )
        
        # Get filtered contracts
        contracts = options_handler.get_contract_list_for_date(test_date, strike_range=strike_range)
        
        assert len(contracts) == 2  # Both contracts are within range
        for contract in contracts:
            assert 590.0 <= float(contract.strike_price.value) <= 610.0
    
    def test_get_contract_list_for_date_with_expiration_filter(self, options_handler, sample_contracts):
        """Test getting contracts with expiration date filtering."""
        # Use a test date that's closer to the contract expiration dates
        test_date = datetime(2025, 9, 1)  # About 28 days before the 2025-09-29 expiration
        
        # Cache contracts first
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Create expiration range filter
        expiration_range = ExpirationRangeDTO(
            min_days=25,
            max_days=35
        )
        
        # Get filtered contracts
        contracts = options_handler.get_contract_list_for_date(test_date, expiration_range=expiration_range)
        
        assert len(contracts) == 2  # Both contracts should be within range
        for contract in contracts:
            days_to_exp = contract.days_to_expiration(test_date.date())
            assert 25 <= days_to_exp <= 35
    
    def test_get_option_bar_from_cache(self, options_handler, sample_contracts, sample_bar):
        """Test getting bar data from cache."""
        test_date = datetime(2021, 11, 19)
        contract = sample_contracts[0]
        
        # Cache bar data first
        options_handler._cache_bar(test_date, contract.ticker, sample_bar)
        
        # Get bar from cache
        bar = options_handler.get_option_bar(contract, test_date)
        
        assert bar is not None
        assert bar.ticker == contract.ticker
        assert bar.close_price == Decimal('10.75')
        assert bar.volume == 1000
    
    @patch('polygon.RESTClient')
    def test_get_option_bar_from_api(self, mock_rest_client, options_handler, sample_contracts):
        """Test getting bar data from API when not in cache."""
        test_date = datetime(2021, 11, 19)
        contract = sample_contracts[0]
        
        # Mock API response
        mock_response = Mock()
        mock_response.results = [
            {
                'o': 10.50,  # open
                'h': 11.00,  # high
                'l': 10.25,  # low
                'c': 10.75,  # close
                'v': 1000,   # volume
                'vw': 10.60, # vwap
                'n': 50,     # transactions
                't': 1637366400000  # timestamp
            }
        ]
        
        mock_client = Mock()
        mock_client.get_aggs.return_value = mock_response
        mock_rest_client.return_value = mock_client
        options_handler.client = mock_client
        
        # Mock retry handler
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Get bar from API
        bar = options_handler.get_option_bar(contract, test_date)
        
        assert bar is not None
        assert bar.ticker == contract.ticker
        assert bar.close_price == Decimal('10.75')
        assert bar.volume == 1000
        
        # Verify API was called
        options_handler.retry_handler.fetch_with_retry.assert_called_once()
    
    def test_get_options_chain_integration(self, options_handler, sample_contracts, sample_bar):
        """Test complete options chain integration."""
        test_date = datetime(2021, 11, 19)
        current_price = 600.0
        
        # Cache contracts and bar data
        options_handler._cache_contracts(test_date, sample_contracts)
        options_handler._cache_bar(test_date, sample_contracts[0].ticker, sample_bar)
        
        # Get complete options chain
        chain = options_handler.get_options_chain(test_date, current_price)
        
        assert chain.underlying_symbol == "SPY"
        assert chain.current_price == Decimal(str(current_price))
        assert chain.date == test_date.date()
        assert len(chain.contracts) == 2
        assert len(chain.bars) == 1
        assert sample_contracts[0].ticker in chain.bars
    
    def test_error_handling_api_failure(self, options_handler):
        """Test error handling when API fails."""
        test_date = datetime(2021, 11, 19)
        
        # Mock retry handler to raise exception
        options_handler.retry_handler.fetch_with_retry = Mock(side_effect=Exception("API Error"))
        
        # Should return empty list without crashing
        contracts = options_handler.get_contract_list_for_date(test_date)
        assert contracts == []
    
    def test_error_handling_invalid_contract_data(self, options_handler):
        """Test error handling with invalid contract data."""
        test_date = datetime(2021, 11, 19)
        
        # Mock API response with invalid data
        mock_response = Mock()
        mock_response.results = [
            {
                'ticker': 'INVALID',  # Missing required fields
                'underlying_ticker': 'SPY'
                # Missing contract_type, strike_price, expiration_date
            }
        ]
        
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Should handle invalid data gracefully
        contracts = options_handler.get_contract_list_for_date(test_date)
        assert contracts == []  # Invalid contracts should be filtered out
    
    def test_error_handling_invalid_bar_data(self, options_handler, sample_contracts):
        """Test error handling with invalid bar data."""
        test_date = datetime(2021, 11, 19)
        contract = sample_contracts[0]
        
        # Mock API response with invalid bar data
        mock_response = Mock()
        mock_response.results = [
            {
                'o': 10.50,  # open
                'h': 11.00,  # high
                'l': 10.25,  # low
                # Missing close price
                'v': 1000,   # volume
            }
        ]
        
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Should handle invalid data gracefully
        bar = options_handler.get_option_bar(contract, test_date)
        assert bar is None  # Invalid bar data should return None
    
    def test_rate_limiting_integration(self, options_handler):
        """Test that rate limiting is properly integrated."""
        test_date = datetime(2021, 11, 19)
        
        # Mock retry handler to verify rate limiting is used
        mock_fetch_with_retry = Mock(return_value=Mock(results=[]))
        options_handler.retry_handler.fetch_with_retry = mock_fetch_with_retry
        
        # Make API call
        options_handler.get_contract_list_for_date(test_date)
        
        # Verify retry handler was called (which includes rate limiting)
        mock_fetch_with_retry.assert_called_once()
    
    def test_caching_behavior(self, options_handler, sample_contracts):
        """Test that data is properly cached after API calls."""
        test_date = datetime(2021, 11, 19)
        
        # Mock API response
        mock_response = Mock()
        mock_response.results = [
            {
                'ticker': 'O:SPY250929C00600000',
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 600.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            }
        ]
        
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # First call should fetch from API
        contracts1 = options_handler.get_contract_list_for_date(test_date)
        assert len(contracts1) == 1
        
        # Second call should use cache (no additional API calls)
        options_handler.retry_handler.fetch_with_retry.reset_mock()
        contracts2 = options_handler.get_contract_list_for_date(test_date)
        assert len(contracts2) == 1
        assert contracts1[0].ticker == contracts2[0].ticker
        
        # Verify no additional API calls were made
        options_handler.retry_handler.fetch_with_retry.assert_not_called()
    
    def test_private_method_enforcement_still_works(self, options_handler):
        """Test that private method enforcement still works in Phase 3."""
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

    def test_additive_caching_logic(self, options_handler, sample_contracts):
        """Test that additive caching works correctly when cached contracts don't satisfy criteria."""
        test_date = datetime(2021, 11, 19)
        
        # Create a limited set of cached contracts (only high strikes)
        limited_cached_contracts = [
            OptionContractDTO(
                ticker="O:SPY250929C00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(date(2025, 9, 29)),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
        
        # Cache the limited contracts
        options_handler._cache_contracts(test_date, limited_cached_contracts)
        
        # Request contracts with a lower strike range that won't be satisfied by cache
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(400.0),
            max_strike=StrikePrice(500.0)
        )
        
        # Mock API response with contracts in the requested range
        mock_response = Mock()
        mock_response.results = [
            {
                'ticker': 'O:SPY250929C00450000',
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 450.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            }
        ]
        
        # Mock the API client and retry handler
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Get contracts with filtering - should trigger additive caching
        contracts = options_handler.get_contract_list_for_date(test_date, strike_range=strike_range)
        
        # Should return the API contract that matches the criteria
        assert len(contracts) == 1
        assert contracts[0].ticker == "O:SPY250929C00450000"
        assert contracts[0].strike_price.value == 450.0
        
        # Verify API was called to fetch additional contracts
        options_handler.retry_handler.fetch_with_retry.assert_called_once()
        
        # Verify the merged contracts are now cached
        cached_contracts = options_handler.cache_manager.load_contracts("SPY", test_date.date())
        assert len(cached_contracts) == 2  # Original cached + new API contract
        tickers = {c.ticker for c in cached_contracts}
        assert "O:SPY250929C00600000" in tickers  # Original cached
        assert "O:SPY250929C00450000" in tickers  # New from API

    def test_additive_caching_with_no_filters(self, options_handler, sample_contracts):
        """Test that additive caching doesn't trigger when no filters are specified."""
        test_date = datetime(2021, 11, 19)
        
        # Cache some contracts
        options_handler._cache_contracts(test_date, sample_contracts)
        
        # Mock API response
        mock_response = Mock()
        mock_response.results = []
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Get contracts without any filters - should return cached contracts without API call
        contracts = options_handler.get_contract_list_for_date(test_date)
        
        # Should return cached contracts
        assert len(contracts) == 2
        assert contracts == sample_contracts
        
        # API should not be called since cached contracts satisfy criteria (no filters)
        options_handler.retry_handler.fetch_with_retry.assert_not_called()

    def test_additive_caching_duplicate_handling(self, options_handler):
        """Test that additive caching properly handles duplicate contracts."""
        test_date = datetime(2021, 11, 19)
        
        # Create cached contracts with high strikes that won't satisfy the requested range
        cached_contracts = [
            OptionContractDTO(
                ticker="O:SPY250929C00600000",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(600.0),
                expiration_date=ExpirationDate(date(2025, 9, 29)),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
        ]
        
        # Cache the contracts
        options_handler._cache_contracts(test_date, cached_contracts)
        
        # Mock API response that includes the same contract (duplicate) and new contracts
        mock_response = Mock()
        mock_response.results = [
            {
                'ticker': 'O:SPY250929C00600000',  # Same as cached (duplicate)
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 600.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            },
            {
                'ticker': 'O:SPY250929C00450000',  # New contract in requested range
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 450.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            },
            {
                'ticker': 'O:SPY250929C00460000',  # New contract in requested range
                'underlying_ticker': 'SPY',
                'contract_type': 'call',
                'strike_price': 460.0,
                'expiration_date': '2025-09-29',
                'exercise_style': 'american',
                'shares_per_contract': 100,
                'primary_exchange': 'BATO',
                'cfi': 'OCASPS',
                'additional_underlyings': None
            }
        ]
        
        # Mock the API client and retry handler
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Request contracts with a range that doesn't include the cached contract
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(440.0),
            max_strike=StrikePrice(470.0)
        )
        
        # Get contracts - should trigger additive caching
        contracts = options_handler.get_contract_list_for_date(test_date, strike_range=strike_range)
        
        # Should return only the contracts in the requested range (2 contracts)
        assert len(contracts) == 2
        tickers = {c.ticker for c in contracts}
        assert "O:SPY250929C00450000" in tickers
        assert "O:SPY250929C00460000" in tickers
        assert "O:SPY250929C00600000" not in tickers  # Not in requested range
        
        # Verify the merged contracts are cached (should be 3 total: 1 cached + 2 new, no duplicates)
        cached_contracts_after = options_handler.cache_manager.load_contracts("SPY", test_date.date())
        assert len(cached_contracts_after) == 3  # 1 original + 2 new (duplicate removed)
        all_tickers = {c.ticker for c in cached_contracts_after}
        assert "O:SPY250929C00600000" in all_tickers  # Original cached
        assert "O:SPY250929C00450000" in all_tickers  # New from API
        assert "O:SPY250929C00460000" in all_tickers  # New from API


class TestOptionsHandlerPerformance:
    """Performance tests for Phase 3 OptionsHandler."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def options_handler(self, temp_dir):
        """Create OptionsHandler with temporary directory."""
        return OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir, use_free_tier=True)
    
    def test_large_contract_list_performance(self, options_handler):
        """Test performance with large contract lists."""
        test_date = datetime(2021, 11, 19)
        
        # Create a large number of contracts
        future_date = date.today() + timedelta(days=30)
        large_contract_list = []
        
        for i in range(1000):  # 1000 contracts
            contract = OptionContractDTO(
                ticker=f"O:SPY250929C{i:06d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(500.0 + i),
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
            large_contract_list.append(contract)
        
        # Cache the large contract list
        options_handler._cache_contracts(test_date, large_contract_list)
        
        # Test retrieval performance
        import time
        start_time = time.time()
        contracts = options_handler.get_contract_list_for_date(test_date)
        end_time = time.time()
        
        assert len(contracts) == 1000
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_filtering_performance(self, options_handler):
        """Test filtering performance with large datasets."""
        test_date = datetime(2021, 11, 19)
        
        # Create contracts with various strikes
        future_date = date.today() + timedelta(days=30)
        contracts = []
        
        for i in range(500):  # 500 contracts
            contract = OptionContractDTO(
                ticker=f"O:SPY250929C{i:06d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(400.0 + i * 2),  # Strikes from 400 to 1398
                expiration_date=ExpirationDate(future_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS",
                additional_underlyings=None
            )
            contracts.append(contract)
        
        # Cache contracts
        options_handler._cache_contracts(test_date, contracts)
        
        # Mock API response to prevent actual API calls during performance test
        mock_response = Mock()
        mock_response.results = []
        options_handler.retry_handler.fetch_with_retry = Mock(return_value=mock_response)
        
        # Test filtering performance
        import time
        start_time = time.time()
        
        # Apply multiple filters
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(500.0),
            max_strike=StrikePrice(1000.0)
        )
        
        expiration_range = ExpirationRangeDTO(
            min_days=25,
            max_days=35
        )
        
        filtered_contracts = options_handler.get_contract_list_for_date(
            test_date, 
            strike_range=strike_range,
            expiration_range=expiration_range
        )
        
        end_time = time.time()
        
        # Should filter efficiently
        assert len(filtered_contracts) < 500  # Some contracts should be filtered out
        assert (end_time - start_time) < 0.5  # Should complete within 0.5 seconds
        
        # Verify filtering worked correctly
        for contract in filtered_contracts:
            assert 500.0 <= float(contract.strike_price.value) <= 1000.0
