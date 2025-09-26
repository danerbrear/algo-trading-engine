"""
Simple validation tests for Phase 5 OptionsHandler.

This module provides basic validation tests to ensure the core functionality works:
- Basic API functionality
- Error handling
- Performance basics
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.common.options_handler import OptionsHandler
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    StrikePrice, ExpirationDate
)
from src.common.models import OptionType


class TestOptionsHandlerPhase5Simple:
    """Simple validation tests for Phase 5 OptionsHandler."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
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
        contracts = []
        
        # Create contracts with future expiration dates
        expirations = ["2025-12-15", "2025-12-17", "2025-12-24", "2026-01-21"]
        strikes = [580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 620.0]
        
        for exp_str in expirations:
            for strike in strikes:
                # Call contracts
                call_ticker = f"O:SPY250115C{int(strike * 1000):08d}"
                call_contract = OptionContractDTO(
                    ticker=call_ticker,
                    underlying_ticker="SPY",
                    contract_type=OptionType.CALL,
                    strike_price=StrikePrice(Decimal(str(strike))),
                    expiration_date=ExpirationDate(datetime.strptime(exp_str, '%Y-%m-%d').date()),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS"
                )
                contracts.append(call_contract)
                
                # Put contracts
                put_ticker = f"O:SPY250115P{int(strike * 1000):08d}"
                put_contract = OptionContractDTO(
                    ticker=put_ticker,
                    underlying_ticker="SPY",
                    contract_type=OptionType.PUT,
                    strike_price=StrikePrice(Decimal(str(strike))),
                    expiration_date=ExpirationDate(datetime.strptime(exp_str, '%Y-%m-%d').date()),
                    exercise_style="american",
                    shares_per_contract=100,
                    primary_exchange="BATO",
                    cfi="OCASPS"
                )
                contracts.append(put_contract)
        
        return contracts
    
    def test_basic_initialization(self, temp_dir):
        """Test basic initialization."""
        handler = OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir)
        assert handler.symbol == "SPY"
        assert handler.api_key == "test_key"
        assert handler.cache_manager is not None
    
    def test_get_contract_list_for_date_with_cache(self, options_handler, sample_contracts):
        """Test getting contracts from cache."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) == len(sample_contracts)
            assert all(isinstance(c, OptionContractDTO) for c in contracts)
    
    def test_get_contract_list_for_date_with_strike_filter(self, options_handler, sample_contracts):
        """Test getting contracts with strike filter."""
        test_date = datetime(2025, 12, 10)
        
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal('590.0')),
            max_strike=StrikePrice(Decimal('610.0'))
        )
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            contracts = options_handler.get_contract_list_for_date(test_date, strike_range=strike_range)
            assert len(contracts) > 0
            assert all(590.0 <= float(c.strike_price.value) <= 610.0 for c in contracts)
    
    def test_get_contract_list_for_date_with_expiration_filter(self, options_handler, sample_contracts):
        """Test getting contracts with expiration filter."""
        test_date = datetime(2025, 12, 10)
        
        expiration_range = ExpirationRangeDTO(min_days=1, max_days=30)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            contracts = options_handler.get_contract_list_for_date(test_date, expiration_range=expiration_range)
            assert len(contracts) > 0
            assert all(isinstance(c, OptionContractDTO) for c in contracts)
    
    def test_get_option_bar_from_cache(self, options_handler, sample_contracts):
        """Test getting option bar from cache."""
        test_date = datetime(2025, 12, 10)
        contract = sample_contracts[0]
        
        sample_bar = OptionBarDTO(
            ticker=contract.ticker,
            timestamp=datetime(2025, 12, 10, 16, 0, 0),
            open_price=Decimal('2.50'),
            high_price=Decimal('2.75'),
            low_price=Decimal('2.25'),
            close_price=Decimal('2.60'),
            volume=150,
            volume_weighted_avg_price=Decimal('2.55'),
            number_of_transactions=25,
            adjusted=True
        )
        
        with patch.object(options_handler.cache_manager, 'load_bar', return_value=sample_bar):
            bar = options_handler.get_option_bar(contract, test_date)
            assert bar is not None
            assert bar.ticker == contract.ticker
            assert bar.volume > 0
    
    def test_get_options_chain(self, options_handler, sample_contracts):
        """Test getting complete options chain."""
        test_date = datetime(2025, 12, 10)
        current_price = 600.0
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            chain = options_handler.get_options_chain(test_date, current_price)
            assert chain.underlying_symbol == "SPY"
            assert chain.current_price == Decimal('600.0')
            assert len(chain.contracts) == len(sample_contracts)
    
    def test_error_handling_missing_api_key(self, temp_dir):
        """Test error handling for missing API key."""
        with pytest.raises(ValueError, match="Polygon.io API key is required"):
            OptionsHandler("SPY", api_key=None, cache_dir=temp_dir)
    
    def test_error_handling_invalid_symbol(self, temp_dir):
        """Test error handling for invalid symbol."""
        with pytest.raises(ValueError):
            OptionsHandler("", api_key="test_key", cache_dir=temp_dir)
    
    def test_error_handling_invalid_strike_range(self):
        """Test error handling for invalid strike range."""
        with pytest.raises(ValueError, match="Min strike cannot be greater than max strike"):
            StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('610.0')),
                max_strike=StrikePrice(Decimal('590.0'))
            )
    
    def test_error_handling_invalid_expiration_range(self):
        """Test error handling for invalid expiration range."""
        with pytest.raises(ValueError, match="Min days cannot be greater than max days"):
            ExpirationRangeDTO(min_days=20, max_days=10)
    
    def test_api_failure_handling(self, options_handler):
        """Test handling of API failures."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("API failure")):
            contracts = options_handler._fetch_contracts_from_api(test_date.date())
            assert contracts == []
    
    def test_cache_failure_handling(self, options_handler):
        """Test handling of cache failures."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         side_effect=Exception("Cache failure")):
            with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                             return_value=Mock(results=[])):
                contracts = options_handler.get_contract_list_for_date(test_date)
                assert contracts == []
    
    def test_performance_basic(self, options_handler, sample_contracts):
        """Test basic performance."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            import time
            
            start_time = time.time()
            contracts = options_handler.get_contract_list_for_date(test_date)
            end_time = time.time()
            
            processing_time = end_time - start_time
            assert processing_time < 1.0  # Should process in less than 1 second
            assert len(contracts) == len(sample_contracts)
    
    def test_memory_usage_basic(self, options_handler, sample_contracts):
        """Test basic memory usage."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            contracts = options_handler.get_contract_list_for_date(test_date)
            
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
            
            # Clean up
            del contracts
            gc.collect()
    
    def test_data_consistency(self, options_handler, sample_contracts):
        """Test data consistency across multiple calls."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            # Get contracts multiple times
            contracts1 = options_handler.get_contract_list_for_date(test_date)
            contracts2 = options_handler.get_contract_list_for_date(test_date)
            
            # Should return identical results
            assert len(contracts1) == len(contracts2)
            assert all(c1.ticker == c2.ticker for c1, c2 in zip(contracts1, contracts2))
    
    def test_edge_cases(self, options_handler):
        """Test edge cases."""
        test_date = datetime(2025, 12, 10)
        
        # Test with empty contract list
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=[]):
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) == 0
        
        # Test with None cache result
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=None):
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) == 0
    
    def test_validation_basic(self, options_handler, sample_contracts):
        """Test basic validation."""
        test_date = datetime(2025, 12, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            contracts = options_handler.get_contract_list_for_date(test_date)
            
            # All contracts should be valid
            for contract in contracts:
                assert isinstance(contract, OptionContractDTO)
                assert contract.ticker is not None
                assert contract.underlying_ticker == "SPY"
                assert contract.strike_price.value > 0
                assert contract.expiration_date.date > date.today()
    
    def test_integration_basic(self, options_handler, sample_contracts):
        """Test basic integration."""
        test_date = datetime(2025, 12, 10)
        current_price = 600.0
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            # Test complete workflow
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) > 0
            
            # Test individual bar
            contract = contracts[0]
            with patch.object(options_handler.cache_manager, 'load_bar', return_value=None):
                bar = options_handler.get_option_bar(contract, test_date)
                # Should return None when no bar data available
                assert bar is None
            
            # Test options chain
            chain = options_handler.get_options_chain(test_date, current_price)
            assert chain.underlying_symbol == "SPY"
            assert chain.current_price == Decimal('600.0')
            assert len(chain.contracts) == len(contracts)
