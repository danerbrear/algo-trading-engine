"""
Error handling tests for Phase 5 OptionsHandler.

This module tests error handling and backward compatibility:
- Clear error messages for old API usage
- Graceful handling of invalid inputs
- API failure scenarios
- Cache corruption handling
- Edge cases and boundary conditions
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.common.options_handler import OptionsHandler
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    StrikePrice, ExpirationDate
)
from src.common.models import OptionType


class TestOptionsHandlerErrorHandling:
    """Test error handling and edge cases."""
    
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
    
    def test_old_api_usage_error_messages(self, options_handler):
        """Test that old API usage produces clear, helpful error messages."""
        
        # Test private method access
        with pytest.raises(AttributeError) as exc_info:
            options_handler._cache_contracts(datetime.now(), [])
        
        error_message = str(exc_info.value)
        assert "private method" in error_message.lower()
        assert "should not be accessed externally" in error_message.lower()
        assert "Use the public API methods instead" in error_message.lower()
        
        # Test other private methods
        private_methods = [
            '_cache_bar',
            '_get_cache_stats',
            '_cached_contracts_satisfy_criteria',
            '_merge_contracts',
            '_apply_contract_filters',
            '_fetch_contracts_from_api',
            '_fetch_bar_from_api',
            '_convert_api_contract_to_dto',
            '_convert_api_bar_to_dto'
        ]
        
        for method_name in private_methods:
            with pytest.raises(AttributeError) as exc_info:
                getattr(options_handler, method_name)
            
            error_message = str(exc_info.value)
            assert "private method" in error_message.lower()
            assert "should not be accessed externally" in error_message.lower()
    
    def test_invalid_initialization_parameters(self, temp_dir):
        """Test error handling for invalid initialization parameters."""
        
        # Test missing API key
        with pytest.raises(ValueError, match="Polygon.io API key is required"):
            OptionsHandler("SPY", api_key=None, cache_dir=temp_dir)
        
        with pytest.raises(ValueError, match="Polygon.io API key is required"):
            OptionsHandler("SPY", api_key="", cache_dir=temp_dir)
        
        # Test invalid symbol
        with pytest.raises(ValueError):
            OptionsHandler("", api_key="test_key", cache_dir=temp_dir)
        
        with pytest.raises(ValueError):
            OptionsHandler(None, api_key="test_key", cache_dir=temp_dir)
        
        # Test invalid cache directory
        with pytest.raises((OSError, PermissionError)):
            OptionsHandler("SPY", api_key="test_key", cache_dir="/invalid/path/that/does/not/exist")
    
    def test_invalid_date_parameters(self, options_handler):
        """Test error handling for invalid date parameters."""
        
        # Test invalid date types
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date("invalid_date")
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date(None)
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date(123)
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date([])
        
        # Test invalid date objects
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date(date(1900, 1, 1))  # Too old
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_contract_list_for_date(date(2100, 1, 1))  # Too far in future
    
    def test_invalid_contract_parameters(self, options_handler):
        """Test error handling for invalid contract parameters."""
        
        # Test invalid contract types
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar("invalid_contract", datetime.now())
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar(None, datetime.now())
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar(123, datetime.now())
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar([], datetime.now())
        
        # Test invalid date for get_option_bar
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar(Mock(), "invalid_date")
        
        with pytest.raises((TypeError, AttributeError)):
            options_handler.get_option_bar(Mock(), None)
    
    def test_invalid_filter_parameters(self, options_handler):
        """Test error handling for invalid filter parameters."""
        
        # Test invalid strike range
        with pytest.raises(ValueError, match="Min strike cannot be greater than max strike"):
            StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('610.0')),
                max_strike=StrikePrice(Decimal('590.0'))
            )
        
        with pytest.raises(ValueError, match="Tolerance cannot be negative"):
            StrikeRangeDTO(
                target_strike=StrikePrice(Decimal('600.0')),
                tolerance=Decimal('-1.0')
            )
        
        # Test invalid expiration range
        with pytest.raises(ValueError, match="Min days cannot be greater than max days"):
            ExpirationRangeDTO(min_days=20, max_days=10)
        
        with pytest.raises(ValueError, match="Days cannot be negative"):
            ExpirationRangeDTO(min_days=-1, max_days=10)
        
        with pytest.raises(ValueError, match="Days cannot be negative"):
            ExpirationRangeDTO(min_days=10, max_days=-1)
    
    def test_api_failure_handling(self, options_handler):
        """Test handling of API failures."""
        
        # Test API timeout
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("API timeout")):
            contracts = options_handler._fetch_contracts_from_api(date.today())
            assert contracts == []
        
        # Test API rate limit exceeded
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("Rate limit exceeded")):
            contracts = options_handler._fetch_contracts_from_api(date.today())
            assert contracts == []
        
        # Test API authentication failure
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("Authentication failed")):
            contracts = options_handler._fetch_contracts_from_api(date.today())
            assert contracts == []
        
        # Test network error
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("Network error")):
            contracts = options_handler._fetch_contracts_from_api(date.today())
            assert contracts == []
    
    def test_cache_corruption_handling(self, options_handler):
        """Test handling of corrupted cache files."""
        
        # Test cache corruption during contract loading
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         side_effect=Exception("Cache corruption")):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts == []
        
        # Test cache corruption during bar loading
        with patch.object(options_handler.cache_manager, 'load_bar', 
                         side_effect=Exception("Cache corruption")):
            contract = Mock()
            contract.ticker = "test_ticker"
            bar = options_handler.get_option_bar(contract, datetime.now())
            assert bar is None
        
        # Test cache corruption during saving
        with patch.object(options_handler.cache_manager, 'save_contracts', 
                         side_effect=Exception("Cache write error")):
            # Should not raise exception, just log error
            options_handler._cache_contracts(datetime.now(), [])
    
    def test_invalid_data_handling(self, options_handler):
        """Test handling of invalid data from API."""
        
        # Test invalid contract data from API
        invalid_contract_data = [
            None,
            {},
            {"invalid": "data"},
            {"ticker": None, "strike_price": 100},
            {"ticker": "test", "strike_price": None},
            {"ticker": "test", "strike_price": "invalid"},
            {"ticker": "test", "strike_price": -100},  # Negative strike
            {"ticker": "test", "strike_price": 100, "contract_type": "invalid"},
            {"ticker": "test", "strike_price": 100, "expiration_date": "invalid_date"},
        ]
        
        for invalid_data in invalid_contract_data:
            with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                             return_value=Mock(results=invalid_data)):
                contracts = options_handler._fetch_contracts_from_api(date.today())
                # Should handle gracefully and return empty list or filter out invalid data
                assert isinstance(contracts, list)
        
        # Test invalid bar data from API
        invalid_bar_data = [
            None,
            {},
            {"invalid": "data"},
            {"o": None, "h": 100, "l": 90, "c": 95},
            {"o": 100, "h": None, "l": 90, "c": 95},
            {"o": 100, "h": 100, "l": None, "c": 95},
            {"o": 100, "h": 100, "l": 90, "c": None},
            {"o": -100, "h": 100, "l": 90, "c": 95},  # Negative price
            {"o": 100, "h": 90, "l": 100, "c": 95},  # High < Low
        ]
        
        for invalid_data in invalid_bar_data:
            with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                             return_value=Mock(results=[invalid_data] if invalid_data else [])):
                contract = Mock()
                contract.ticker = "test_ticker"
                bar = options_handler._fetch_bar_from_api(contract, date.today(), 1, "day")
                # Should handle gracefully and return None for invalid data
                assert bar is None or isinstance(bar, OptionBarDTO)
    
    def test_edge_cases_and_boundary_conditions(self, options_handler):
        """Test edge cases and boundary conditions."""
        
        # Test with empty contract list
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=[]):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts == []
        
        # Test with None cache result
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=None):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts == []
        
        # Test with mixed valid/invalid contracts
        valid_contract = OptionContractDTO(
            ticker="O:SPY250115C00600000",
            underlying_ticker="SPY",
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('600.0')),
            expiration_date=ExpirationDate(date(2025, 1, 15)),
            exercise_style="american",
            shares_per_contract=100
        )
        
        mixed_contracts = [valid_contract, None, "invalid", {}, valid_contract]
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=mixed_contracts):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            # Should filter out invalid contracts
            assert len(contracts) == 2  # Only valid contracts
            assert all(isinstance(c, OptionContractDTO) for c in contracts)
        
        # Test with extreme values
        extreme_contract = OptionContractDTO(
            ticker="O:SPY250115C99999999",  # Very high strike
            underlying_ticker="SPY",
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('999999.99')),
            expiration_date=ExpirationDate(date(2030, 12, 31)),  # Far future
            exercise_style="american",
            shares_per_contract=100
        )
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=[extreme_contract]):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert len(contracts) == 1
            assert contracts[0].strike_price.value == Decimal('999999.99')
    
    def test_memory_pressure_handling(self, options_handler):
        """Test handling under memory pressure."""
        
        # Test with very large dataset
        large_contracts = []
        for i in range(10000):  # 10,000 contracts
            contract = OptionContractDTO(
                ticker=f"O:SPY250115C{int(600 + i % 100):08d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal(str(600 + i % 100))),
                expiration_date=ExpirationDate(date(2025, 1, 15)),
                exercise_style="american",
                shares_per_contract=100
            )
            large_contracts.append(contract)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=large_contracts):
            # Should handle large dataset without crashing
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert len(contracts) == 10000
        
        # Test with memory-constrained environment
        with patch('psutil.Process') as mock_process:
            mock_memory = Mock()
            mock_memory.rss = 100 * 1024 * 1024  # 100MB
            mock_process.return_value.memory_info.return_value = mock_memory
            
            # Should still work under memory pressure
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert isinstance(contracts, list)
    
    def test_concurrent_error_handling(self, options_handler):
        """Test error handling in concurrent scenarios."""
        import threading
        import queue
        
        errors = queue.Queue()
        results = queue.Queue()
        
        def worker(worker_id):
            try:
                # Simulate API failure for some workers
                if worker_id % 2 == 0:
                    with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                                     side_effect=Exception("API failure")):
                        contracts = options_handler._fetch_contracts_from_api(date.today())
                        results.put((worker_id, len(contracts)))
                else:
                    with patch.object(options_handler.cache_manager, 'load_contracts', 
                                     return_value=[]):
                        contracts = options_handler.get_contract_list_for_date(datetime.now())
                        results.put((worker_id, len(contracts)))
            except Exception as e:
                errors.put((worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        worker_results = list(results.queue)
        worker_errors = list(errors.queue)
        
        # Should handle errors gracefully
        assert len(worker_results) >= 3  # At least some should succeed
        assert len(worker_errors) <= 2  # Some may fail, but not all
    
    def test_graceful_degradation(self, options_handler):
        """Test graceful degradation when components fail."""
        
        # Test with cache manager failure
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         side_effect=Exception("Cache failure")):
            with patch.object(options_handler.cache_manager, 'save_contracts', 
                             side_effect=Exception("Cache write failure")):
                # Should still work by falling back to API
                with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                                 return_value=Mock(results=[])):
                    contracts = options_handler.get_contract_list_for_date(datetime.now())
                    assert contracts == []
        
        # Test with API failure
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("API failure")):
            contracts = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts == []
        
        # Test with partial failure
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         return_value=[]):
            with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                             side_effect=Exception("API failure")):
                contracts = options_handler.get_contract_list_for_date(datetime.now())
                assert contracts == []
    
    def test_error_message_clarity(self, options_handler):
        """Test that error messages are clear and helpful."""
        
        # Test private method error message
        with pytest.raises(AttributeError) as exc_info:
            options_handler._cache_contracts(datetime.now(), [])
        
        error_message = str(exc_info.value)
        assert "private method" in error_message
        assert "should not be accessed externally" in error_message
        assert "Use the public API methods instead" in error_message
        
        # Test invalid parameter error messages
        with pytest.raises(ValueError) as exc_info:
            StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('610.0')),
                max_strike=StrikePrice(Decimal('590.0'))
            )
        
        error_message = str(exc_info.value)
        assert "Min strike cannot be greater than max strike" in error_message
        
        # Test missing API key error message
        with pytest.raises(ValueError) as exc_info:
            OptionsHandler("SPY", api_key=None, cache_dir="temp")
        
        error_message = str(exc_info.value)
        assert "Polygon.io API key is required" in error_message
    
    def test_recovery_from_errors(self, options_handler):
        """Test recovery from various error conditions."""
        
        # Test recovery from API failure
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', 
                         side_effect=Exception("API failure")):
            contracts1 = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts1 == []
        
        # Should recover and work normally after error
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         return_value=[]):
            contracts2 = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts2 == []
        
        # Test recovery from cache corruption
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         side_effect=Exception("Cache corruption")):
            contracts1 = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts1 == []
        
        # Should recover and work normally after error
        with patch.object(options_handler.cache_manager, 'load_contracts', 
                         return_value=[]):
            contracts2 = options_handler.get_contract_list_for_date(datetime.now())
            assert contracts2 == []
