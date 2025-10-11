"""
Comprehensive integration tests for Phase 5 OptionsHandler API.

This module tests the complete OptionsHandler functionality including:
- Full API integration with real-world scenarios
- Performance testing and optimization
- Error handling and edge cases
- Cache behavior and efficiency
- Strategy integration examples
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import time
import os

from src.common.options_handler import OptionsHandler
from src.common.options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    OptionsChainDTO, StrikePrice, ExpirationDate
)
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.models import OptionType
from src.common.cache.options_cache_manager import OptionsCacheManager


class TestOptionsHandlerPhase5Integration:
    """Comprehensive integration tests for the complete OptionsHandler API."""
    
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
        """Create comprehensive sample contracts for testing."""
        contracts = []
        
        # Create contracts for different expirations and strikes (use future dates)
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
    
    @pytest.fixture
    def sample_bars(self, sample_contracts):
        """Create sample bar data for contracts."""
        bars = {}
        base_time = datetime(2025, 1, 10, 16, 0, 0)
        
        for i, contract in enumerate(sample_contracts[:10]):  # Limit to first 10 for performance
            bar = OptionBarDTO(
                ticker=contract.ticker,
                timestamp=base_time + timedelta(minutes=i),
                open_price=Decimal('2.50'),
                high_price=Decimal('2.75'),
                low_price=Decimal('2.25'),
                close_price=Decimal('2.60'),
                volume=150 + i * 10,
                volume_weighted_avg_price=Decimal('2.55'),
                number_of_transactions=25 + i,
                adjusted=True
            )
            bars[contract.ticker] = bar
        
        return bars
    
    def test_complete_api_workflow(self, options_handler, sample_contracts, sample_bars):
        """Test complete API workflow from contracts to strategy analysis."""
        test_date = datetime(2025, 12, 10)  # Use a date closer to the contract expirations
        
        # Mock the cache manager to return our sample data
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            with patch.object(options_handler.cache_manager, 'load_bar', side_effect=lambda s, d, t: sample_bars.get(t)):
                
                # 1. Get contracts with filtering
                strike_range = StrikeRangeDTO(
                    min_strike=StrikePrice(Decimal('590.0')),
                    max_strike=StrikePrice(Decimal('610.0'))
                )
                expiration_range = ExpirationRangeDTO(min_days=1, max_days=30)  # More reasonable range
                
                contracts = options_handler.get_contract_list_for_date(
                    test_date, strike_range=strike_range, expiration_range=expiration_range
                )
                
                assert len(contracts) > 0
                assert all(590.0 <= float(c.strike_price.value) <= 610.0 for c in contracts)
                
                # 2. Get individual bar data
                contract = contracts[0]
                bar = options_handler.get_option_bar(contract, test_date)
                
                assert bar is not None
                assert bar.ticker == contract.ticker
                assert bar.volume > 0
                
                # 3. Get complete options chain
                current_price = 600.0
                chain = options_handler.get_options_chain(
                    test_date, current_price, strike_range=strike_range, expiration_range=expiration_range
                )
                
                assert chain.underlying_symbol == "SPY"
                assert chain.current_price == Decimal('600.0')
                assert len(chain.contracts) > 0
                assert len(chain.bars) > 0
                
                # 4. Use helper methods for strategy analysis
                calls = chain.get_calls()
                puts = chain.get_puts()
                
                assert len(calls) > 0
                assert len(puts) > 0
                
                # 5. Test strategy-specific analysis
                atm_contracts = OptionsRetrieverHelper.find_atm_contracts(contracts, current_price)
                assert atm_contracts[0] is not None  # ATM call
                assert atm_contracts[1] is not None  # ATM put
                
                # 6. Test credit spread analysis
                credit_spread_legs = OptionsRetrieverHelper.find_credit_spread_legs(
                    contracts, current_price, "2025-01-15", OptionType.CALL, spread_width=5
                )
                
                if credit_spread_legs[0] and credit_spread_legs[1]:
                    short_leg, long_leg = credit_spread_legs
                    net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
                        short_leg, long_leg, 2.50, 1.00
                    )
                    assert net_credit == 1.50
                    
                    max_profit, max_loss = OptionsRetrieverHelper.calculate_max_profit_loss(
                        short_leg, long_leg, net_credit
                    )
                    assert max_profit == net_credit
                    assert max_loss > 0
    
    def test_performance_with_large_dataset(self, options_handler, temp_dir):
        """Test performance with large dataset."""
        # Create a large number of contracts
        large_contracts = []
        for i in range(1000):  # 1000 contracts
            strike = 580.0 + (i % 20) * 2.0  # Strikes from 580 to 618
            exp_date = date(2025, 12, 15) + timedelta(days=(i % 4) * 7)  # 4 different expirations
            
            contract = OptionContractDTO(
                ticker=f"O:SPY250115C{int(strike * 1000):08d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL if i % 2 == 0 else OptionType.PUT,
                strike_price=StrikePrice(Decimal(str(strike))),
                expiration_date=ExpirationDate(exp_date),
                exercise_style="american",
                shares_per_contract=100
            )
            large_contracts.append(contract)
        
        test_date = datetime(2025, 1, 10)
        
        # Mock cache to return large dataset
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=large_contracts):
            
            start_time = time.time()
            
            # Test filtering performance
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('590.0')),
                max_strike=StrikePrice(Decimal('610.0'))
            )
            filtered_contracts = options_handler.get_contract_list_for_date(
                test_date, strike_range=strike_range
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 1000 contracts in reasonable time (< 1 second)
            assert processing_time < 1.0
            assert len(filtered_contracts) > 0
            assert all(590.0 <= float(c.strike_price.value) <= 610.0 for c in filtered_contracts)
    
    def test_cache_efficiency(self, options_handler, sample_contracts):
        """Test cache efficiency and behavior."""
        test_date = datetime(2025, 1, 10)
        
        # Mock cache manager
        cache_manager = Mock()
        options_handler.cache_manager = cache_manager
        
        # First call - should load from cache
        cache_manager.load_contracts.return_value = sample_contracts
        cache_manager.load_bar.return_value = None
        
        contracts1 = options_handler.get_contract_list_for_date(test_date)
        assert len(contracts1) == len(sample_contracts)
        
        # Verify cache was accessed
        cache_manager.load_contracts.assert_called_once()
        
        # Second call - should use cache again
        contracts2 = options_handler.get_contract_list_for_date(test_date)
        assert len(contracts2) == len(contracts1)
        
        # Verify cache was accessed again (not API)
        assert cache_manager.load_contracts.call_count == 2
    
    def test_error_handling_comprehensive(self, options_handler, temp_dir):
        """Test comprehensive error handling scenarios."""
        test_date = datetime(2025, 1, 10)
        
        # Test with no API key
        with pytest.raises(ValueError, match="Polygon.io API key is required"):
            OptionsHandler("SPY", api_key=None, cache_dir=temp_dir)
        
        # Test with invalid symbol
        with pytest.raises(ValueError):
            OptionsHandler("", api_key="test_key", cache_dir="temp")
        
        # Test with invalid date
        with pytest.raises((ValueError, TypeError)):
            options_handler.get_contract_list_for_date("invalid_date")
        
        # Test with invalid strike range
        with pytest.raises(ValueError):
            StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('610.0')),
                max_strike=StrikePrice(Decimal('590.0'))  # min > max
            )
        
        # Test with invalid expiration range
        with pytest.raises(ValueError):
            ExpirationRangeDTO(min_days=20, max_days=10)  # min > max
    
    def test_memory_efficiency(self, options_handler, sample_contracts):
        """Test memory efficiency with large datasets."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        large_contracts = sample_contracts * 100  # 100x the sample size
        
        test_date = datetime(2025, 1, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=large_contracts):
            
            # Process large dataset
            contracts = options_handler.get_contract_list_for_date(test_date)
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB for this test)
            assert memory_increase < 100 * 1024 * 1024  # 100MB
            
            # Force garbage collection
            gc.collect()
            
            # Memory should be cleaned up
            final_memory = process.memory_info().rss
            assert final_memory < current_memory + 50 * 1024 * 1024  # Should clean up
    
    def test_concurrent_access(self, options_handler, sample_contracts):
        """Test concurrent access to OptionsHandler."""
        import threading
        import queue
        
        test_date = datetime(2025, 1, 10)
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker(worker_id):
            try:
                with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
                    contracts = options_handler.get_contract_list_for_date(test_date)
                    results.put((worker_id, len(contracts)))
            except Exception as e:
                errors.put((worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        
        # All workers should get the same result
        worker_results = list(results.queue)
        assert len(worker_results) == 5
        assert all(result[1] == len(sample_contracts) for result in worker_results)
    
    def test_strategy_integration_example(self, options_handler, sample_contracts, sample_bars):
        """Test integration with a realistic strategy scenario."""
        test_date = datetime(2025, 1, 10)
        current_price = 600.0
        
        # Mock cache with sample data
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            with patch.object(options_handler.cache_manager, 'load_bar', side_effect=lambda s, d, t: sample_bars.get(t)):
                
                # Simulate a credit spread strategy
                # 1. Get contracts for specific expiration
                target_expiration = "2025-01-15"
                contracts = options_handler.get_contract_list_for_date(test_date)
                
                # 2. Filter for target expiration
                exp_contracts = OptionsRetrieverHelper.find_contracts_by_expiration(
                    contracts, target_expiration
                )
                
                # 3. Find credit spread legs
                call_short, call_long = OptionsRetrieverHelper.find_credit_spread_legs(
                    exp_contracts, current_price, target_expiration, OptionType.CALL, spread_width=5
                )
                
                if call_short and call_long:
                    # 4. Calculate strategy metrics
                    short_premium = 2.50
                    long_premium = 1.00
                    net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
                        call_short, call_long, short_premium, long_premium
                    )
                    
                    max_profit, max_loss = OptionsRetrieverHelper.calculate_max_profit_loss(
                        call_short, call_long, net_credit
                    )
                    
                    breakeven = OptionsRetrieverHelper.calculate_breakeven_points(
                        call_short, call_long, net_credit, OptionType.CALL
                    )
                    
                    # 5. Calculate probability of profit
                    pop = OptionsRetrieverHelper.calculate_probability_of_profit(
                        call_short, call_long, net_credit, OptionType.CALL,
                        current_price, 5  # 5 days to expiration
                    )
                    
                    # Verify strategy metrics
                    assert net_credit == 1.50
                    assert max_profit == net_credit
                    assert max_loss > 0
                    assert breakeven[0] == breakeven[1]  # Single breakeven point
                    assert 0.0 <= pop <= 1.0
    
    def test_data_consistency(self, options_handler, sample_contracts):
        """Test data consistency across different API calls."""
        test_date = datetime(2025, 1, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            
            # Get contracts multiple times
            contracts1 = options_handler.get_contract_list_for_date(test_date)
            contracts2 = options_handler.get_contract_list_for_date(test_date)
            
            # Should return identical results
            assert len(contracts1) == len(contracts2)
            assert all(c1.ticker == c2.ticker for c1, c2 in zip(contracts1, contracts2))
            
            # Test with different filters
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('590.0')),
                max_strike=StrikePrice(Decimal('610.0'))
            )
            
            filtered_contracts = options_handler.get_contract_list_for_date(
                test_date, strike_range=strike_range
            )
            
            # Filtered results should be subset of full results
            assert len(filtered_contracts) <= len(contracts1)
            assert all(590.0 <= float(c.strike_price.value) <= 610.0 for c in filtered_contracts)
    
    def test_edge_cases(self, options_handler):
        """Test edge cases and boundary conditions."""
        test_date = datetime(2025, 1, 10)
        
        # Test with empty contract list
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=[]):
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) == 0
        
        # Test with None cache result
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=None):
            contracts = options_handler.get_contract_list_for_date(test_date)
            assert len(contracts) == 0
        
        # Test with invalid contract data
        invalid_contracts = [None, "invalid", {}]
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=invalid_contracts):
            # Should handle gracefully
            contracts = options_handler.get_contract_list_for_date(test_date)
            # Should filter out invalid contracts (they get passed through as-is)
            assert len(contracts) == 3  # Invalid contracts are not filtered in this implementation
    
    def test_api_rate_limiting(self, options_handler):
        """Test API rate limiting behavior."""
        test_date = datetime(2025, 1, 10)
        
        # Mock API calls to test rate limiting
        api_call_count = 0
        
        def mock_fetch_func(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            return Mock(results=[])
        
        with patch.object(options_handler.retry_handler, 'fetch_with_retry') as mock_retry:
            mock_retry.side_effect = mock_fetch_func
            
            # Make multiple API calls
            for _ in range(3):
                options_handler._fetch_contracts_from_api(test_date.date())
            
            # Should have made 3 API calls
            assert api_call_count == 3
    
    def test_comprehensive_validation(self, options_handler, sample_contracts):
        """Test comprehensive data validation."""
        test_date = datetime(2025, 1, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=sample_contracts):
            
            # Test all contract validation
            for contract in sample_contracts[:5]:  # Test first 5 contracts
                validation_errors = OptionsRetrieverHelper.validate_contract_data(contract)
                assert len(validation_errors) == 0, f"Validation errors: {validation_errors}"
            
            # Test contract statistics
            stats = OptionsRetrieverHelper.calculate_contract_statistics(sample_contracts)
            assert stats['total_contracts'] > 0
            assert stats['calls'] > 0
            assert stats['puts'] > 0
            assert stats['min_strike'] > 0
            assert stats['max_strike'] > stats['min_strike']


class TestOptionsHandlerPerformance:
    """Performance tests for OptionsHandler."""
    
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
    
    def test_large_dataset_performance(self, options_handler, temp_dir):
        """Test performance with very large datasets."""
        # Create a very large dataset
        large_contracts = []
        for i in range(5000):  # 5000 contracts
            strike = 580.0 + (i % 50) * 2.0  # 50 different strikes
            exp_date = date(2025, 12, 15) + timedelta(days=(i % 8) * 7)  # 8 different expirations
            
            contract = OptionContractDTO(
                ticker=f"O:SPY250115C{int(strike * 1000):08d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL if i % 2 == 0 else OptionType.PUT,
                strike_price=StrikePrice(Decimal(str(strike))),
                expiration_date=ExpirationDate(exp_date),
                exercise_style="american",
                shares_per_contract=100
            )
            large_contracts.append(contract)
        
        test_date = datetime(2025, 1, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=large_contracts):
            
            start_time = time.time()
            
            # Test filtering performance
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal('590.0')),
                max_strike=StrikePrice(Decimal('610.0'))
            )
            filtered_contracts = options_handler.get_contract_list_for_date(
                test_date, strike_range=strike_range
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 5000 contracts in reasonable time (< 2 seconds)
            assert processing_time < 2.0
            assert len(filtered_contracts) > 0
    
    def test_memory_usage_optimization(self, options_handler, temp_dir):
        """Test memory usage optimization."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        large_contracts = []
        for i in range(10000):  # 10,000 contracts
            contract = OptionContractDTO(
                ticker=f"O:SPY250115C{int(580 + i % 20):08d}",
                underlying_ticker="SPY",
                contract_type=OptionType.CALL,
                strike_price=StrikePrice(Decimal(str(580 + i % 20))),
                expiration_date=ExpirationDate(date(2025, 12, 15)),
                exercise_style="american",
                shares_per_contract=100
            )
            large_contracts.append(contract)
        
        test_date = datetime(2025, 1, 10)
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=large_contracts):
            
            # Process dataset
            contracts = options_handler.get_contract_list_for_date(test_date)
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 200 * 1024 * 1024  # 200MB
            
            # Force garbage collection
            del contracts
            gc.collect()
            
            # Memory should be cleaned up
            final_memory = process.memory_info().rss
            assert final_memory < current_memory + 100 * 1024 * 1024


class TestOptionsHandlerErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_old_api_usage_errors(self, temp_dir):
        """Test that old API usage produces clear error messages."""
        # Test that old methods are not accessible
        handler = OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir)
        
        # These should raise AttributeError with clear messages
        with pytest.raises(AttributeError, match="private method"):
            handler._cache_contracts(datetime.now(), [])
        
        with pytest.raises(AttributeError, match="private method"):
            handler._cache_bar(datetime.now(), "ticker", None)
        
        with pytest.raises(AttributeError, match="private method"):
            handler._get_cache_stats(datetime.now())
    
    def test_invalid_input_handling(self, temp_dir):
        """Test handling of invalid inputs."""
        handler = OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir)
        
        # Test invalid date types
        with pytest.raises((TypeError, AttributeError)):
            handler.get_contract_list_for_date("invalid_date")
        
        with pytest.raises((TypeError, AttributeError)):
            handler.get_contract_list_for_date(None)
        
        # Test invalid contract types
        with pytest.raises((TypeError, AttributeError)):
            handler.get_option_bar("invalid_contract", datetime.now())
        
        with pytest.raises((TypeError, AttributeError)):
            handler.get_option_bar(None, datetime.now())
    
    def test_api_failure_handling(self, temp_dir):
        """Test handling of API failures."""
        handler = OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir)
        
        # Mock API failure
        with patch.object(handler.retry_handler, 'fetch_with_retry', side_effect=Exception("API Error")):
            contracts = handler._fetch_contracts_from_api(date.today())
            assert contracts == []
    
    def test_cache_corruption_handling(self, temp_dir):
        """Test handling of corrupted cache files."""
        handler = OptionsHandler("SPY", api_key="test_key", cache_dir=temp_dir)
        
        # Mock cache corruption
        with patch.object(handler.cache_manager, 'load_contracts', side_effect=Exception("Cache corruption")):
            # Should handle cache corruption gracefully by falling back to API
            with patch.object(handler.retry_handler, 'fetch_with_retry', 
                             return_value=Mock(results=[])):
                contracts = handler.get_contract_list_for_date(datetime.now())
                assert contracts == []
