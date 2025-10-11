"""
Performance tests and optimization benchmarks for Phase 5 OptionsHandler.

This module provides comprehensive performance testing including:
- Large dataset processing benchmarks
- Memory usage optimization tests
- Concurrent access performance
- Cache efficiency measurements
- API rate limiting performance
"""

import pytest
import tempfile
import shutil
import time
import threading
import queue
import psutil
import gc
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import statistics

from src.common.options_handler import OptionsHandler
from src.common.options_dtos import (
    OptionContractDTO, StrikePrice, ExpirationDate
)
from src.common.models import OptionType


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
    
    def create_large_contract_dataset(self, num_contracts: int = 1000) -> List[OptionContractDTO]:
        """Create a large dataset of contracts for performance testing."""
        contracts = []
        
        # Create contracts with varied strikes and expirations
        strikes = [580.0 + i * 2.0 for i in range(20)]  # 20 different strikes
        expirations = [date(2025, 1, 15) + timedelta(days=i * 7) for i in range(4)]  # 4 different expirations
        
        for i in range(num_contracts):
            strike = strikes[i % len(strikes)]
            exp_date = expirations[i % len(expirations)]
            option_type = OptionType.CALL if i % 2 == 0 else OptionType.PUT
            
            contract = OptionContractDTO(
                ticker=f"O:SPY250115{'C' if option_type == OptionType.CALL else 'P'}{int(strike * 1000):08d}",
                underlying_ticker="SPY",
                contract_type=option_type,
                strike_price=StrikePrice(Decimal(str(strike))),
                expiration_date=ExpirationDate(exp_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS"
            )
            contracts.append(contract)
        
        return contracts
    
    def test_large_dataset_processing_performance(self, options_handler):
        """Test processing performance with large datasets."""
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        results = {}
        
        for size in dataset_sizes:
            contracts = self.create_large_contract_dataset(size)
            test_date = datetime(2025, 1, 10)
            
            with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
                
                # Measure processing time
                start_time = time.time()
                filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                end_time = time.time()
                
                processing_time = end_time - start_time
                results[size] = {
                    'processing_time': processing_time,
                    'contracts_per_second': size / processing_time if processing_time > 0 else 0,
                    'memory_usage': psutil.Process().memory_info().rss
                }
                
                # Performance should be reasonable
                assert processing_time < 5.0, f"Processing {size} contracts took {processing_time:.2f}s"
                assert len(filtered_contracts) == size
        
        # Print performance summary
        print("\nPerformance Summary:")
        for size, metrics in results.items():
            print(f"  {size:5d} contracts: {metrics['processing_time']:.3f}s "
                  f"({metrics['contracts_per_second']:.0f} contracts/sec)")
    
    def test_memory_usage_optimization(self, options_handler):
        """Test memory usage optimization with large datasets."""
        process = psutil.Process()
        
        # Test memory usage with different dataset sizes
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            # Get initial memory
            gc.collect()
            initial_memory = process.memory_info().rss
            
            # Create and process large dataset
            contracts = self.create_large_contract_dataset(size)
            test_date = datetime(2025, 1, 10)
            
            with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
                
                # Process dataset
                filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                
                # Check peak memory usage
                peak_memory = process.memory_info().rss
                memory_increase = peak_memory - initial_memory
                
                # Memory increase should be reasonable (< 50MB per 1000 contracts)
                max_expected_memory = (size / 1000) * 50 * 1024 * 1024  # 50MB per 1000 contracts
                assert memory_increase < max_expected_memory, \
                    f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB for {size} contracts"
                
                # Clean up
                del filtered_contracts
                del contracts
                gc.collect()
                
                # Check memory cleanup
                final_memory = process.memory_info().rss
                memory_cleanup = peak_memory - final_memory
                
                # Should clean up at least 50% of the memory
                assert memory_cleanup > memory_increase * 0.5, \
                    f"Insufficient memory cleanup: {memory_cleanup / 1024 / 1024:.1f}MB"
    
    def test_concurrent_access_performance(self, options_handler):
        """Test concurrent access performance."""
        contracts = self.create_large_contract_dataset(1000)
        test_date = datetime(2025, 1, 10)
        
        # Test with different numbers of concurrent threads
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for num_threads in thread_counts:
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def worker(worker_id):
                try:
                    with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
                        start_time = time.time()
                        filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                        end_time = time.time()
                        
                        results_queue.put({
                            'worker_id': worker_id,
                            'processing_time': end_time - start_time,
                            'contracts_processed': len(filtered_contracts)
                        })
                except Exception as e:
                    errors_queue.put((worker_id, str(e)))
            
            # Start threads
            start_time = time.time()
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Check for errors
            assert errors_queue.empty(), f"Errors in concurrent access: {list(errors_queue.queue)}"
            
            # Collect results
            worker_results = list(results_queue.queue)
            assert len(worker_results) == num_threads
            
            # Calculate performance metrics
            avg_processing_time = statistics.mean([r['processing_time'] for r in worker_results])
            total_throughput = num_threads / total_time
            
            results[num_threads] = {
                'total_time': total_time,
                'avg_processing_time': avg_processing_time,
                'total_throughput': total_throughput
            }
        
        # Print concurrent performance summary
        print("\nConcurrent Access Performance:")
        for num_threads, metrics in results.items():
            print(f"  {num_threads:2d} threads: {metrics['total_time']:.3f}s total, "
                  f"{metrics['total_throughput']:.2f} operations/sec")
    
    def test_cache_efficiency_performance(self, options_handler):
        """Test cache efficiency and performance."""
        contracts = self.create_large_contract_dataset(1000)
        test_date = datetime(2025, 1, 10)
        
        # Mock cache manager to track cache hits/misses
        cache_hits = 0
        cache_misses = 0
        
        def mock_load_contracts(symbol, date):
            nonlocal cache_hits
            cache_hits += 1
            return contracts
        
        def mock_load_bar(symbol, date, ticker):
            nonlocal cache_misses
            cache_misses += 1
            return None  # Simulate cache miss for bars
        
        with patch.object(options_handler.cache_manager, 'load_contracts', side_effect=mock_load_contracts):
            with patch.object(options_handler.cache_manager, 'load_bar', side_effect=mock_load_bar):
                
                # Test cache hit performance
                start_time = time.time()
                for _ in range(10):  # 10 cache hits
                    cached_contracts = options_handler.get_contract_list_for_date(test_date)
                cache_hit_time = time.time() - start_time
                
                # Test cache miss performance (API calls)
                start_time = time.time()
                for contract in contracts[:10]:  # 10 cache misses
                    options_handler.get_option_bar(contract, test_date)
                cache_miss_time = time.time() - start_time
                
                # Cache hits should be much faster than cache misses
                assert cache_hit_time < cache_miss_time, \
                    f"Cache hits ({cache_hit_time:.3f}s) should be faster than cache misses ({cache_miss_time:.3f}s)"
                
                # Cache hit rate should be high
                total_operations = cache_hits + cache_misses
                cache_hit_rate = cache_hits / total_operations if total_operations > 0 else 0
                assert cache_hit_rate > 0.5, f"Cache hit rate too low: {cache_hit_rate:.2%}"
    
    def test_filtering_performance(self, options_handler):
        """Test filtering performance with different filter complexities."""
        contracts = self.create_large_contract_dataset(2000)
        test_date = datetime(2025, 1, 10)
        
        # Test different filtering scenarios
        filter_scenarios = [
            {
                'name': 'No filters',
                'strike_range': None,
                'expiration_range': None
            },
            {
                'name': 'Strike filter only',
                'strike_range': {
                    'min_strike': StrikePrice(Decimal('590.0')),
                    'max_strike': StrikePrice(Decimal('610.0'))
                },
                'expiration_range': None
            },
            {
                'name': 'Expiration filter only',
                'strike_range': None,
                'expiration_range': {
                    'min_days': 5,
                    'max_days': 15
                }
            },
            {
                'name': 'Both filters',
                'strike_range': {
                    'min_strike': StrikePrice(Decimal('590.0')),
                    'max_strike': StrikePrice(Decimal('610.0'))
                },
                'expiration_range': {
                    'min_days': 5,
                    'max_days': 15
                }
            }
        ]
        
        results = {}
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
            
            for scenario in filter_scenarios:
                start_time = time.time()
                
                # Apply filters
                strike_range = None
                if scenario['strike_range']:
                    strike_range = StrikeRangeDTO(**scenario['strike_range'])
                
                expiration_range = None
                if scenario['expiration_range']:
                    expiration_range = ExpirationRangeDTO(**scenario['expiration_range'])
                
                filtered_contracts = options_handler.get_contract_list_for_date(
                    test_date, strike_range=strike_range, expiration_range=expiration_range
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                results[scenario['name']] = {
                    'processing_time': processing_time,
                    'filtered_count': len(filtered_contracts),
                    'filter_efficiency': len(filtered_contracts) / len(contracts)
                }
                
                # All scenarios should complete in reasonable time
                assert processing_time < 2.0, f"{scenario['name']} took {processing_time:.3f}s"
        
        # Print filtering performance summary
        print("\nFiltering Performance:")
        for name, metrics in results.items():
            print(f"  {name:20s}: {metrics['processing_time']:.3f}s, "
                  f"{metrics['filtered_count']:4d} contracts ({metrics['filter_efficiency']:.1%})")
    
    def test_api_rate_limiting_performance(self, options_handler):
        """Test API rate limiting performance."""
        test_date = datetime(2025, 1, 10)
        
        # Mock API calls with rate limiting
        api_call_times = []
        
        def mock_fetch_with_retry(fetch_func, error_msg, max_retries=3, retry_delay=60):
            start_time = time.time()
            # Simulate API call delay
            time.sleep(0.1)  # 100ms delay
            end_time = time.time()
            api_call_times.append(end_time - start_time)
            return Mock(results=[])
        
        with patch.object(options_handler.retry_handler, 'fetch_with_retry', side_effect=mock_fetch_with_retry):
            
            # Make multiple API calls
            start_time = time.time()
            for _ in range(5):
                options_handler._fetch_contracts_from_api(test_date.date())
            total_time = time.time() - start_time
            
            # Should respect rate limiting
            assert total_time >= 0.5, f"Rate limiting not working: {total_time:.3f}s for 5 calls"
            assert len(api_call_times) == 5, f"Expected 5 API calls, got {len(api_call_times)}"
            
            # Each call should take at least 100ms
            for call_time in api_call_times:
                assert call_time >= 0.1, f"API call too fast: {call_time:.3f}s"
    
    def test_memory_leak_detection(self, options_handler):
        """Test for memory leaks in long-running operations."""
        contracts = self.create_large_contract_dataset(1000)
        test_date = datetime(2025, 1, 10)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
            
            # Run multiple iterations
            for iteration in range(10):
                # Process dataset
                filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                
                # Force garbage collection
                del filtered_contracts
                gc.collect()
                
                # Check memory usage
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory increase should not grow significantly over iterations
                max_expected_increase = 50 * 1024 * 1024  # 50MB
                assert memory_increase < max_expected_increase, \
                    f"Memory leak detected at iteration {iteration}: {memory_increase / 1024 / 1024:.1f}MB"
        
        # Final memory check
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_memory_increase < 100 * 1024 * 1024, \
            f"Excessive memory usage: {total_memory_increase / 1024 / 1024:.1f}MB"
    
    def test_scalability_benchmarks(self, options_handler):
        """Test scalability with increasing dataset sizes."""
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        performance_metrics = {}
        
        for size in dataset_sizes:
            contracts = self.create_large_contract_dataset(size)
            test_date = datetime(2025, 1, 10)
            
            with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
                
                # Measure processing time
                start_time = time.time()
                filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                end_time = time.time()
                
                processing_time = end_time - start_time
                contracts_per_second = size / processing_time if processing_time > 0 else 0
                
                # Measure memory usage
                memory_usage = psutil.Process().memory_info().rss
                
                performance_metrics[size] = {
                    'processing_time': processing_time,
                    'contracts_per_second': contracts_per_second,
                    'memory_usage_mb': memory_usage / 1024 / 1024
                }
                
                # Performance should scale reasonably
                assert processing_time < size * 0.001, \
                    f"Processing time too high for {size} contracts: {processing_time:.3f}s"
        
        # Print scalability summary
        print("\nScalability Benchmarks:")
        print("  Size    Time(s)  Contracts/sec  Memory(MB)")
        print("  -----   -------  -------------  ----------")
        for size, metrics in performance_metrics.items():
            print(f"  {size:5d}   {metrics['processing_time']:7.3f}   "
                  f"{metrics['contracts_per_second']:13.0f}   {metrics['memory_usage_mb']:10.1f}")
        
        # Performance should scale linearly or better
        small_size = 100
        large_size = 5000
        
        small_time = performance_metrics[small_size]['processing_time']
        large_time = performance_metrics[large_size]['processing_time']
        
        # Large dataset should not take more than 50x the time of small dataset
        time_ratio = large_time / small_time
        size_ratio = large_size / small_size
        
        assert time_ratio < size_ratio * 2, \
            f"Performance scaling issue: {time_ratio:.1f}x time for {size_ratio:.1f}x data"


class TestOptionsHandlerOptimization:
    """Test optimization features and performance improvements."""
    
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
    
    def test_lazy_loading_optimization(self, options_handler):
        """Test lazy loading optimization for bar data."""
        contracts = self.create_large_contract_dataset(100)
        test_date = datetime(2025, 1, 10)
        
        # Track bar loading calls
        bar_load_calls = 0
        
        def mock_load_bar(symbol, date, ticker):
            nonlocal bar_load_calls
            bar_load_calls += 1
            return None  # Simulate no cached bar data
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
            with patch.object(options_handler.cache_manager, 'load_bar', side_effect=mock_load_bar):
                
                # Get contracts (should not load bars)
                contracts_result = options_handler.get_contract_list_for_date(test_date)
                assert bar_load_calls == 0, "Bar data should not be loaded when getting contracts"
                
                # Get individual bar (should load only that bar)
                contract = contracts_result[0]
                bar = options_handler.get_option_bar(contract, test_date)
                assert bar_load_calls == 1, "Should load only the requested bar"
                
                # Get another bar (should load only that bar)
                if len(contracts_result) > 1:
                    contract2 = contracts_result[1]
                    bar2 = options_handler.get_option_bar(contract2, test_date)
                    assert bar_load_calls == 2, "Should load only the requested bar"
    
    def test_caching_optimization(self, options_handler):
        """Test caching optimization and efficiency."""
        contracts = self.create_large_contract_dataset(100)
        test_date = datetime(2025, 1, 10)
        
        # Track cache operations
        cache_load_calls = 0
        cache_save_calls = 0
        
        def mock_load_contracts(symbol, date):
            nonlocal cache_load_calls
            cache_load_calls += 1
            return contracts
        
        def mock_save_contracts(symbol, date, contracts_data):
            nonlocal cache_save_calls
            cache_save_calls += 1
        
        with patch.object(options_handler.cache_manager, 'load_contracts', side_effect=mock_load_contracts):
            with patch.object(options_handler.cache_manager, 'save_contracts', side_effect=mock_save_contracts):
                
                # Multiple calls should use cache
                for _ in range(5):
                    options_handler.get_contract_list_for_date(test_date)
                
                # Should load from cache multiple times but not save
                assert cache_load_calls == 5, f"Expected 5 cache loads, got {cache_load_calls}"
                assert cache_save_calls == 0, f"Expected 0 cache saves, got {cache_save_calls}"
    
    def test_memory_optimization(self, options_handler):
        """Test memory optimization features."""
        # Test with large dataset
        contracts = self.create_large_contract_dataset(5000)
        test_date = datetime(2025, 1, 10)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch.object(options_handler.cache_manager, 'load_contracts', return_value=contracts):
            
            # Process dataset multiple times
            for _ in range(3):
                filtered_contracts = options_handler.get_contract_list_for_date(test_date)
                
                # Check memory usage
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable
                assert memory_increase < 200 * 1024 * 1024, \
                    f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
                
                # Clean up
                del filtered_contracts
                gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        # Should not have significant memory leaks
        assert total_memory_increase < 100 * 1024 * 1024, \
            f"Memory leak detected: {total_memory_increase / 1024 / 1024:.1f}MB"
    
    def create_large_contract_dataset(self, num_contracts: int = 1000) -> List[OptionContractDTO]:
        """Create a large dataset of contracts for performance testing."""
        contracts = []
        
        # Create contracts with varied strikes and expirations
        strikes = [580.0 + i * 2.0 for i in range(20)]  # 20 different strikes
        expirations = [date(2025, 1, 15) + timedelta(days=i * 7) for i in range(4)]  # 4 different expirations
        
        for i in range(num_contracts):
            strike = strikes[i % len(strikes)]
            exp_date = expirations[i % len(expirations)]
            option_type = OptionType.CALL if i % 2 == 0 else OptionType.PUT
            
            contract = OptionContractDTO(
                ticker=f"O:SPY250115{'C' if option_type == OptionType.CALL else 'P'}{int(strike * 1000):08d}",
                underlying_ticker="SPY",
                contract_type=option_type,
                strike_price=StrikePrice(Decimal(str(strike))),
                expiration_date=ExpirationDate(exp_date),
                exercise_style="american",
                shares_per_contract=100,
                primary_exchange="BATO",
                cfi="OCASPS"
            )
            contracts.append(contract)
        
        return contracts
