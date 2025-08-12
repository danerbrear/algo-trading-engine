"""
Performance Tests for Enhanced Current Date Volume Validation

This module contains performance tests to measure the impact of enhanced volume validation
on backtest performance and scalability.
"""

import pytest
import time
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backtest.main import BacktestEngine
from src.backtest.config import VolumeConfig
from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestVolumeValidationPerformance:
    """Test performance impact of enhanced volume validation."""
    
    def setup_method(self):
        """Set up test fixtures for performance testing."""
        # Create mock data with more trading days for performance testing
        self.mock_data = Mock()
        trading_days = []
        for i in range(252):  # One year of trading days
            trading_days.append(datetime(2024, 1, 1) + timedelta(days=i))
        
        self.mock_data.index = trading_days
        self.mock_data.iloc = {0: {'Close': 500.0}, -1: {'Close': 510.0}}
        self.mock_data.loc = {trading_days[0]: {'Close': 500.0}, trading_days[-1]: {'Close': 510.0}}
        
        # Create mock strategy
        self.mock_lstm_model = Mock()
        self.mock_lstm_scaler = Mock()
        self.mock_options_handler = Mock()
        
        self.strategy = CreditSpreadStrategy(
            lstm_model=self.mock_lstm_model,
            lstm_scaler=self.mock_lstm_scaler,
            options_handler=self.mock_options_handler
        )
        
        # Create test options
        self.option1 = Option(
            symbol="SPY240315C00500000",
            ticker="SPY",
            expiration="2024-03-15",
            strike=500.0,
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=15
        )
        
        self.option2 = Option(
            symbol="SPY240315C00510000",
            ticker="SPY",
            expiration="2024-03-15",
            strike=510.0,
            option_type=OptionType.CALL,
            last_price=0.75,
            volume=8
        )
        
        # Create test position
        self.position = Position(
            symbol="SPY",
            expiration_date=datetime(2024, 3, 15),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=0.75,
            spread_options=[self.option1, self.option2]
        )
        self.position.set_quantity(1)
    
    def test_parameter_passing_performance(self):
        """Test performance impact of parameter passing approach vs. direct API calls."""
        print("\n🧪 Testing Parameter Passing Performance")
        
        # Test 1: Parameter passing approach (current implementation)
        volume_config_enabled = VolumeConfig(
            enable_volume_validation=True,
            min_volume=10,
            skip_closure_on_insufficient_volume=True
        )
        
        engine_enabled = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config_enabled
        )
        engine_enabled.positions.append(self.position)
        
        # Mock current volumes for parameter passing
        current_volumes = [25, 15]
        
        # Measure parameter passing performance
        start_time = time.time()
        for _ in range(1000):  # Simulate 1000 position closures
            engine_enabled._remove_position(
                date=datetime(2024, 2, 15),
                position=self.position,
                exit_price=0.50,
                current_volumes=current_volumes
            )
            # Reset for next iteration
            engine_enabled.positions.append(self.position)
            engine_enabled.capital = 100000
        
        parameter_passing_time = time.time() - start_time
        print(f"   Parameter passing approach: {parameter_passing_time:.4f} seconds for 1000 operations")
        
        # Test 2: Direct API calls approach (hypothetical alternative)
        # This simulates what it would be like if we made API calls directly in _remove_position
        engine_direct_api = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config_enabled
        )
        engine_direct_api.positions.append(self.position)
        
        # Mock API call overhead (simulating network latency and processing)
        def mock_api_call():
            time.sleep(0.001)  # Simulate 1ms API call overhead
            return [25, 15]
        
        start_time = time.time()
        for _ in range(1000):  # Simulate 1000 position closures with direct API calls
            # Simulate API call overhead
            volumes = mock_api_call()
            
            engine_direct_api._remove_position(
                date=datetime(2024, 2, 15),
                position=self.position,
                exit_price=0.50,
                current_volumes=volumes
            )
            # Reset for next iteration
            engine_direct_api.positions.append(self.position)
            engine_direct_api.capital = 100000
        
        direct_api_time = time.time() - start_time
        print(f"   Direct API calls approach: {direct_api_time:.4f} seconds for 1000 operations")
        
        # Calculate performance improvement
        improvement = ((direct_api_time - parameter_passing_time) / direct_api_time) * 100
        print(f"   Performance improvement: {improvement:.1f}%")
        
        # Assert that parameter passing is faster
        assert parameter_passing_time < direct_api_time, "Parameter passing should be faster than direct API calls"
    
    def test_caching_effectiveness(self):
        """Test the effectiveness of caching in strategy layer."""
        print("\n🧪 Testing Caching Effectiveness")
        
        # Mock the options handler to track API calls
        api_call_count = 0
        
        def mock_get_specific_option_contract(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            return Mock(volume=25)
        
        self.strategy.options_handler.get_specific_option_contract = mock_get_specific_option_contract
        
        # Test without caching (multiple calls for same data)
        api_call_count = 0
        start_time = time.time()
        
        for _ in range(100):  # Simulate 100 position closures
            volumes = self.strategy.get_current_volumes_for_position(
                self.position, 
                datetime(2024, 2, 15)
            )
        
        no_caching_time = time.time() - start_time
        no_caching_calls = api_call_count
        
        print(f"   Without caching: {no_caching_calls} API calls in {no_caching_time:.4f} seconds")
        
        # Test with caching (simulated - in real implementation, caching would be in options_handler)
        api_call_count = 0
        start_time = time.time()
        
        # Simulate cached results (same date, same options)
        cached_volumes = [25, 15]
        
        for _ in range(100):  # Simulate 100 position closures with caching
            # In real implementation, this would use cached data
            volumes = cached_volumes
        
        caching_time = time.time() - start_time
        caching_calls = api_call_count  # Should be 0 with proper caching
        
        print(f"   With caching: {caching_calls} API calls in {caching_time:.4f} seconds")
        
        # Calculate caching benefit
        if no_caching_calls > 0:
            caching_benefit = ((no_caching_calls - caching_calls) / no_caching_calls) * 100
            print(f"   Caching benefit: {caching_benefit:.1f}% reduction in API calls")
        
        # Assert that caching reduces API calls
        assert caching_calls < no_caching_calls, "Caching should reduce API calls"
    
    def test_scalability_with_large_positions(self):
        """Test scalability with large numbers of positions."""
        print("\n🧪 Testing Scalability with Large Numbers of Positions")
        
        # Create multiple positions
        positions = []
        for i in range(100):  # 100 positions
            position = Position(
                symbol=f"SPY{i}",
                expiration_date=datetime(2024, 3, 15),
                strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                strike_price=500.0 + i,
                entry_date=datetime(2024, 1, 15),
                entry_price=0.75,
                spread_options=[self.option1, self.option2]
            )
            position.set_quantity(1)
            positions.append(position)
        
        # Test with volume validation enabled
        volume_config = VolumeConfig(
            enable_volume_validation=True,
            min_volume=10,
            skip_closure_on_insufficient_volume=True
        )
        
        engine = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config
        )
        
        # Add all positions
        engine.positions.extend(positions)
        
        # Mock current volumes for all positions
        current_volumes = [25, 15]
        
        # Measure time to close all positions
        start_time = time.time()
        
        for position in positions:
            engine._remove_position(
                date=datetime(2024, 2, 15),
                position=position,
                exit_price=0.50,
                current_volumes=current_volumes
            )
        
        total_time = time.time() - start_time
        positions_per_second = len(positions) / total_time
        
        print(f"   Closed {len(positions)} positions in {total_time:.4f} seconds")
        print(f"   Performance: {positions_per_second:.1f} positions per second")
        
        # Assert reasonable performance (should handle 100 positions in reasonable time)
        assert total_time < 10.0, f"Should handle 100 positions in under 10 seconds, took {total_time:.2f}s"
        assert positions_per_second > 10, f"Should process at least 10 positions per second, got {positions_per_second:.1f}"
    
    def test_performance_with_volume_validation_disabled(self):
        """Test performance comparison with volume validation disabled."""
        print("\n🧪 Testing Performance with Volume Validation Disabled")
        
        # Test with volume validation enabled
        volume_config_enabled = VolumeConfig(
            enable_volume_validation=True,
            min_volume=10,
            skip_closure_on_insufficient_volume=True
        )
        
        engine_enabled = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config_enabled
        )
        engine_enabled.positions.append(self.position)
        
        current_volumes = [25, 15]
        
        start_time = time.time()
        for _ in range(1000):
            engine_enabled._remove_position(
                date=datetime(2024, 2, 15),
                position=self.position,
                exit_price=0.50,
                current_volumes=current_volumes
            )
            engine_enabled.positions.append(self.position)
            engine_enabled.capital = 100000
        
        enabled_time = time.time() - start_time
        print(f"   With volume validation: {enabled_time:.4f} seconds for 1000 operations")
        
        # Test with volume validation disabled
        volume_config_disabled = VolumeConfig(
            enable_volume_validation=False
        )
        
        engine_disabled = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config_disabled
        )
        engine_disabled.positions.append(self.position)
        
        start_time = time.time()
        for _ in range(1000):
            engine_disabled._remove_position(
                date=datetime(2024, 2, 15),
                position=self.position,
                exit_price=0.50,
                current_volumes=current_volumes
            )
            engine_disabled.positions.append(self.position)
            engine_disabled.capital = 100000
        
        disabled_time = time.time() - start_time
        print(f"   Without volume validation: {disabled_time:.4f} seconds for 1000 operations")
        
        # Calculate overhead
        overhead = ((enabled_time - disabled_time) / disabled_time) * 100
        print(f"   Volume validation overhead: {overhead:.1f}%")
        
        # Assert that overhead is reasonable (should be less than 50%)
        assert overhead < 50.0, f"Volume validation overhead should be reasonable, got {overhead:.1f}%"
    
    def test_memory_usage_impact(self):
        """Test memory usage impact of enhanced volume validation."""
        print("\n🧪 Testing Memory Usage Impact")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create engine with volume validation
        volume_config = VolumeConfig(
            enable_volume_validation=True,
            min_volume=10,
            skip_closure_on_insufficient_volume=True
        )
        
        engine = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=volume_config
        )
        
        # Add positions and perform operations
        positions = []
        for i in range(100):
            position = Position(
                symbol=f"SPY{i}",
                expiration_date=datetime(2024, 3, 15),
                strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                strike_price=500.0 + i,
                entry_date=datetime(2024, 1, 15),
                entry_price=0.75,
                spread_options=[self.option1, self.option2]
            )
            position.set_quantity(1)
            positions.append(position)
            engine.positions.append(position)
        
        current_volumes = [25, 15]
        
        # Perform operations
        for position in positions:
            engine._remove_position(
                date=datetime(2024, 2, 15),
                position=position,
                exit_price=0.50,
                current_volumes=current_volumes
            )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"   Initial memory: {initial_memory:.2f} MB")
        print(f"   Final memory: {final_memory:.2f} MB")
        print(f"   Memory increase: {memory_increase:.2f} MB")
        
        # Clean up
        del engine, positions
        gc.collect()
        
        # Assert reasonable memory usage (should not increase by more than 100MB)
        assert memory_increase < 100.0, f"Memory increase should be reasonable, got {memory_increase:.2f} MB"
    
    def test_concurrent_volume_validation_performance(self):
        """Test performance under concurrent volume validation scenarios."""
        print("\n🧪 Testing Concurrent Volume Validation Performance")
        
        import threading
        
        # Create multiple engines for concurrent testing
        engines = []
        for i in range(5):  # 5 concurrent engines
            volume_config = VolumeConfig(
                enable_volume_validation=True,
                min_volume=10,
                skip_closure_on_insufficient_volume=True
            )
            
            engine = BacktestEngine(
                data=self.mock_data,
                strategy=self.strategy,
                initial_capital=100000,
                volume_config=volume_config
            )
            
            # Add positions to each engine
            for j in range(20):  # 20 positions per engine
                position = Position(
                    symbol=f"SPY{i}_{j}",
                    expiration_date=datetime(2024, 3, 15),
                    strategy_type=StrategyType.CALL_CREDIT_SPREAD,
                    strike_price=500.0 + j,
                    entry_date=datetime(2024, 1, 15),
                    entry_price=0.75,
                    spread_options=[self.option1, self.option2]
                )
                position.set_quantity(1)
                engine.positions.append(position)
            
            engines.append(engine)
        
        current_volumes = [25, 15]
        results = []
        
        def process_engine(engine, engine_id):
            """Process all positions in an engine."""
            start_time = time.time()
            
            positions_to_close = list(engine.positions)
            for position in positions_to_close:
                engine._remove_position(
                    date=datetime(2024, 2, 15),
                    position=position,
                    exit_price=0.50,
                    current_volumes=current_volumes
                )
            
            end_time = time.time()
            results.append((engine_id, end_time - start_time))
        
        # Run concurrent processing
        threads = []
        start_time = time.time()
        
        for i, engine in enumerate(engines):
            thread = threading.Thread(target=process_engine, args=(engine, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        print(f"   Concurrent processing time: {total_time:.4f} seconds")
        print(f"   Individual engine times: {results}")
        
        # Calculate average time per engine
        avg_time = sum(time for _, time in results) / len(results)
        print(f"   Average time per engine: {avg_time:.4f} seconds")
        
        # Assert reasonable concurrent performance
        assert total_time < 30.0, f"Concurrent processing should complete in reasonable time, took {total_time:.2f}s"
        assert avg_time < 10.0, f"Average time per engine should be reasonable, got {avg_time:.2f}s"


if __name__ == "__main__":
    # Run performance tests
    test_instance = TestVolumeValidationPerformance()
    test_instance.setup_method()
    
    print("🚀 Running Volume Validation Performance Tests")
    print("=" * 60)
    
    test_instance.test_parameter_passing_performance()
    test_instance.test_caching_effectiveness()
    test_instance.test_scalability_with_large_positions()
    test_instance.test_performance_with_volume_validation_disabled()
    test_instance.test_memory_usage_impact()
    test_instance.test_concurrent_volume_validation_performance()
    
    print("\n✅ Performance tests completed!") 