"""
Tests for Moving Average Velocity Analysis

This module tests the MA velocity analysis functionality to ensure it works correctly
with the existing data retrieval infrastructure.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from algo_trading_engine.analysis.ma_velocity_analysis import MAVelocityAnalyzer, TrendSignal, MAVelocityResult


class TestMAVelocityAnalysis:
    """Test cases for MA velocity analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MAVelocityAnalyzer(symbol='SPY')
        
        # Create mock data for testing
        dates = pd.date_range('2023-07-01', periods=180, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic price data with clear trends
        base_price = 400
        prices = []
        
        for i in range(180):
            if i < 30:
                # Initial uptrend
                price = base_price + i * 1.0 + np.random.normal(0, 0.5)
            elif i < 60:
                # Downtrend
                price = base_price + 30 - (i - 30) * 1.2 + np.random.normal(0, 0.5)
            elif i < 90:
                # Strong uptrend
                price = base_price + 6 + (i - 60) * 1.5 + np.random.normal(0, 0.5)
            elif i < 120:
                # Downtrend
                price = base_price + 51 - (i - 90) * 1.3 + np.random.normal(0, 0.5)
            elif i < 150:
                # Uptrend
                price = base_price + 12 + (i - 120) * 1.1 + np.random.normal(0, 0.5)
            else:
                # Sideways with noise
                price = base_price + 45 + np.random.normal(0, 1.0)
            
            prices.append(max(price, 1))  # Ensure positive prices
        
        self.mock_data = pd.DataFrame({
            'Close': prices,
            'Open': [p * 0.999 for p in prices],
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Volume': np.random.randint(1000000, 10000000, 180)
        }, index=dates)
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        assert self.analyzer.symbol == 'SPY'
        assert self.analyzer.data is None
        assert self.analyzer.trend_signals == []
        
        # Check that start_date is approximately 6 months ago
        expected_start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        actual_start = self.analyzer.start_date
        assert actual_start[:7] == expected_start[:7]  # Check year and month
    
    def test_calculate_moving_averages(self):
        """Test moving average calculation."""
        short_periods = [5, 10]
        long_periods = [20, 50]
        
        result = self.analyzer.calculate_moving_averages(
            self.mock_data, short_periods, long_periods
        )
        
        # Check that all MAs were calculated
        for period in short_periods + long_periods:
            assert f'SMA_{period}' in result.columns
        
        # Check that MAs are reasonable (not all NaN)
        for period in short_periods + long_periods:
            ma_col = f'SMA_{period}'
            assert not result[ma_col].isna().all()
            assert len(result[ma_col].dropna()) > 0
    
    def test_identify_trend_signals(self):
        """Test trend signal identification."""
        # Add some MAs to the data first
        data_with_ma = self.analyzer.calculate_moving_averages(
            self.mock_data, [10], [20]
        )
        
        signals = self.analyzer.identify_trend_signals(data_with_ma, [10], [20])
        
        # Should identify some signals
        assert isinstance(signals, list)
        assert len(signals) > 0
        
        # Check signal structure
        for signal in signals:
            assert isinstance(signal, TrendSignal)
            assert signal.signal_type in ['up', 'down']
            assert signal.short_ma == 10
            assert signal.long_ma == 20
            assert signal.ma_velocity > 0
    
    def test_check_trend_success(self):
        """Test trend success checking."""
        # Create data with a clear upward trend
        trend_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        })
        
        # Test upward trend success
        success, duration, trend_return = self.analyzer._check_trend_success(
            trend_data, 0, 'up', min_duration=3, max_duration=10
        )
        
        assert success is True
        assert duration >= 3
        assert trend_return > 0
        
        # Test downward trend
        down_trend_data = pd.DataFrame({
            'Close': [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        })
        
        success, duration, trend_return = self.analyzer._check_trend_success(
            down_trend_data, 0, 'down', min_duration=3, max_duration=10
        )
        
        assert success is True
        assert duration >= 3
        assert trend_return < 0
    
    def test_calculate_success_rates(self):
        """Test success rate calculation."""
        # Create mock signals
        signals = [
            TrendSignal(
                signal_date=pd.Timestamp('2023-07-01'),
                ma_velocity=1.05,
                short_ma=10,
                long_ma=20,
                signal_type='up',
                success=True,
                trend_duration=5,
                trend_return=0.05
            ),
            TrendSignal(
                signal_date=pd.Timestamp('2023-07-05'),
                ma_velocity=0.95,
                short_ma=10,
                long_ma=20,
                signal_type='up',
                success=False,
                trend_duration=0,
                trend_return=0.0
            ),
            TrendSignal(
                signal_date=pd.Timestamp('2023-07-08'),
                ma_velocity=1.02,
                short_ma=10,
                long_ma=20,
                signal_type='up',
                success=True,
                trend_duration=4,
                trend_return=0.03
            ),
            TrendSignal(
                signal_date=pd.Timestamp('2023-07-10'),
                ma_velocity=0.90,
                short_ma=10,
                long_ma=20,
                signal_type='down',
                success=True,
                trend_duration=4,
                trend_return=-0.03
            )
        ]
        
        results = self.analyzer.calculate_success_rates(signals)
        
        # Should have results for both directions
        assert 'up' in results
        assert 'down' in results
        
        # Check upward trend results
        up_results = results['up']
        assert len(up_results) == 1
        up_result = up_results[0]
        assert up_result.short_ma == 10
        assert up_result.long_ma == 20
        assert up_result.success_rate == 2/3  # 2 successes out of 3 signals
        assert up_result.total_signals == 3
        assert up_result.successful_signals == 2
    
    def test_find_optimal_ma_combinations(self):
        """Test finding optimal MA combinations."""
        # Use the mock data
        self.analyzer.data = self.mock_data
        
        optimal_combinations = self.analyzer.find_optimal_ma_combinations(
            short_periods=[5, 10], 
            long_periods=[20, 30]
        )
        
        # Should return a dictionary
        assert isinstance(optimal_combinations, dict)
        
        # May have results for up, down, or both directions
        for trend_type, result in optimal_combinations.items():
            assert trend_type in ['up', 'down']
            assert isinstance(result, MAVelocityResult)
            assert result.short_ma < result.long_ma
            assert 0 <= result.success_rate <= 1
    
    def test_generate_report(self):
        """Test report generation."""
        # Create mock optimal combinations
        optimal_combinations = {
            'up': MAVelocityResult(
                short_ma=10,
                long_ma=20,
                velocity=0.5,
                success_rate=0.75,
                total_signals=8,
                successful_signals=6,
                avg_trend_duration=5.5,
                avg_trend_return=0.05
            ),
            'down': MAVelocityResult(
                short_ma=5,
                long_ma=30,
                velocity=0.167,
                success_rate=0.60,
                total_signals=5,
                successful_signals=3,
                avg_trend_duration=4.2,
                avg_trend_return=-0.03
            )
        }
        
        # Add some mock signals
        self.analyzer.trend_signals = [
            TrendSignal(
                signal_date=pd.Timestamp('2023-07-01'),
                ma_velocity=1.05,
                short_ma=10,
                long_ma=20,
                signal_type='up',
                success=True,
                trend_duration=5,
                trend_return=0.05
            )
        ]
        self.analyzer.data = self.mock_data
        
        report = self.analyzer.generate_report(optimal_combinations)
        
        # Check report content
        assert isinstance(report, str)
        assert "MOVING AVERAGE VELOCITY/ELASTICITY ANALYSIS REPORT" in report
        assert "SPY" in report
        assert "UPWARD TREND SIGNALS" in report
        assert "DOWNWARD TREND SIGNALS" in report
        assert "SMA 10/20" in report  # MA combination for upward
        # Note: SMA 5/30 appears in optimal combinations section, not in longest trends


def test_trend_signal_dataclass():
    """Test TrendSignal dataclass."""
    signal = TrendSignal(
        signal_date=pd.Timestamp('2023-07-01'),
        ma_velocity=1.05,
        short_ma=10,
        long_ma=20,
        signal_type='up',
        success=True,
        trend_duration=5,
        trend_return=0.05
    )
    
    assert signal.signal_date == pd.Timestamp('2023-07-01')
    assert signal.ma_velocity == 1.05
    assert signal.short_ma == 10
    assert signal.long_ma == 20
    assert signal.signal_type == 'up'
    assert signal.success is True
    assert signal.trend_duration == 5
    assert signal.trend_return == 0.05


def test_ma_velocity_result_dataclass():
    """Test MAVelocityResult dataclass."""
    result = MAVelocityResult(
        short_ma=10,
        long_ma=20,
        velocity=0.5,
        success_rate=0.75,
        total_signals=8,
        successful_signals=6,
        avg_trend_duration=5.5,
        avg_trend_return=0.05
    )
    
    assert result.short_ma == 10
    assert result.long_ma == 20
    assert result.velocity == 0.5
    assert result.success_rate == 0.75
    assert result.total_signals == 8
    assert result.successful_signals == 6
    assert result.avg_trend_duration == 5.5
    assert result.avg_trend_return == 0.05


if __name__ == "__main__":
    pytest.main([__file__])
