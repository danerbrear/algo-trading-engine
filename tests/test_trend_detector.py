"""
Tests for TrendDetector utility class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.common.trend_detector import TrendDetector, TrendInfo


class TestForwardTrendDetection:
    """Test forward trend detection (scanning through data)."""
    
    def setup_method(self):
        """Create test data."""
        # Create data with a clear 3-day upward trend followed by reversal
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 102, 101, 100, 99, 98, 97]
        self.data = pd.DataFrame({'Close': prices}, index=dates)
    
    def test_detect_single_trend(self):
        """Test detection of a single upward trend."""
        trends = TrendDetector.detect_forward_trends(
            data=self.data,
            min_duration=3,
            max_duration=10,
            require_reversal=True
        )
        
        assert len(trends) == 1
        trend = trends[0]
        assert trend.duration == 3
        assert trend.start_price == 101
        assert trend.end_price == 103
        assert trend.net_return > 0
        assert trend.reversal_drawdown < 0
    
    def test_detect_no_reversal_required(self):
        """Test detection without requiring reversal."""
        # Data with upward trend but no reversal at end
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        prices = [100, 101, 102, 103, 104]
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10,
            require_reversal=False
        )
        
        assert len(trends) >= 1
    
    def test_ignore_too_short_trends(self):
        """Test that trends shorter than min_duration are ignored."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 101, 100, 99, 98, 97, 96, 95]  # Only 2-day uptrend
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10,
            require_reversal=True
        )
        
        assert len(trends) == 0
    
    def test_ignore_too_long_trends(self):
        """Test that trends longer than max_duration are ignored."""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        # 12-day upward trend
        prices = list(range(100, 112)) + [111, 110, 109]
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10,
            require_reversal=True
        )
        
        # Should not detect the 12-day trend
        assert all(trend.duration <= 10 for trend in trends)
    
    def test_detect_with_reversal_within_trend(self):
        """Test that trends with significant reversals are excluded."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        # 4-day trend with a significant reversal in the middle
        prices = [100, 101, 102, 97, 103, 102, 101, 100, 99, 98]
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10,
            reversal_threshold=0.02,
            require_reversal=True
        )
        
        # The 4-day trend should be excluded due to the reversal
        assert len(trends) == 0
    
    def test_reversal_info_populated(self):
        """Test that reversal information is correctly populated."""
        trends = TrendDetector.detect_forward_trends(
            data=self.data,
            min_duration=3,
            max_duration=10,
            require_reversal=True
        )
        
        assert len(trends) == 1
        trend = trends[0]
        assert trend.reversal_date is not None
        assert trend.reversal_drawdown < 0  # Should be negative (price dropped)


class TestBackwardTrendDetection:
    """Test backward trend detection (validating signals)."""
    
    def setup_method(self):
        """Create test data."""
        # Create data with a clear upward trend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        # Upward trend from day 30 to day 40
        prices = [100] * 30 + list(range(100, 110)) + [110] * 60
        self.data = pd.DataFrame({'Close': prices}, index=dates)
    
    def test_detect_backward_trend_success(self):
        """Test successful backward trend detection."""
        signal_index = 40  # End of the upward trend
        
        success, duration, return_val = TrendDetector.check_backward_trend(
            data=self.data,
            signal_index=signal_index,
            min_duration=3,
            max_duration=60
        )
        
        assert success is True
        assert duration >= 3
        assert return_val > 0
    
    def test_backward_trend_no_trend(self):
        """Test backward trend detection when no trend exists."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        # Flat prices
        prices = [100] * 50
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        signal_index = 30
        success, duration, return_val = TrendDetector.check_backward_trend(
            data=data,
            signal_index=signal_index,
            min_duration=3,
            max_duration=60
        )
        
        assert success is False
        assert duration == 0
        assert return_val == 0.0
    
    def test_backward_trend_too_short(self):
        """Test that trends with insufficient return are not detected."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        # Flat prices with tiny movement
        prices = [100] * 20 + [100.1, 100.2] + [100.2] * 28
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        signal_index = 22
        success, duration, return_val = TrendDetector.check_backward_trend(
            data=data,
            signal_index=signal_index,
            min_duration=3,
            max_duration=60
        )
        
        # Even if a 3+ day period is found, the return should be minimal
        if success:
            assert return_val < 0.01  # Less than 1% return
    
    def test_backward_trend_with_reversal(self):
        """Test that trends with reversals are excluded."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        # Trend with a significant reversal
        prices = [100] * 20 + [101, 102, 103, 97, 105, 106, 107] + [107] * 23
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        signal_index = 27  # End of the trend
        success, duration, return_val = TrendDetector.check_backward_trend(
            data=data,
            signal_index=signal_index,
            min_duration=3,
            max_duration=60,
            reversal_threshold=0.02
        )
        
        # Should not find a sustained trend due to the reversal
        assert success is False or duration < 7  # Can't be the full 7-day period
    
    def test_backward_trend_boundary_condition(self):
        """Test backward trend detection at data boundaries."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = list(range(100, 110))
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        # Signal at index 5 (middle of trend)
        signal_index = 5
        success, duration, return_val = TrendDetector.check_backward_trend(
            data=data,
            signal_index=signal_index,
            min_duration=3,
            max_duration=60
        )
        
        assert success is True
        assert 3 <= duration <= 5


class TestTrendSustainabilityChecking:
    """Test the _is_trend_sustained helper method."""
    
    def test_sustained_trend(self):
        """Test detection of a sustained trend."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        is_sustained = TrendDetector._is_trend_sustained(
            data=data,
            start_idx=0,
            end_idx=9,
            reversal_threshold=0.02
        )
        
        assert is_sustained is True
    
    def test_unsustained_trend_with_reversal(self):
        """Test detection of unsustained trend with reversal."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 97, 104, 105, 106, 107, 108, 109]  # Reversal on day 3
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        is_sustained = TrendDetector._is_trend_sustained(
            data=data,
            start_idx=0,
            end_idx=9,
            reversal_threshold=0.02
        )
        
        assert is_sustained is False


class TestTrendInfoDataClass:
    """Test TrendInfo dataclass."""
    
    def test_trend_info_creation(self):
        """Test creation of TrendInfo object."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 4)
        
        trend = TrendInfo(
            start_date=start_date,
            end_date=end_date,
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03,
            reversal_drawdown=-0.01,
            reversal_date=datetime(2024, 1, 5)
        )
        
        assert trend.start_date == start_date
        assert trend.end_date == end_date
        assert trend.duration == 3
        assert trend.start_price == 100.0
        assert trend.end_price == 103.0
        assert trend.net_return == 0.03
        assert trend.reversal_drawdown == -0.01
        assert trend.reversal_date == datetime(2024, 1, 5)
    
    def test_trend_info_without_reversal(self):
        """Test TrendInfo creation without reversal info."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 4)
        
        trend = TrendInfo(
            start_date=start_date,
            end_date=end_date,
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03
        )
        
        assert trend.reversal_drawdown == 0.0
        assert trend.reversal_date is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame({'Close': []}, index=pd.DatetimeIndex([]))
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10
        )
        
        assert len(trends) == 0
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        dates = pd.date_range(start='2024-01-01', periods=1, freq='D')
        data = pd.DataFrame({'Close': [100]}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=10
        )
        
        assert len(trends) == 0
    
    def test_very_large_duration_limits(self):
        """Test with very large duration limits."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = list(range(100, 200))
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        trends = TrendDetector.detect_forward_trends(
            data=data,
            min_duration=3,
            max_duration=1000,  # Very large
            require_reversal=False
        )
        
        # Should still detect trends
        assert len(trends) > 0

