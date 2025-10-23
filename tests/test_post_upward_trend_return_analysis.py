"""
Unit tests for Post-Upward Trend Return Analysis.

Tests the functionality of the PostUpwardTrendReturnAnalyzer class, including:
- Trend identification
- Post-trend return calculation
- Statistical analysis
- Report generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)

from src.analysis.post_upward_trend_return_analysis import (
    PostUpwardTrendReturnAnalyzer,
    UpwardTrend,
    PostTrendStatistics,
    PostTrendAnalysisResult
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # Create price data with known upward trends
    prices = [100.0]
    
    # Days 1-4: 3-day upward trend (positive returns)
    prices.extend([101.0, 102.0, 103.0])
    
    # Day 5: negative return (ends trend)
    prices.append(102.5)
    
    # Days 6-7: Post-trend period
    prices.extend([102.0, 101.5])
    
    # Days 8-10: flat/negative
    prices.extend([101.0, 100.5, 100.0])
    
    # Days 11-15: 5-day upward trend
    prices.extend([101.0, 102.0, 103.0, 104.0, 105.0])
    
    # Day 16: negative return (ends trend)
    prices.append(104.5)
    
    # Days 17-19: Post-trend period
    prices.extend([105.0, 105.5, 106.0])
    
    # Fill remaining days with random walk
    for _ in range(50 - len(prices)):
        last_price = prices[-1]
        change = np.random.uniform(-0.5, 0.5)
        prices.append(last_price + change)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices
    }, index=dates[:len(prices)])
    
    # Calculate returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    return data


@pytest.fixture
def analyzer():
    """Create an analyzer instance for testing."""
    return PostUpwardTrendReturnAnalyzer(symbol='SPY', analysis_period_months=12)


class TestUpwardTrendIdentification:
    """Test suite for upward trend identification."""
    
    def test_identify_simple_upward_trend(self, analyzer, sample_data):
        """Test identification of a simple upward trend."""
        trends = analyzer.identify_upward_trends(sample_data)
        
        # Should identify at least the known trends
        assert len(trends) >= 2
        
        # Check first trend (3-day)
        first_trend = trends[0]
        assert first_trend.duration >= 3
        assert first_trend.duration <= 10
        assert first_trend.end_price > first_trend.start_price
        assert first_trend.total_return > 0
    
    def test_trend_duration_constraints(self, analyzer, sample_data):
        """Test that trends are correctly constrained to 3-10 days."""
        trends = analyzer.identify_upward_trends(sample_data)
        
        for trend in trends:
            assert 3 <= trend.duration <= 10, f"Trend duration {trend.duration} outside 3-10 range"
    
    def test_post_trend_data_availability(self, analyzer, sample_data):
        """Test that post-trend return data is populated when available."""
        trends = analyzer.identify_upward_trends(sample_data)
        
        # At least some trends should have post-trend data
        trends_with_1d = [t for t in trends if t.return_1d_after is not None]
        assert len(trends_with_1d) > 0
    
    def test_empty_data_handling(self, analyzer):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Daily_Return'])
        
        trends = analyzer.identify_upward_trends(empty_data)
        assert len(trends) == 0


class TestPostTrendReturnCalculation:
    """Test suite for post-trend return calculation."""
    
    def test_calculate_returns_with_sufficient_data(self, analyzer, sample_data):
        """Test calculation of post-trend returns when sufficient data exists."""
        # Use trend ending at index 4 (has 3+ days of data after)
        trend_end_idx = 4
        
        returns = analyzer.calculate_post_trend_returns(sample_data, trend_end_idx)
        return_1d, return_2d, return_3d, price_1d, price_2d, price_3d = returns
        
        # All returns should be calculated
        assert return_1d is not None
        assert return_2d is not None
        assert return_3d is not None
        
        # Prices should be populated
        assert price_1d is not None
        assert price_2d is not None
        assert price_3d is not None
        
        # Returns should be reasonable (between -50% and +50%)
        assert -0.5 <= return_1d <= 0.5
        assert -0.5 <= return_2d <= 0.5
        assert -0.5 <= return_3d <= 0.5
    
    def test_calculate_returns_at_end_of_data(self, analyzer, sample_data):
        """Test calculation when trend ends near end of dataset."""
        # Use last index
        trend_end_idx = len(sample_data) - 1
        
        returns = analyzer.calculate_post_trend_returns(sample_data, trend_end_idx)
        return_1d, return_2d, return_3d, price_1d, price_2d, price_3d = returns
        
        # All returns should be None (no data available)
        assert return_1d is None
        assert return_2d is None
        assert return_3d is None
    
    def test_calculate_returns_with_partial_data(self, analyzer, sample_data):
        """Test calculation when only some post-trend data is available."""
        # Use index with only 1 day of data after
        trend_end_idx = len(sample_data) - 2
        
        returns = analyzer.calculate_post_trend_returns(sample_data, trend_end_idx)
        return_1d, return_2d, return_3d, price_1d, price_2d, price_3d = returns
        
        # 1-day return should be available
        assert return_1d is not None
        
        # 2-day and 3-day returns should be None
        assert return_2d is None
        assert return_3d is None
    
    def test_return_calculation_accuracy(self, analyzer):
        """Test accuracy of return calculations."""
        # Create simple test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0]
        data = pd.DataFrame({
            'Close': prices
        }, index=dates)
        
        # Calculate returns for trend ending at index 3 (price = 103)
        returns = analyzer.calculate_post_trend_returns(data, 3)
        return_1d, return_2d, return_3d, price_1d, price_2d, price_3d = returns
        
        # Expected returns
        # 1-day: (102 - 103) / 103 = -0.0097...
        # 2-day: (101 - 103) / 103 = -0.0194...
        # 3-day: (100 - 103) / 103 = -0.0291...
        
        assert abs(return_1d - (-0.00970873786)) < 0.0001
        assert abs(return_2d - (-0.01941747573)) < 0.0001
        assert abs(return_3d - (-0.02912621359)) < 0.0001


class TestStatisticalAnalysis:
    """Test suite for statistical analysis."""
    
    def test_calculate_statistics_with_trends(self, analyzer):
        """Test statistical calculation with valid trends."""
        # Create mock trends
        trends = [
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-01'),
                end_date=pd.Timestamp('2024-01-03'),
                duration=3,
                start_price=100.0,
                end_price=103.0,
                total_return=0.03,
                return_1d_after=0.01,
                return_2d_after=0.02,
                return_3d_after=0.03,
                price_1d_after=104.0,
                price_2d_after=105.0,
                price_3d_after=106.0
            ),
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-10'),
                end_date=pd.Timestamp('2024-01-14'),
                duration=5,
                start_price=100.0,
                end_price=105.0,
                total_return=0.05,
                return_1d_after=-0.01,
                return_2d_after=-0.02,
                return_3d_after=-0.03,
                price_1d_after=104.0,
                price_2d_after=103.0,
                price_3d_after=102.0
            )
        ]
        
        stats = analyzer.calculate_statistics(trends)
        
        # Check counts
        assert stats.total_trends_analyzed == 2
        assert stats.trends_with_1d_data == 2
        assert stats.trends_with_2d_data == 2
        assert stats.trends_with_3d_data == 2
        
        # Check average returns
        assert stats.avg_return_1d == 0.0  # (0.01 + -0.01) / 2
        assert stats.avg_return_2d == 0.0  # (0.02 + -0.02) / 2
        assert stats.avg_return_3d == 0.0  # (0.03 + -0.03) / 2
        
        # Check negative return percentages
        assert stats.pct_negative_1d == 50.0  # 1 out of 2
        assert stats.pct_negative_2d == 50.0
        assert stats.pct_negative_3d == 50.0
    
    def test_calculate_statistics_empty_trends(self, analyzer):
        """Test statistical calculation with no trends."""
        trends = []
        stats = analyzer.calculate_statistics(trends)
        
        # All values should be zero
        assert stats.total_trends_analyzed == 0
        assert stats.avg_return_1d == 0.0
        assert stats.avg_return_2d == 0.0
        assert stats.avg_return_3d == 0.0
    
    def test_statistics_by_duration(self, analyzer):
        """Test grouping statistics by trend duration."""
        trends = [
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-01'),
                end_date=pd.Timestamp('2024-01-03'),
                duration=3,
                start_price=100.0,
                end_price=103.0,
                total_return=0.03,
                return_1d_after=0.01,
                return_2d_after=0.02,
                return_3d_after=0.03,
                price_1d_after=104.0,
                price_2d_after=105.0,
                price_3d_after=106.0
            ),
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-10'),
                end_date=pd.Timestamp('2024-01-12'),
                duration=3,
                start_price=100.0,
                end_price=103.0,
                total_return=0.03,
                return_1d_after=0.02,
                return_2d_after=0.03,
                return_3d_after=0.04,
                price_1d_after=105.0,
                price_2d_after=106.0,
                price_3d_after=107.0
            )
        ]
        
        stats = analyzer.calculate_statistics(trends)
        
        # Check that 3-day trends have correct averages
        assert stats.returns_by_duration_1d[3] == 0.015  # (0.01 + 0.02) / 2
        assert stats.returns_by_duration_2d[3] == 0.025  # (0.02 + 0.03) / 2
        assert stats.returns_by_duration_3d[3] == 0.035  # (0.03 + 0.04) / 2


class TestReportGeneration:
    """Test suite for report generation."""
    
    def test_generate_report_with_results(self, analyzer):
        """Test report generation with valid results."""
        # Create mock result
        trends = [
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-01'),
                end_date=pd.Timestamp('2024-01-03'),
                duration=3,
                start_price=100.0,
                end_price=103.0,
                total_return=0.03,
                return_1d_after=0.01,
                return_2d_after=0.02,
                return_3d_after=0.03,
                price_1d_after=104.0,
                price_2d_after=105.0,
                price_3d_after=106.0
            )
        ]
        
        stats = PostTrendStatistics(
            total_trends_analyzed=1,
            trends_with_1d_data=1,
            trends_with_2d_data=1,
            trends_with_3d_data=1,
            avg_return_1d=0.01,
            avg_return_2d=0.02,
            avg_return_3d=0.03,
            median_return_1d=0.01,
            median_return_2d=0.02,
            median_return_3d=0.03,
            std_return_1d=0.0,
            std_return_2d=0.0,
            std_return_3d=0.0,
            max_return_1d=0.01,
            max_return_2d=0.02,
            max_return_3d=0.03,
            min_return_1d=0.01,
            min_return_2d=0.02,
            min_return_3d=0.03,
            returns_by_duration_1d={3: 0.01},
            returns_by_duration_2d={3: 0.02},
            returns_by_duration_3d={3: 0.03},
            pct_negative_1d=0.0,
            pct_negative_2d=0.0,
            pct_negative_3d=0.0
        )
        
        result = PostTrendAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-12-31',
            total_trading_days=250,
            upward_trends=trends,
            statistics=stats
        )
        
        report = analyzer.generate_report(result)
        
        # Check that report contains key sections
        assert 'POST-UPWARD TREND RETURN ANALYSIS REPORT' in report
        assert 'TREND IDENTIFICATION SUMMARY' in report
        assert 'AVERAGE RETURNS AFTER TREND ENDS' in report
        assert 'RETURN STATISTICS - 1 DAY AFTER' in report
        assert 'RETURN STATISTICS - 2 DAYS AFTER' in report
        assert 'RETURN STATISTICS - 3 DAYS AFTER' in report
        assert 'AVERAGE 1-DAY RETURN BY TREND DURATION' in report
        
        # Check that values are included
        assert '1.000%' in report  # 1% return
        assert '2.000%' in report  # 2% return
        assert '3.000%' in report  # 3% return
    
    def test_report_format(self, analyzer):
        """Test that report is properly formatted."""
        # Create minimal result
        trends = []
        stats = PostTrendStatistics(
            total_trends_analyzed=0,
            trends_with_1d_data=0,
            trends_with_2d_data=0,
            trends_with_3d_data=0,
            avg_return_1d=0.0,
            avg_return_2d=0.0,
            avg_return_3d=0.0,
            median_return_1d=0.0,
            median_return_2d=0.0,
            median_return_3d=0.0,
            std_return_1d=0.0,
            std_return_2d=0.0,
            std_return_3d=0.0,
            max_return_1d=0.0,
            max_return_2d=0.0,
            max_return_3d=0.0,
            min_return_1d=0.0,
            min_return_2d=0.0,
            min_return_3d=0.0,
            returns_by_duration_1d={},
            returns_by_duration_2d={},
            returns_by_duration_3d={},
            pct_negative_1d=0.0,
            pct_negative_2d=0.0,
            pct_negative_3d=0.0
        )
        
        result = PostTrendAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-12-31',
            total_trading_days=250,
            upward_trends=trends,
            statistics=stats
        )
        
        report = analyzer.generate_report(result)
        
        # Check formatting
        assert report.startswith('=' * 80)
        assert report.endswith('=' * 80)
        assert '\n' in report  # Multi-line report


class TestValueObjects:
    """Test suite for Value Objects."""
    
    def test_upward_trend_immutability(self):
        """Test that UpwardTrend is immutable."""
        trend = UpwardTrend(
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-03'),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            total_return=0.03,
            return_1d_after=0.01,
            return_2d_after=0.02,
            return_3d_after=0.03,
            price_1d_after=104.0,
            price_2d_after=105.0,
            price_3d_after=106.0
        )
        
        # Attempt to modify should raise error
        with pytest.raises(AttributeError):
            trend.duration = 5
    
    def test_post_trend_statistics_immutability(self):
        """Test that PostTrendStatistics is immutable."""
        stats = PostTrendStatistics(
            total_trends_analyzed=1,
            trends_with_1d_data=1,
            trends_with_2d_data=1,
            trends_with_3d_data=1,
            avg_return_1d=0.01,
            avg_return_2d=0.02,
            avg_return_3d=0.03,
            median_return_1d=0.01,
            median_return_2d=0.02,
            median_return_3d=0.03,
            std_return_1d=0.0,
            std_return_2d=0.0,
            std_return_3d=0.0,
            max_return_1d=0.01,
            max_return_2d=0.02,
            max_return_3d=0.03,
            min_return_1d=0.01,
            min_return_2d=0.02,
            min_return_3d=0.03,
            returns_by_duration_1d={},
            returns_by_duration_2d={},
            returns_by_duration_3d={},
            pct_negative_1d=0.0,
            pct_negative_2d=0.0,
            pct_negative_3d=0.0
        )
        
        # Attempt to modify should raise error
        with pytest.raises(AttributeError):
            stats.total_trends_analyzed = 2


class TestEdgeCases:
    """Test suite for edge cases."""
    
    def test_single_day_data(self, analyzer):
        """Test handling of single day of data."""
        dates = pd.date_range(start='2024-01-01', periods=1, freq='D')
        data = pd.DataFrame({
            'Open': [100.0],
            'High': [101.0],
            'Low': [99.0],
            'Close': [100.5],
            'Daily_Return': [0.0]
        }, index=dates)
        
        trends = analyzer.identify_upward_trends(data)
        assert len(trends) == 0
    
    def test_all_negative_returns(self, analyzer):
        """Test handling of data with only negative returns."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        prices = [100.0 - i for i in range(20)]  # Decreasing prices
        data = pd.DataFrame({
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Close': prices
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        trends = analyzer.identify_upward_trends(data)
        assert len(trends) == 0
    
    def test_very_long_upward_trend(self, analyzer):
        """Test handling of upward trend longer than 10 days."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        prices = [100.0 + i for i in range(20)]  # 19 days of increases
        data = pd.DataFrame({
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Close': prices
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        trends = analyzer.identify_upward_trends(data)
        
        # Should identify a 10-day trend (capped at max)
        if trends:
            assert all(t.duration <= 10 for t in trends)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

