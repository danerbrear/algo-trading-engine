"""
Unit tests for the Daily Drawdown Likelihood Analysis module.

Tests cover:
- Value Object creation and validation
- Daily drawdown detection logic
- Likelihood calculation by day position
- Statistical analysis
- Report generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from analysis.daily_drawdown_likelihood_analysis import (
    DailyDrawdown,
    TrendWithDailyDrawdowns,
    DailyLikelihoodStatistics,
    DailyDrawdownAnalysisResult,
    DailyDrawdownLikelihoodAnalyzer
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_daily_drawdown():
    """Create a sample DailyDrawdown object for testing."""
    return DailyDrawdown(
        date=pd.Timestamp('2024-01-02'),
        day_position=1,
        had_drawdown=True,
        open_price=100.5,
        high_price=101.0,
        low_price=99.5,
        close_price=100.8,
        prev_close=100.0,
        drawdown_magnitude=0.005,
        drawdown_type='intraday'
    )


@pytest.fixture
def sample_trend_with_daily_data():
    """Create a sample TrendWithDailyDrawdowns object."""
    daily_dds = [
        DailyDrawdown(
            date=pd.Timestamp('2024-01-02'),
            day_position=1,
            had_drawdown=True,
            open_price=100.5,
            high_price=101.0,
            low_price=99.5,
            close_price=100.8,
            prev_close=100.0,
            drawdown_magnitude=0.005,
            drawdown_type='intraday'
        ),
        DailyDrawdown(
            date=pd.Timestamp('2024-01-03'),
            day_position=2,
            had_drawdown=False,
            open_price=101.0,
            high_price=102.0,
            low_price=101.0,
            close_price=102.0,
            prev_close=100.8,
            drawdown_magnitude=0.0,
            drawdown_type='none'
        ),
        DailyDrawdown(
            date=pd.Timestamp('2024-01-04'),
            day_position=3,
            had_drawdown=True,
            open_price=102.5,
            high_price=103.0,
            low_price=101.5,
            close_price=102.8,
            prev_close=102.0,
            drawdown_magnitude=0.005,
            drawdown_type='intraday'
        )
    ]
    
    return TrendWithDailyDrawdowns(
        start_date=pd.Timestamp('2024-01-02'),
        end_date=pd.Timestamp('2024-01-04'),
        duration=3,
        start_price=100.0,
        end_price=102.8,
        total_return=0.028,
        daily_drawdowns=tuple(daily_dds),
        total_days_with_drawdowns=2,
        drawdown_frequency=2/3
    )


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    # Create data with some drawdowns
    data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 104, 105, 106, 107],
        'High': [101, 102, 103, 104, 105, 106, 105, 106, 107, 108],
        'Low': [99.5, 100.5, 101, 102.5, 103.5, 104, 103, 104.5, 105.5, 106.5],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 104.5, 105.5, 106.5, 107.5]
    }, index=dates)
    
    data['Daily_Return'] = data['Close'].pct_change()
    
    return data


@pytest.fixture
def analyzer():
    """Create an analyzer instance for testing."""
    return DailyDrawdownLikelihoodAnalyzer(symbol='SPY', analysis_period_months=12)


# ============================================================================
# Value Object Tests
# ============================================================================

class TestDailyDrawdown:
    """Tests for the DailyDrawdown Value Object."""
    
    def test_creation(self, sample_daily_drawdown):
        """Test that DailyDrawdown can be created with valid data."""
        assert sample_daily_drawdown.day_position == 1
        assert sample_daily_drawdown.had_drawdown is True
        assert sample_daily_drawdown.drawdown_magnitude == 0.005
        assert sample_daily_drawdown.drawdown_type == 'intraday'
    
    def test_immutability(self, sample_daily_drawdown):
        """Test that DailyDrawdown is immutable."""
        with pytest.raises(FrozenInstanceError):
            sample_daily_drawdown.day_position = 2
    
    def test_no_drawdown(self):
        """Test creating a DailyDrawdown with no drawdown."""
        dd = DailyDrawdown(
            date=pd.Timestamp('2024-01-02'),
            day_position=1,
            had_drawdown=False,
            open_price=100.0,
            high_price=101.0,
            low_price=100.0,
            close_price=101.0,
            prev_close=99.5,
            drawdown_magnitude=0.0,
            drawdown_type='none'
        )
        assert dd.had_drawdown is False
        assert dd.drawdown_magnitude == 0.0


class TestTrendWithDailyDrawdowns:
    """Tests for the TrendWithDailyDrawdowns Value Object."""
    
    def test_creation(self, sample_trend_with_daily_data):
        """Test that TrendWithDailyDrawdowns can be created."""
        assert sample_trend_with_daily_data.duration == 3
        assert len(sample_trend_with_daily_data.daily_drawdowns) == 3
        assert sample_trend_with_daily_data.total_days_with_drawdowns == 2
        assert sample_trend_with_daily_data.drawdown_frequency == pytest.approx(2/3)
    
    def test_immutability(self, sample_trend_with_daily_data):
        """Test that TrendWithDailyDrawdowns is immutable."""
        with pytest.raises(FrozenInstanceError):
            sample_trend_with_daily_data.duration = 5


class TestDailyLikelihoodStatistics:
    """Tests for the DailyLikelihoodStatistics Value Object."""
    
    def test_creation(self):
        """Test that DailyLikelihoodStatistics can be created."""
        stats = DailyLikelihoodStatistics(
            day_position=3,
            total_trends_reaching_day=50,
            trends_with_drawdown_on_day=40,
            likelihood_percentage=80.0,
            average_drawdown_magnitude=0.015,
            max_drawdown_magnitude=0.05,
            min_drawdown_magnitude=0.001
        )
        assert stats.day_position == 3
        assert stats.likelihood_percentage == 80.0
    
    def test_immutability(self):
        """Test that DailyLikelihoodStatistics is immutable."""
        stats = DailyLikelihoodStatistics(
            day_position=1,
            total_trends_reaching_day=10,
            trends_with_drawdown_on_day=5,
            likelihood_percentage=50.0,
            average_drawdown_magnitude=0.01,
            max_drawdown_magnitude=0.02,
            min_drawdown_magnitude=0.005
        )
        with pytest.raises(FrozenInstanceError):
            stats.likelihood_percentage = 60.0


class TestDailyDrawdownAnalysisResult:
    """Tests for the DailyDrawdownAnalysisResult Value Object."""
    
    def test_creation(self, sample_trend_with_daily_data):
        """Test that DailyDrawdownAnalysisResult can be created."""
        stats = {
            3: DailyLikelihoodStatistics(
                day_position=3,
                total_trends_reaching_day=1,
                trends_with_drawdown_on_day=1,
                likelihood_percentage=100.0,
                average_drawdown_magnitude=0.005,
                max_drawdown_magnitude=0.005,
                min_drawdown_magnitude=0.005
            )
        }
        
        result = DailyDrawdownAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-01-31',
            total_trading_days=20,
            total_trends_analyzed=1,
            trends_with_daily_data=(sample_trend_with_daily_data,),
            likelihood_by_day_position=stats,
            overall_daily_likelihood=80.0
        )
        
        assert result.total_trends_analyzed == 1
        assert result.overall_daily_likelihood == 80.0


# ============================================================================
# Analyzer Tests
# ============================================================================

class TestDailyDrawdownLikelihoodAnalyzer:
    """Tests for the DailyDrawdownLikelihoodAnalyzer class."""
    
    def test_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer.symbol == 'SPY'
        assert analyzer.analysis_period_months == 12
        assert analyzer.data is None
    
    def test_check_daily_drawdown_intraday(self, analyzer, sample_ohlc_data):
        """Test detection of intraday drawdown."""
        analyzer.data = sample_ohlc_data
        
        # Day 1 (index 1): Low (100.5) < Open (101)
        had_dd, magnitude, dd_type = analyzer.check_daily_drawdown(sample_ohlc_data, 1)
        assert had_dd is True
        assert magnitude > 0
        assert dd_type in ['intraday', 'gap_down']
    
    def test_check_daily_drawdown_no_drawdown(self, analyzer):
        """Test when no drawdown occurred."""
        # Perfect upward day
        data = pd.DataFrame({
            'Open': [100, 101],
            'High': [100, 102],
            'Low': [100, 101],
            'Close': [100, 102]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        had_dd, magnitude, dd_type = analyzer.check_daily_drawdown(data, 1)
        assert had_dd is False
        assert magnitude == 0.0
        assert dd_type == 'none'
    
    def test_check_daily_drawdown_gap_down(self, analyzer):
        """Test detection of gap down."""
        data = pd.DataFrame({
            'Open': [100, 99],  # Gap down
            'High': [100, 99.5],
            'Low': [100, 98.5],
            'Close': [100, 99.2]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        had_dd, magnitude, dd_type = analyzer.check_daily_drawdown(data, 1)
        assert had_dd is True
        assert dd_type == 'gap_down'
    
    def test_identify_upward_trends(self, analyzer, sample_ohlc_data):
        """Test identification of upward trends."""
        analyzer.data = sample_ohlc_data
        
        trends = analyzer.identify_upward_trends(sample_ohlc_data)
        
        # Should find at least one trend
        assert len(trends) > 0
        
        # Each trend should be a tuple of (start_idx, end_idx)
        for start, end in trends:
            assert end >= start
            assert end - start + 1 >= 3  # Minimum 3 days
            assert end - start + 1 <= 10  # Maximum 10 days
    
    def test_analyze_trend_daily_drawdowns(self, analyzer, sample_ohlc_data):
        """Test analysis of daily drawdowns for a trend."""
        analyzer.data = sample_ohlc_data
        
        # Analyze a 3-day trend (indices 1-3)
        trend = analyzer.analyze_trend_daily_drawdowns(sample_ohlc_data, 1, 3)
        
        assert trend.duration == 3
        assert len(trend.daily_drawdowns) == 3
        assert all(dd.day_position >= 1 and dd.day_position <= 3 
                  for dd in trend.daily_drawdowns)
    
    def test_calculate_likelihood_by_day_position(self, analyzer, sample_trend_with_daily_data):
        """Test calculation of likelihood by day position."""
        trends = [sample_trend_with_daily_data]
        
        likelihood_stats = analyzer.calculate_likelihood_by_day_position(trends)
        
        # Should have statistics for days 1-3
        assert 1 in likelihood_stats
        assert 2 in likelihood_stats
        assert 3 in likelihood_stats
        
        # Check day 1 statistics
        day1_stats = likelihood_stats[1]
        assert day1_stats.total_trends_reaching_day == 1
        assert day1_stats.trends_with_drawdown_on_day == 1
        assert day1_stats.likelihood_percentage == 100.0
        
        # Check day 2 statistics (no drawdown)
        day2_stats = likelihood_stats[2]
        assert day2_stats.trends_with_drawdown_on_day == 0
        assert day2_stats.likelihood_percentage == 0.0
    
    def test_calculate_likelihood_multiple_trends(self, analyzer):
        """Test likelihood calculation with multiple trends."""
        # Create multiple trends with varying drawdown patterns
        dd1 = DailyDrawdown(
            date=pd.Timestamp('2024-01-01'),
            day_position=1,
            had_drawdown=True,
            open_price=100,
            high_price=101,
            low_price=99,
            close_price=100.5,
            prev_close=99.5,
            drawdown_magnitude=0.01,
            drawdown_type='intraday'
        )
        
        dd2_no_drawdown = DailyDrawdown(
            date=pd.Timestamp('2024-02-01'),
            day_position=1,
            had_drawdown=False,
            open_price=100,
            high_price=101,
            low_price=100,
            close_price=101,
            prev_close=99.5,
            drawdown_magnitude=0.0,
            drawdown_type='none'
        )
        
        trend1 = TrendWithDailyDrawdowns(
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-03'),
            duration=3,
            start_price=99.5,
            end_price=102,
            total_return=0.025,
            daily_drawdowns=(dd1, dd1, dd1),
            total_days_with_drawdowns=3,
            drawdown_frequency=1.0
        )
        
        trend2 = TrendWithDailyDrawdowns(
            start_date=pd.Timestamp('2024-02-01'),
            end_date=pd.Timestamp('2024-02-03'),
            duration=3,
            start_price=99.5,
            end_price=102,
            total_return=0.025,
            daily_drawdowns=(dd2_no_drawdown, dd2_no_drawdown, dd2_no_drawdown),
            total_days_with_drawdowns=0,
            drawdown_frequency=0.0
        )
        
        likelihood_stats = analyzer.calculate_likelihood_by_day_position([trend1, trend2])
        
        # Day 1 should have 50% likelihood (1 out of 2 trends)
        assert likelihood_stats[1].likelihood_percentage == 50.0
    
    def test_generate_report(self, analyzer, sample_trend_with_daily_data):
        """Test report generation."""
        stats = {
            1: DailyLikelihoodStatistics(
                day_position=1,
                total_trends_reaching_day=1,
                trends_with_drawdown_on_day=1,
                likelihood_percentage=100.0,
                average_drawdown_magnitude=0.005,
                max_drawdown_magnitude=0.005,
                min_drawdown_magnitude=0.005
            ),
            2: DailyLikelihoodStatistics(
                day_position=2,
                total_trends_reaching_day=1,
                trends_with_drawdown_on_day=0,
                likelihood_percentage=0.0,
                average_drawdown_magnitude=0.0,
                max_drawdown_magnitude=0.0,
                min_drawdown_magnitude=0.0
            ),
            3: DailyLikelihoodStatistics(
                day_position=3,
                total_trends_reaching_day=1,
                trends_with_drawdown_on_day=1,
                likelihood_percentage=100.0,
                average_drawdown_magnitude=0.005,
                max_drawdown_magnitude=0.005,
                min_drawdown_magnitude=0.005
            )
        }
        
        result = DailyDrawdownAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-01-31',
            total_trading_days=20,
            total_trends_analyzed=1,
            trends_with_daily_data=(sample_trend_with_daily_data,),
            likelihood_by_day_position=stats,
            overall_daily_likelihood=66.67
        )
        
        report = analyzer.generate_report(result)
        
        # Check that report contains key sections
        assert 'DAILY DRAWDOWN LIKELIHOOD ANALYSIS REPORT' in report
        assert 'Symbol: SPY' in report
        assert 'OVERALL STATISTICS' in report
        assert 'LIKELIHOOD BY DAY POSITION' in report
        assert 'INSIGHTS' in report
        assert 'TREND DURATION BREAKDOWN' in report


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete analysis workflow."""
    
    def test_complete_workflow_with_sample_data(self, sample_ohlc_data):
        """Test the complete workflow with sample data."""
        analyzer = DailyDrawdownLikelihoodAnalyzer(symbol='TEST', analysis_period_months=1)
        
        # Manually set data (bypassing API call)
        analyzer.data = sample_ohlc_data
        
        # Identify trends
        trends = analyzer.identify_upward_trends(sample_ohlc_data)
        assert len(trends) > 0
        
        # Analyze daily drawdowns
        trends_with_data = []
        for start, end in trends:
            trend = analyzer.analyze_trend_daily_drawdowns(sample_ohlc_data, start, end)
            trends_with_data.append(trend)
        
        # Calculate likelihood
        likelihood_stats = analyzer.calculate_likelihood_by_day_position(trends_with_data)
        assert len(likelihood_stats) > 0
        
        # Each trend should have daily drawdown data
        for trend in trends_with_data:
            assert len(trend.daily_drawdowns) == trend.duration
    
    def test_edge_case_all_days_have_drawdowns(self):
        """Test with data where every day has a drawdown."""
        # Create data where low is always below open
        dates = pd.date_range(start='2024-01-01', periods=6, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105],
            'High': [101, 102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103, 104],  # Always below open
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5]
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer = DailyDrawdownLikelihoodAnalyzer(symbol='TEST', analysis_period_months=1)
        analyzer.data = data
        
        trends = analyzer.identify_upward_trends(data)
        
        if trends:
            trends_with_data = []
            for start, end in trends:
                trend = analyzer.analyze_trend_daily_drawdowns(data, start, end)
                trends_with_data.append(trend)
            
            # All days should have drawdowns
            for trend in trends_with_data:
                assert trend.drawdown_frequency == 1.0
    
    def test_edge_case_no_drawdowns(self):
        """Test with data where no day has a drawdown."""
        # Create data where low equals open (perfect upward movement)
        dates = pd.date_range(start='2024-01-01', periods=6, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105],
            'High': [101, 102, 103, 104, 105, 106],
            'Low': [100, 101, 102, 103, 104, 105],  # Equals open
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5]
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer = DailyDrawdownLikelihoodAnalyzer(symbol='TEST', analysis_period_months=1)
        analyzer.data = data
        
        trends = analyzer.identify_upward_trends(data)
        
        if trends:
            trends_with_data = []
            for start, end in trends:
                trend = analyzer.analyze_trend_daily_drawdowns(data, start, end)
                trends_with_data.append(trend)
            
            likelihood_stats = analyzer.calculate_likelihood_by_day_position(trends_with_data)
            
            # All likelihood percentages should be 0
            for stats in likelihood_stats.values():
                assert stats.likelihood_percentage == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

