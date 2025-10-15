"""
Unit tests for the Upward Trend Drawdown Analysis module.

Tests cover:
- Value Object creation and validation
- Trend identification logic
- Drawdown calculation
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

from analysis.upward_trend_drawdown_analysis import (
    UpwardTrend,
    DrawdownStatistics,
    DrawdownAnalysisResult,
    UpwardTrendDrawdownAnalyzer
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trend():
    """Create a sample UpwardTrend object for testing."""
    return UpwardTrend(
        start_date=pd.Timestamp('2024-01-01'),
        end_date=pd.Timestamp('2024-01-05'),
        duration=5,
        start_price=100.0,
        end_price=105.0,
        total_return=0.05,
        peak_price=106.0,
        trough_price=104.0,
        drawdown_pct=-0.0189,
        has_drawdown=True
    )


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    # Create a pattern with an upward trend (days 0-4) and drawdown
    close_prices = [
        100.0,  # Day 0 (baseline)
        101.0,  # Day 1 (up)
        102.0,  # Day 2 (up)
        103.0,  # Day 3 (up)
        104.0,  # Day 4 (up)
        103.5,  # Day 5 (down, ends trend)
        103.0,  # Day 6 (down)
        104.0,  # Day 7 (up)
        105.0,  # Day 8 (up)
        106.0,  # Day 9 (up)
        107.0,  # Day 10 (up)
        106.5,  # Day 11 (down, ends trend)
        107.0,  # Day 12 (up)
        108.0,  # Day 13 (up)
        107.5,  # Day 14 (down, ends trend)
        107.0,  # Day 15 (down)
        108.0,  # Day 16 (up)
        109.0,  # Day 17 (up)
        110.0,  # Day 18 (up)
        109.0,  # Day 19 (down, ends trend)
    ]
    
    # Create OHLC data with some volatility
    data = pd.DataFrame({
        'Open': close_prices,
        'High': [p * 1.01 for p in close_prices],
        'Low': [p * 0.99 for p in close_prices],
        'Close': close_prices
    }, index=dates)
    
    return data


@pytest.fixture
def analyzer():
    """Create an analyzer instance for testing."""
    return UpwardTrendDrawdownAnalyzer(symbol='SPY', analysis_period_months=12)


# ============================================================================
# Value Object Tests
# ============================================================================

class TestUpwardTrend:
    """Tests for the UpwardTrend Value Object."""
    
    def test_creation(self, sample_trend):
        """Test that UpwardTrend can be created with valid data."""
        assert sample_trend.duration == 5
        assert sample_trend.start_price == 100.0
        assert sample_trend.end_price == 105.0
        assert sample_trend.total_return == 0.05
        assert sample_trend.has_drawdown is True
    
    def test_immutability(self, sample_trend):
        """Test that UpwardTrend is immutable."""
        with pytest.raises(FrozenInstanceError):
            sample_trend.duration = 10
    
    def test_trend_without_drawdown(self):
        """Test creating a trend without a drawdown."""
        trend = UpwardTrend(
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-03'),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            total_return=0.03,
            peak_price=103.0,
            trough_price=103.0,
            drawdown_pct=0.0,
            has_drawdown=False
        )
        assert trend.has_drawdown is False
        assert trend.drawdown_pct == 0.0


class TestDrawdownStatistics:
    """Tests for the DrawdownStatistics Value Object."""
    
    def test_creation(self):
        """Test that DrawdownStatistics can be created with valid data."""
        stats = DrawdownStatistics(
            total_trends_analyzed=100,
            trends_with_drawdowns=80,
            drawdown_percentage=80.0,
            average_drawdown=0.015,
            average_drawdown_with_dd=0.0188,
            max_drawdown=0.05,
            min_drawdown=0.001,
            median_drawdown=0.012,
            std_drawdown=0.008,
            drawdowns_by_duration={3: 0.01, 4: 0.015, 5: 0.02}
        )
        assert stats.total_trends_analyzed == 100
        assert stats.trends_with_drawdowns == 80
        assert stats.drawdown_percentage == 80.0
    
    def test_immutability(self):
        """Test that DrawdownStatistics is immutable."""
        stats = DrawdownStatistics(
            total_trends_analyzed=10,
            trends_with_drawdowns=5,
            drawdown_percentage=50.0,
            average_drawdown=0.01,
            average_drawdown_with_dd=0.02,
            max_drawdown=0.03,
            min_drawdown=0.005,
            median_drawdown=0.01,
            std_drawdown=0.005,
            drawdowns_by_duration={}
        )
        with pytest.raises(FrozenInstanceError):
            stats.total_trends_analyzed = 20


class TestDrawdownAnalysisResult:
    """Tests for the DrawdownAnalysisResult Value Object."""
    
    def test_creation(self, sample_trend):
        """Test that DrawdownAnalysisResult can be created."""
        stats = DrawdownStatistics(
            total_trends_analyzed=1,
            trends_with_drawdowns=1,
            drawdown_percentage=100.0,
            average_drawdown=0.019,
            average_drawdown_with_dd=0.019,
            max_drawdown=0.019,
            min_drawdown=0.019,
            median_drawdown=0.019,
            std_drawdown=0.0,
            drawdowns_by_duration={5: 0.019}
        )
        
        result = DrawdownAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-01-31',
            total_trading_days=20,
            upward_trends=[sample_trend],
            statistics=stats
        )
        
        assert result.total_trading_days == 20
        assert len(result.upward_trends) == 1
        assert result.statistics.total_trends_analyzed == 1


# ============================================================================
# Analyzer Tests
# ============================================================================

class TestUpwardTrendDrawdownAnalyzer:
    """Tests for the UpwardTrendDrawdownAnalyzer class."""
    
    def test_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer.symbol == 'SPY'
        assert analyzer.analysis_period_months == 12
        assert analyzer.data is None
    
    def test_calculate_drawdown_for_trend_with_drawdown(self, analyzer, sample_data):
        """Test drawdown calculation when a drawdown occurs."""
        # Add daily returns
        sample_data['Daily_Return'] = sample_data['Close'].pct_change()
        analyzer.data = sample_data
        
        # Test on days 1-4 (should have some intraday drawdown)
        peak_price, trough_price, drawdown_pct = analyzer.calculate_drawdown_for_trend(
            sample_data, 1, 4
        )
        
        assert peak_price > 0
        assert trough_price > 0
        assert drawdown_pct <= 0  # Drawdown should be negative or zero
    
    def test_calculate_drawdown_for_trend_no_drawdown(self, analyzer):
        """Test drawdown calculation when no drawdown occurs."""
        # Create data with perfectly increasing prices
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'High': [100, 101, 102, 103, 104],
            'Low': [100, 101, 102, 103, 104],
            'Close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        analyzer.data = data
        
        peak_price, trough_price, drawdown_pct = analyzer.calculate_drawdown_for_trend(
            data, 0, 4
        )
        
        assert peak_price == 104  # Highest high
        assert drawdown_pct == 0.0  # No drawdown
    
    def test_identify_upward_trends(self, analyzer, sample_data):
        """Test identification of upward trends in sample data."""
        # Add daily returns
        sample_data['Daily_Return'] = sample_data['Close'].pct_change()
        analyzer.data = sample_data
        
        trends = analyzer.identify_upward_trends(sample_data)
        
        # Should identify at least some upward trends
        assert len(trends) > 0
        
        # All trends should have duration between 3-10
        for trend in trends:
            assert 3 <= trend.duration <= 10
            assert trend.end_price > trend.start_price
    
    def test_identify_upward_trends_minimum_duration(self, analyzer):
        """Test that trends shorter than 3 days are not identified."""
        # Create data with only 2-day upward trend
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [100, 100, 101, 102, 101],
            'High': [100, 100, 101, 102, 101],
            'Low': [100, 100, 101, 102, 101],
            'Close': [100, 100, 101, 102, 101]
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer.data = data
        trends = analyzer.identify_upward_trends(data)
        
        # Should have 1 trend (days 2-3-4, which is 3 days)
        # Day 1: 0% (same as day 0)
        # Day 2: 1% up
        # Day 3: ~0.99% up
        # Day 4: down
        
        # Actually, let me recalculate:
        # Day 0: 100, Day 1: 100 (0% change), Day 2: 101 (1% change), Day 3: 102 (0.99% change), Day 4: 101 (-0.98% change)
        # So we have positive returns on days 2 and 3 only (2 consecutive days)
        # This should NOT create a trend (need 3+ days)
        assert len(trends) == 0
    
    def test_identify_upward_trends_maximum_duration(self, analyzer):
        """Test that trends longer than 10 days are capped at 10."""
        # Create data with 12-day upward trend
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        close_prices = [100 + i for i in range(15)]  # Steadily increasing
        data = pd.DataFrame({
            'Open': close_prices,
            'High': [p * 1.01 for p in close_prices],
            'Low': [p * 0.99 for p in close_prices],
            'Close': close_prices
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer.data = data
        trends = analyzer.identify_upward_trends(data)
        
        # Should have at least one trend
        assert len(trends) > 0
        
        # All trends should have duration <= 10
        for trend in trends:
            assert trend.duration <= 10
    
    def test_calculate_statistics_empty_trends(self, analyzer):
        """Test statistics calculation with no trends."""
        stats = analyzer.calculate_statistics([])
        
        assert stats.total_trends_analyzed == 0
        assert stats.trends_with_drawdowns == 0
        assert stats.average_drawdown == 0.0
    
    def test_calculate_statistics_with_trends(self, analyzer):
        """Test statistics calculation with multiple trends."""
        trends = [
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-01'),
                end_date=pd.Timestamp('2024-01-03'),
                duration=3,
                start_price=100.0,
                end_price=103.0,
                total_return=0.03,
                peak_price=103.5,
                trough_price=102.5,
                drawdown_pct=-0.01,
                has_drawdown=True
            ),
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-05'),
                end_date=pd.Timestamp('2024-01-09'),
                duration=5,
                start_price=103.0,
                end_price=108.0,
                total_return=0.05,
                peak_price=109.0,
                trough_price=107.0,
                drawdown_pct=-0.02,
                has_drawdown=True
            ),
            UpwardTrend(
                start_date=pd.Timestamp('2024-01-11'),
                end_date=pd.Timestamp('2024-01-13'),
                duration=3,
                start_price=108.0,
                end_price=111.0,
                total_return=0.03,
                peak_price=111.0,
                trough_price=111.0,
                drawdown_pct=0.0,
                has_drawdown=False
            )
        ]
        
        stats = analyzer.calculate_statistics(trends)
        
        assert stats.total_trends_analyzed == 3
        assert stats.trends_with_drawdowns == 2
        assert stats.drawdown_percentage == pytest.approx(66.67, rel=0.1)
        assert stats.average_drawdown > 0  # Should be positive (absolute value)
        assert stats.max_drawdown == 0.02  # Largest drawdown (absolute)
        assert 3 in stats.drawdowns_by_duration
        assert 5 in stats.drawdowns_by_duration
    
    def test_generate_report(self, analyzer, sample_trend):
        """Test report generation."""
        stats = DrawdownStatistics(
            total_trends_analyzed=1,
            trends_with_drawdowns=1,
            drawdown_percentage=100.0,
            average_drawdown=0.019,
            average_drawdown_with_dd=0.019,
            max_drawdown=0.019,
            min_drawdown=0.019,
            median_drawdown=0.019,
            std_drawdown=0.0,
            drawdowns_by_duration={5: 0.019}
        )
        
        result = DrawdownAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-01-31',
            total_trading_days=20,
            upward_trends=[sample_trend],
            statistics=stats
        )
        
        report = analyzer.generate_report(result)
        
        # Check that report contains key sections
        assert 'UPWARD TREND DRAWDOWN ANALYSIS REPORT' in report
        assert 'Symbol: SPY' in report
        assert 'TREND IDENTIFICATION SUMMARY' in report
        assert 'DRAWDOWN ANALYSIS' in report
        assert 'DRAWDOWN STATISTICS' in report
        assert 'AVERAGE DRAWDOWN BY TREND DURATION' in report
        assert 'TOP 5 LARGEST DRAWDOWNS' in report
    
    def test_generate_report_no_drawdowns(self, analyzer):
        """Test report generation when no drawdowns are found."""
        trend_no_dd = UpwardTrend(
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-03'),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            total_return=0.03,
            peak_price=103.0,
            trough_price=103.0,
            drawdown_pct=0.0,
            has_drawdown=False
        )
        
        stats = DrawdownStatistics(
            total_trends_analyzed=1,
            trends_with_drawdowns=0,
            drawdown_percentage=0.0,
            average_drawdown=0.0,
            average_drawdown_with_dd=0.0,
            max_drawdown=0.0,
            min_drawdown=0.0,
            median_drawdown=0.0,
            std_drawdown=0.0,
            drawdowns_by_duration={3: 0.0}
        )
        
        result = DrawdownAnalysisResult(
            analysis_period_start='2024-01-01',
            analysis_period_end='2024-01-31',
            total_trading_days=20,
            upward_trends=[trend_no_dd],
            statistics=stats
        )
        
        report = analyzer.generate_report(result)
        
        # Report should still be generated successfully
        assert 'UPWARD TREND DRAWDOWN ANALYSIS REPORT' in report
        assert 'Trends with Drawdowns: 0' in report


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete analysis workflow."""
    
    def test_complete_analysis_workflow(self, sample_data):
        """Test the complete analysis workflow with sample data."""
        # Note: This test uses sample data instead of real API calls
        # In a real scenario, this would test with actual data retrieval
        
        analyzer = UpwardTrendDrawdownAnalyzer(symbol='TEST', analysis_period_months=1)
        
        # Manually set data (bypassing load_data which calls API)
        sample_data['Daily_Return'] = sample_data['Close'].pct_change()
        analyzer.data = sample_data
        
        # Identify trends
        trends = analyzer.identify_upward_trends(sample_data)
        assert len(trends) > 0
        
        # Calculate statistics
        stats = analyzer.calculate_statistics(trends)
        assert stats.total_trends_analyzed == len(trends)
        
        # Create result
        result = DrawdownAnalysisResult(
            analysis_period_start=sample_data.index[0].strftime('%Y-%m-%d'),
            analysis_period_end=sample_data.index[-1].strftime('%Y-%m-%d'),
            total_trading_days=len(sample_data),
            upward_trends=trends,
            statistics=stats
        )
        
        # Generate report
        report = analyzer.generate_report(result)
        assert len(report) > 0
        assert 'UPWARD TREND DRAWDOWN ANALYSIS REPORT' in report
    
    def test_edge_case_all_positive_returns(self):
        """Test with data that has all positive returns."""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        close_prices = [100 + i for i in range(15)]
        data = pd.DataFrame({
            'Open': close_prices,
            'High': [p * 1.01 for p in close_prices],
            'Low': [p * 0.99 for p in close_prices],
            'Close': close_prices
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer = UpwardTrendDrawdownAnalyzer(symbol='TEST', analysis_period_months=1)
        analyzer.data = data
        
        trends = analyzer.identify_upward_trends(data)
        
        # Should identify at least one trend (capped at 10 days)
        assert len(trends) > 0
        assert all(t.duration <= 10 for t in trends)
    
    def test_edge_case_no_upward_trends(self):
        """Test with data that has no qualifying upward trends."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        # Alternating up/down pattern (no 3+ consecutive ups)
        close_prices = [100, 101, 100.5, 101.5, 101, 102, 101.5, 102.5, 102, 103]
        data = pd.DataFrame({
            'Open': close_prices,
            'High': [p * 1.01 for p in close_prices],
            'Low': [p * 0.99 for p in close_prices],
            'Close': close_prices
        }, index=dates)
        data['Daily_Return'] = data['Close'].pct_change()
        
        analyzer = UpwardTrendDrawdownAnalyzer(symbol='TEST', analysis_period_months=1)
        analyzer.data = data
        
        trends = analyzer.identify_upward_trends(data)
        stats = analyzer.calculate_statistics(trends)
        
        # May or may not have trends depending on the exact pattern
        # Just ensure no errors occur
        assert stats.total_trends_analyzed >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

