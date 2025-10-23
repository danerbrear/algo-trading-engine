"""
Post-Upward Trend Return Analysis

This module analyzes returns that occur in the 1-3 days following upward trends in SPY.
An upward trend is defined as 3-10 consecutive days of positive returns.

The analysis provides statistical insights into the average returns following upward trends
and helps understand whether trends tend to reverse or continue.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import data retrieval classes
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from common.data_retriever import DataRetriever


@dataclass(frozen=True)
class UpwardTrend:
    """Represents an identified upward trend period."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int  # Number of days (3-10)
    start_price: float  # Price at start of trend
    end_price: float  # Price at end of trend
    total_return: float  # Percentage return during the trend
    
    # Post-trend returns (can be None if insufficient data)
    return_1d_after: Optional[float]  # 1-day return after trend ends
    return_2d_after: Optional[float]  # 2-day return after trend ends
    return_3d_after: Optional[float]  # 3-day return after trend ends
    
    # Post-trend prices
    price_1d_after: Optional[float]
    price_2d_after: Optional[float]
    price_3d_after: Optional[float]


@dataclass(frozen=True)
class PostTrendStatistics:
    """Statistical summary of returns following upward trends."""
    total_trends_analyzed: int
    trends_with_1d_data: int
    trends_with_2d_data: int
    trends_with_3d_data: int
    
    # Overall averages
    avg_return_1d: float  # Average 1-day return across all trends
    avg_return_2d: float  # Average 2-day return across all trends
    avg_return_3d: float  # Average 3-day return across all trends
    
    # Statistical measures for each period
    median_return_1d: float
    median_return_2d: float
    median_return_3d: float
    
    std_return_1d: float
    std_return_2d: float
    std_return_3d: float
    
    max_return_1d: float
    max_return_2d: float
    max_return_3d: float
    
    min_return_1d: float
    min_return_2d: float
    min_return_3d: float
    
    # Returns grouped by trend duration
    returns_by_duration_1d: Dict[int, float]  # Average 1-day return by trend duration (3-10)
    returns_by_duration_2d: Dict[int, float]  # Average 2-day return by trend duration (3-10)
    returns_by_duration_3d: Dict[int, float]  # Average 3-day return by trend duration (3-10)
    
    # Percentage of negative returns (reversals/drawdowns)
    pct_negative_1d: float
    pct_negative_2d: float
    pct_negative_3d: float


@dataclass(frozen=True)
class PostTrendAnalysisResult:
    """Complete results of the post-trend return analysis."""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    upward_trends: List[UpwardTrend]
    statistics: PostTrendStatistics


class PostUpwardTrendReturnAnalyzer:
    """
    Analyzes returns that occur in the 1-3 days following upward trends in SPY.
    
    This analyzer identifies periods of continuous positive returns (3-10 days)
    and measures the returns over the subsequent 1, 2, and 3 trading days.
    """
    
    def __init__(self, symbol: str = 'SPY', analysis_period_months: int = 12):
        """
        Initialize the analyzer.
        
        Args:
            symbol: Stock symbol to analyze (default: 'SPY')
            analysis_period_months: Number of months to analyze (default: 12)
        """
        self.symbol = symbol
        self.analysis_period_months = analysis_period_months
        
        # Calculate start date
        today = datetime.now()
        start_date = today - timedelta(days=analysis_period_months * 30)
        self.start_date = start_date.strftime('%Y-%m-%d')
        
        self.data_retriever = DataRetriever(symbol=symbol, lstm_start_date=self.start_date)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical price data for the specified period.
        
        Returns:
            DataFrame with OHLC data and calculated returns
        """
        print(f"📊 Loading {self.symbol} data from {self.start_date}...")
        
        # Fetch data using the existing data retriever
        self.data = self.data_retriever.fetch_data_for_period(
            self.start_date, 
            'post_trend_analysis'
        )
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        print(f"✅ Loaded {len(self.data)} days of {self.symbol} data")
        print(f"   Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def identify_upward_trends(self, data: pd.DataFrame) -> List[UpwardTrend]:
        """
        Identify all upward trends (3-10 consecutive days of positive returns)
        and calculate post-trend returns.
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of UpwardTrend objects with post-trend return data
        """
        print("\n🔍 Identifying upward trends and calculating post-trend returns...")
        trends = []
        i = 1  # Start at 1 because returns start at index 1
        
        while i < len(data) - 5:  # Need buffer for post-trend analysis
            # Check if current day has positive return
            if data['Daily_Return'].iloc[i] > 0:
                trend_start_idx = i
                consecutive_positive_days = 0
                
                # Count consecutive positive returns
                j = i
                while j < len(data) and data['Daily_Return'].iloc[j] > 0:
                    consecutive_positive_days += 1
                    j += 1
                    
                    # Cap at 10 days max
                    if consecutive_positive_days >= 10:
                        break
                
                # If trend is between 3-10 days, analyze it
                if 3 <= consecutive_positive_days <= 10:
                    trend_end_idx = trend_start_idx + consecutive_positive_days - 1
                    
                    # Calculate post-trend returns
                    (return_1d, return_2d, return_3d,
                     price_1d, price_2d, price_3d) = self.calculate_post_trend_returns(
                        data, trend_end_idx
                    )
                    
                    # Get price information
                    start_price = data['Close'].iloc[trend_start_idx - 1]  # Price before trend
                    end_price = data['Close'].iloc[trend_end_idx]
                    total_return = (end_price - start_price) / start_price
                    
                    # Create UpwardTrend object
                    trend = UpwardTrend(
                        start_date=data.index[trend_start_idx],
                        end_date=data.index[trend_end_idx],
                        duration=consecutive_positive_days,
                        start_price=start_price,
                        end_price=end_price,
                        total_return=total_return,
                        return_1d_after=return_1d,
                        return_2d_after=return_2d,
                        return_3d_after=return_3d,
                        price_1d_after=price_1d,
                        price_2d_after=price_2d,
                        price_3d_after=price_3d
                    )
                    trends.append(trend)
                
                # Move index past this trend
                i = j
            else:
                i += 1
        
        print(f"✅ Identified {len(trends)} upward trends")
        
        # Print breakdown by duration
        duration_counts = {}
        for trend in trends:
            duration_counts[trend.duration] = duration_counts.get(trend.duration, 0) + 1
        
        print("   Trends by duration:")
        for duration in range(3, 11):
            count = duration_counts.get(duration, 0)
            print(f"     {duration} days: {count} trends")
        
        return trends
    
    def calculate_post_trend_returns(
        self, 
        data: pd.DataFrame, 
        trend_end_idx: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float], 
               Optional[float], Optional[float], Optional[float]]:
        """
        Calculate returns for 1, 2, and 3 days after the trend ends.
        
        Args:
            data: DataFrame with price data
            trend_end_idx: Index where the upward trend ended
            
        Returns:
            Tuple of (return_1d, return_2d, return_3d, price_1d, price_2d, price_3d)
        """
        end_price = data['Close'].iloc[trend_end_idx]
        
        # Calculate 1-day return
        if trend_end_idx + 1 < len(data):
            price_1d = data['Close'].iloc[trend_end_idx + 1]
            return_1d = (price_1d - end_price) / end_price
        else:
            price_1d = None
            return_1d = None
        
        # Calculate 2-day return (cumulative from end of trend)
        if trend_end_idx + 2 < len(data):
            price_2d = data['Close'].iloc[trend_end_idx + 2]
            return_2d = (price_2d - end_price) / end_price
        else:
            price_2d = None
            return_2d = None
        
        # Calculate 3-day return (cumulative from end of trend)
        if trend_end_idx + 3 < len(data):
            price_3d = data['Close'].iloc[trend_end_idx + 3]
            return_3d = (price_3d - end_price) / end_price
        else:
            price_3d = None
            return_3d = None
        
        return return_1d, return_2d, return_3d, price_1d, price_2d, price_3d
    
    def calculate_statistics(self, trends: List[UpwardTrend]) -> PostTrendStatistics:
        """
        Calculate statistical measures of post-trend returns.
        
        Args:
            trends: List of identified upward trends
            
        Returns:
            PostTrendStatistics object with summary metrics
        """
        print("\n📈 Calculating statistics...")
        
        if not trends:
            # Return empty statistics if no trends found
            return PostTrendStatistics(
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
        
        # Filter trends with available data
        trends_1d = [t for t in trends if t.return_1d_after is not None]
        trends_2d = [t for t in trends if t.return_2d_after is not None]
        trends_3d = [t for t in trends if t.return_3d_after is not None]
        
        # Calculate overall statistics for each period
        def calc_period_stats(trend_list, return_attr):
            if not trend_list:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            returns = [getattr(t, return_attr) for t in trend_list]
            avg = np.mean(returns)
            median = np.median(returns)
            std = np.std(returns)
            max_ret = max(returns)
            min_ret = min(returns)
            pct_negative = sum(1 for r in returns if r < 0) / len(returns) * 100
            
            return avg, median, std, max_ret, min_ret, pct_negative
        
        # Calculate for each period
        (avg_1d, median_1d, std_1d,
         max_1d, min_1d, pct_neg_1d) = calc_period_stats(trends_1d, 'return_1d_after')
        
        (avg_2d, median_2d, std_2d,
         max_2d, min_2d, pct_neg_2d) = calc_period_stats(trends_2d, 'return_2d_after')
        
        (avg_3d, median_3d, std_3d,
         max_3d, min_3d, pct_neg_3d) = calc_period_stats(trends_3d, 'return_3d_after')
        
        # Group by duration for each period
        def calc_by_duration(trend_list, return_attr):
            by_duration = {}
            for duration in range(3, 11):
                duration_trends = [t for t in trend_list if t.duration == duration]
                if duration_trends:
                    returns = [getattr(t, return_attr) for t in duration_trends]
                    by_duration[duration] = np.mean(returns)
                else:
                    by_duration[duration] = 0.0
            return by_duration
        
        returns_by_dur_1d = calc_by_duration(trends_1d, 'return_1d_after')
        returns_by_dur_2d = calc_by_duration(trends_2d, 'return_2d_after')
        returns_by_dur_3d = calc_by_duration(trends_3d, 'return_3d_after')
        
        print(f"✅ Statistics calculated")
        print(f"   Total trends: {len(trends)}")
        print(f"   Trends with 1-day data: {len(trends_1d)}")
        print(f"   Trends with 2-day data: {len(trends_2d)}")
        print(f"   Trends with 3-day data: {len(trends_3d)}")
        print(f"   Average 1-day return: {avg_1d*100:.3f}%")
        print(f"   Average 2-day return: {avg_2d*100:.3f}%")
        print(f"   Average 3-day return: {avg_3d*100:.3f}%")
        
        return PostTrendStatistics(
            total_trends_analyzed=len(trends),
            trends_with_1d_data=len(trends_1d),
            trends_with_2d_data=len(trends_2d),
            trends_with_3d_data=len(trends_3d),
            avg_return_1d=avg_1d,
            avg_return_2d=avg_2d,
            avg_return_3d=avg_3d,
            median_return_1d=median_1d,
            median_return_2d=median_2d,
            median_return_3d=median_3d,
            std_return_1d=std_1d,
            std_return_2d=std_2d,
            std_return_3d=std_3d,
            max_return_1d=max_1d,
            max_return_2d=max_2d,
            max_return_3d=max_3d,
            min_return_1d=min_1d,
            min_return_2d=min_2d,
            min_return_3d=min_3d,
            returns_by_duration_1d=returns_by_dur_1d,
            returns_by_duration_2d=returns_by_dur_2d,
            returns_by_duration_3d=returns_by_dur_3d,
            pct_negative_1d=pct_neg_1d,
            pct_negative_2d=pct_neg_2d,
            pct_negative_3d=pct_neg_3d
        )
    
    def run_analysis(self) -> PostTrendAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            PostTrendAnalysisResult with all findings
        """
        print("🚀 Starting Post-Upward Trend Return Analysis...")
        
        # Load data
        data = self.load_data()
        
        # Identify upward trends
        trends = self.identify_upward_trends(data)
        
        # Calculate statistics
        statistics = self.calculate_statistics(trends)
        
        # Create result object
        result = PostTrendAnalysisResult(
            analysis_period_start=data.index[0].strftime('%Y-%m-%d'),
            analysis_period_end=data.index[-1].strftime('%Y-%m-%d'),
            total_trading_days=len(data),
            upward_trends=trends,
            statistics=statistics
        )
        
        print("\n✅ Analysis complete!")
        
        return result
    
    def generate_report(self, result: PostTrendAnalysisResult) -> str:
        """
        Generate a text report of the analysis results.
        
        Args:
            result: Complete analysis results
            
        Returns:
            Formatted report string
        """
        stats = result.statistics
        trends = result.upward_trends
        
        report = []
        report.append("=" * 80)
        report.append("POST-UPWARD TREND RETURN ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Symbol: {self.symbol}")
        report.append(f"Analysis Period: {result.analysis_period_start} to {result.analysis_period_end}")
        report.append(f"Total Trading Days: {result.total_trading_days}")
        report.append("")
        
        # Trend identification summary
        report.append("TREND IDENTIFICATION SUMMARY:")
        report.append(f"  Total Upward Trends (3-10 days): {stats.total_trends_analyzed}")
        report.append("  Trends by Duration:")
        
        duration_counts = {}
        for trend in trends:
            duration_counts[trend.duration] = duration_counts.get(trend.duration, 0) + 1
        
        for duration in range(3, 11):
            count = duration_counts.get(duration, 0)
            report.append(f"    {duration} days: {count} trends")
        report.append("")
        
        # Post-trend return analysis
        report.append("POST-TREND RETURN ANALYSIS:")
        pct_1d = (stats.trends_with_1d_data / stats.total_trends_analyzed * 100) if stats.total_trends_analyzed > 0 else 0
        pct_2d = (stats.trends_with_2d_data / stats.total_trends_analyzed * 100) if stats.total_trends_analyzed > 0 else 0
        pct_3d = (stats.trends_with_3d_data / stats.total_trends_analyzed * 100) if stats.total_trends_analyzed > 0 else 0
        report.append(f"  Trends with 1-day data: {stats.trends_with_1d_data} ({pct_1d:.1f}%)")
        report.append(f"  Trends with 2-day data: {stats.trends_with_2d_data} ({pct_2d:.1f}%)")
        report.append(f"  Trends with 3-day data: {stats.trends_with_3d_data} ({pct_3d:.1f}%)")
        report.append("")
        
        # Average returns after trend ends
        report.append("AVERAGE RETURNS AFTER TREND ENDS:")
        report.append(f"  1-Day Average Return: {stats.avg_return_1d*100:.3f}%")
        report.append(f"  2-Day Average Return: {stats.avg_return_2d*100:.3f}%")
        report.append(f"  3-Day Average Return: {stats.avg_return_3d*100:.3f}%")
        report.append("")
        
        # Return statistics - 1 day after
        report.append("RETURN STATISTICS - 1 DAY AFTER:")
        report.append(f"  Mean: {stats.avg_return_1d*100:.3f}%")
        report.append(f"  Median: {stats.median_return_1d*100:.3f}%")
        report.append(f"  Std Dev: {stats.std_return_1d*100:.3f}%")
        report.append(f"  Max: {stats.max_return_1d*100:.3f}%")
        report.append(f"  Min: {stats.min_return_1d*100:.3f}%")
        report.append(f"  Negative Returns (Reversals): {stats.pct_negative_1d:.1f}%")
        report.append("")
        
        # Return statistics - 2 days after
        report.append("RETURN STATISTICS - 2 DAYS AFTER:")
        report.append(f"  Mean: {stats.avg_return_2d*100:.3f}%")
        report.append(f"  Median: {stats.median_return_2d*100:.3f}%")
        report.append(f"  Std Dev: {stats.std_return_2d*100:.3f}%")
        report.append(f"  Max: {stats.max_return_2d*100:.3f}%")
        report.append(f"  Min: {stats.min_return_2d*100:.3f}%")
        report.append(f"  Negative Returns (Reversals): {stats.pct_negative_2d:.1f}%")
        report.append("")
        
        # Return statistics - 3 days after
        report.append("RETURN STATISTICS - 3 DAYS AFTER:")
        report.append(f"  Mean: {stats.avg_return_3d*100:.3f}%")
        report.append(f"  Median: {stats.median_return_3d*100:.3f}%")
        report.append(f"  Std Dev: {stats.std_return_3d*100:.3f}%")
        report.append(f"  Max: {stats.max_return_3d*100:.3f}%")
        report.append(f"  Min: {stats.min_return_3d*100:.3f}%")
        report.append(f"  Negative Returns (Reversals): {stats.pct_negative_3d:.1f}%")
        report.append("")
        
        # Average 1-day return by trend duration
        report.append("AVERAGE 1-DAY RETURN BY TREND DURATION:")
        for duration in range(3, 11):
            avg_ret = stats.returns_by_duration_1d.get(duration, 0.0)
            report.append(f"  {duration}-day trends: {avg_ret*100:.3f}%")
        report.append("")
        
        # Average 2-day return by trend duration
        report.append("AVERAGE 2-DAY RETURN BY TREND DURATION:")
        for duration in range(3, 11):
            avg_ret = stats.returns_by_duration_2d.get(duration, 0.0)
            report.append(f"  {duration}-day trends: {avg_ret*100:.3f}%")
        report.append("")
        
        # Average 3-day return by trend duration
        report.append("AVERAGE 3-DAY RETURN BY TREND DURATION:")
        for duration in range(3, 11):
            avg_ret = stats.returns_by_duration_3d.get(duration, 0.0)
            report.append(f"  {duration}-day trends: {avg_ret*100:.3f}%")
        report.append("")
        
        # Top 5 best post-trend returns (3-day)
        trends_with_3d = [t for t in trends if t.return_3d_after is not None]
        if trends_with_3d:
            sorted_trends = sorted(trends_with_3d, key=lambda t: t.return_3d_after, reverse=True)
            top_5 = sorted_trends[:5]
            
            report.append("TOP 5 BEST POST-TREND RETURNS (3-day):")
            for i, trend in enumerate(top_5, 1):
                report.append(
                    f"  {i}. {trend.end_date.strftime('%Y-%m-%d')}: "
                    f"{trend.return_3d_after*100:+.2f}% ({trend.duration}-day trend)"
                )
            report.append("")
        
        # Top 5 worst post-trend returns (3-day)
        if trends_with_3d:
            sorted_trends = sorted(trends_with_3d, key=lambda t: t.return_3d_after)
            bottom_5 = sorted_trends[:5]
            
            report.append("TOP 5 WORST POST-TREND RETURNS (3-day):")
            for i, trend in enumerate(bottom_5, 1):
                report.append(
                    f"  {i}. {trend.end_date.strftime('%Y-%m-%d')}: "
                    f"{trend.return_3d_after*100:+.2f}% ({trend.duration}-day trend)"
                )
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_results(self, result: PostTrendAnalysisResult, save_path: str = None):
        """
        Create visualizations of the analysis results.
        
        Args:
            result: Complete analysis results
            save_path: Optional path to save the plot
        """
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # Plot 1: SPY Price with Upward Trends and Post-Trend Markers (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_with_trends(ax1, result)
        
        # Plot 2: Return Distribution Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_return_distributions(ax2, result)
        
        # Plot 3: Average Return by Trend Duration (Grouped Bar Chart)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_return_by_duration(ax3, result)
        
        # Plot 4: Cumulative Returns Analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_cumulative_returns(ax4, result)
        
        # Plot 5: Reversal Probability by Trend Duration
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_reversal_probability(ax5, result)
        
        fig.suptitle(
            f'{self.symbol} Post-Upward Trend Return Analysis\n'
            f'{result.analysis_period_start} to {result.analysis_period_end}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
        print("✅ Visualizations complete!")
    
    def _plot_price_with_trends(self, ax, result: PostTrendAnalysisResult):
        """Plot SPY price with upward trends and post-trend returns highlighted."""
        # Plot price
        ax.plot(self.data.index, self.data['Close'],
                label=f'{self.symbol} Close', color='blue', alpha=0.7, linewidth=1.5)
        
        # Highlight upward trends
        for trend in result.upward_trends:
            ax.axvspan(trend.start_date, trend.end_date,
                      alpha=0.2, color='green', label='_nolegend_')
        
        # Mark post-trend returns
        trends_with_returns = [t for t in result.upward_trends if t.return_3d_after is not None]
        for trend in trends_with_returns:
            # Get the date 3 days after trend ends
            trend_end_idx = self.data.index.get_loc(trend.end_date)
            if trend_end_idx + 3 < len(self.data):
                post_date = self.data.index[trend_end_idx + 3]
                post_price = trend.price_3d_after
                
                # Color based on return direction
                color = 'red' if trend.return_3d_after < 0 else 'darkgreen'
                marker = 'v' if trend.return_3d_after < 0 else '^'
                
                ax.scatter(post_date, post_price,
                          color=color, s=50, marker=marker, zorder=5, alpha=0.7, label='_nolegend_')
        
        # Add annotations for significant returns (|return| > 2%)
        significant_returns = [t for t in trends_with_returns if abs(t.return_3d_after) > 0.02]
        for trend in significant_returns[:10]:  # Annotate top 10
            trend_end_idx = self.data.index.get_loc(trend.end_date)
            if trend_end_idx + 3 < len(self.data):
                post_date = self.data.index[trend_end_idx + 3]
                post_price = trend.price_3d_after
                
                ax.annotate(
                    f'{trend.return_3d_after*100:+.1f}%',
                    xy=(post_date, post_price),
                    xytext=(10, 10 if trend.return_3d_after > 0 else -10),
                    textcoords='offset points',
                    fontsize=7,
                    color='darkgreen' if trend.return_3d_after > 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
                )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('SPY Price with Upward Trends (Green) and Post-Trend Returns')
        ax.grid(True, alpha=0.3)
        ax.legend(['SPY Close', 'Upward Trend', 'Negative Return', 'Positive Return'], loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_return_distributions(self, ax, result: PostTrendAnalysisResult):
        """Plot overlapping histograms of return distributions."""
        stats = result.statistics
        
        # Get returns for each period
        returns_1d = [t.return_1d_after * 100 for t in result.upward_trends if t.return_1d_after is not None]
        returns_2d = [t.return_2d_after * 100 for t in result.upward_trends if t.return_2d_after is not None]
        returns_3d = [t.return_3d_after * 100 for t in result.upward_trends if t.return_3d_after is not None]
        
        if not returns_1d and not returns_2d and not returns_3d:
            ax.text(0.5, 0.5, 'No return data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Return Distribution Comparison')
            return
        
        # Plot histograms
        bins = 30
        alpha = 0.5
        
        if returns_1d:
            ax.hist(returns_1d, bins=bins, color='blue', alpha=alpha, label='1-day', density=True)
            ax.axvline(stats.avg_return_1d * 100, color='blue', linestyle='--', linewidth=2)
        
        if returns_2d:
            ax.hist(returns_2d, bins=bins, color='orange', alpha=alpha, label='2-day', density=True)
            ax.axvline(stats.avg_return_2d * 100, color='orange', linestyle='--', linewidth=2)
        
        if returns_3d:
            ax.hist(returns_3d, bins=bins, color='green', alpha=alpha, label='3-day', density=True)
            ax.axvline(stats.avg_return_3d * 100, color='green', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    def _plot_return_by_duration(self, ax, result: PostTrendAnalysisResult):
        """Plot grouped bar chart of average return by trend duration."""
        stats = result.statistics
        durations = list(range(3, 11))
        
        # Get returns for each duration
        returns_1d = [stats.returns_by_duration_1d.get(d, 0) * 100 for d in durations]
        returns_2d = [stats.returns_by_duration_2d.get(d, 0) * 100 for d in durations]
        returns_3d = [stats.returns_by_duration_3d.get(d, 0) * 100 for d in durations]
        
        # Set up bar positions
        x = np.arange(len(durations))
        width = 0.25
        
        # Create bars
        ax.bar(x - width, returns_1d, width, label='1-day', color='blue', alpha=0.7)
        ax.bar(x, returns_2d, width, label='2-day', color='orange', alpha=0.7)
        ax.bar(x + width, returns_3d, width, label='3-day', color='green', alpha=0.7)
        
        # Add horizontal line at 0
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Trend Duration (days)')
        ax.set_ylabel('Average Return (%)')
        ax.set_title('Average Return by Trend Duration')
        ax.set_xticks(x)
        ax.set_xticklabels(durations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_cumulative_returns(self, ax, result: PostTrendAnalysisResult):
        """Plot cumulative average returns over the post-trend period."""
        stats = result.statistics
        
        # Create data for cumulative returns
        days = [0, 1, 2, 3]
        avg_returns = [0, stats.avg_return_1d * 100, stats.avg_return_2d * 100, stats.avg_return_3d * 100]
        
        # Plot line
        ax.plot(days, avg_returns, marker='o', linewidth=2.5, markersize=8, color='steelblue')
        
        # Add value labels
        for day, ret in zip(days, avg_returns):
            ax.annotate(f'{ret:.2f}%', xy=(day, ret), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9)
        
        # Add horizontal line at 0
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Days After Trend Ends')
        ax.set_ylabel('Cumulative Average Return (%)')
        ax.set_title('Cumulative Returns Analysis')
        ax.set_xticks(days)
        ax.grid(True, alpha=0.3)
    
    def _plot_reversal_probability(self, ax, result: PostTrendAnalysisResult):
        """Plot reversal probability (percentage of negative returns) by trend duration."""
        stats = result.statistics
        durations = list(range(3, 11))
        
        # Calculate reversal percentages by duration
        def calc_reversal_by_duration(period):
            reversal_by_dur = {}
            for duration in durations:
                trends_dur = [t for t in result.upward_trends 
                            if t.duration == duration and getattr(t, f'return_{period}d_after') is not None]
                if trends_dur:
                    neg_count = sum(1 for t in trends_dur if getattr(t, f'return_{period}d_after') < 0)
                    reversal_by_dur[duration] = (neg_count / len(trends_dur)) * 100
                else:
                    reversal_by_dur[duration] = 0.0
            return reversal_by_dur
        
        reversal_1d = [calc_reversal_by_duration(1).get(d, 0) for d in durations]
        reversal_2d = [calc_reversal_by_duration(2).get(d, 0) for d in durations]
        reversal_3d = [calc_reversal_by_duration(3).get(d, 0) for d in durations]
        
        # Set up bar positions
        x = np.arange(len(durations))
        width = 0.25
        
        # Create bars
        ax.bar(x - width, reversal_1d, width, label='1-day', color='lightcoral', alpha=0.7)
        ax.bar(x, reversal_2d, width, label='2-day', color='indianred', alpha=0.7)
        ax.bar(x + width, reversal_3d, width, label='3-day', color='darkred', alpha=0.7)
        
        # Add overall reversal rate lines
        ax.axhline(stats.pct_negative_1d, color='lightcoral', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(stats.pct_negative_2d, color='indianred', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(stats.pct_negative_3d, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Trend Duration (days)')
        ax.set_ylabel('Percentage of Negative Returns (%)')
        ax.set_title('Reversal Probability by Trend Duration')
        ax.set_xticks(x)
        ax.set_xticklabels(durations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')


def main():
    """
    Main function to run the post-upward trend return analysis.
    """
    print("🚀 Starting Post-Upward Trend Return Analysis...")
    
    # Initialize analyzer (will use last 12 months by default)
    analyzer = PostUpwardTrendReturnAnalyzer(symbol='SPY', analysis_period_months=12)
    
    # Run analysis
    result = analyzer.run_analysis()
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print("\n" + report)
    
    # Create plots
    analyzer.plot_results(result, save_path='post_upward_trend_return_analysis.png')
    
    print("\n✅ Analysis complete!")
    
    return result


if __name__ == "__main__":
    main()

