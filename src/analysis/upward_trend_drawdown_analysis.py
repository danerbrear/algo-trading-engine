"""
Upward Trend Drawdown Analysis

This module analyzes drawdowns that occur during upward trends in SPY.
An upward trend is defined as 3-10 consecutive days of positive returns.
A drawdown is a decline in price from a peak during the trend period.

The analysis provides statistical insights into how frequently drawdowns occur
during upward trends and their average magnitude.
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
    start_price: float
    end_price: float
    total_return: float  # Percentage return over the trend
    peak_price: float  # Highest price during the trend
    trough_price: float  # Lowest price after peak during the trend
    drawdown_pct: float  # Maximum drawdown during the trend
    has_drawdown: bool  # Whether a drawdown occurred


@dataclass(frozen=True)
class DrawdownStatistics:
    """Statistical summary of drawdowns across all upward trends."""
    total_trends_analyzed: int
    trends_with_drawdowns: int
    drawdown_percentage: float  # Percentage of trends with drawdowns
    average_drawdown: float  # Average drawdown across all trends
    average_drawdown_with_dd: float  # Average drawdown for trends that had drawdowns
    max_drawdown: float
    min_drawdown: float
    median_drawdown: float
    std_drawdown: float
    drawdowns_by_duration: Dict[int, float]  # Average drawdown by trend duration (3-10 days)


@dataclass(frozen=True)
class DrawdownAnalysisResult:
    """Complete results of the drawdown analysis."""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    upward_trends: List[UpwardTrend]
    statistics: DrawdownStatistics


class UpwardTrendDrawdownAnalyzer:
    """
    Analyzes drawdowns that occur during upward trends in SPY.
    
    This analyzer identifies periods of continuous positive returns (3-10 days)
    and measures any price declines (drawdowns) that occur during these trends.
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
            'drawdown_analysis'
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
        Identify all upward trends (3-10 consecutive days of positive returns).
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of UpwardTrend objects
        """
        print("\n🔍 Identifying upward trends...")
        trends = []
        i = 1  # Start at 1 because returns start at index 1
        
        while i < len(data) - 2:  # Need at least 3 days for minimum trend
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
                    
                    # Get price information
                    start_price = data['Close'].iloc[trend_start_idx - 1]  # Price before trend
                    end_price = data['Close'].iloc[trend_end_idx]
                    total_return = (end_price - start_price) / start_price
                    
                    # Calculate drawdown for this trend
                    peak_price, trough_price, drawdown_pct = self.calculate_drawdown_for_trend(
                        data, trend_start_idx, trend_end_idx
                    )
                    
                    # Create UpwardTrend object
                    trend = UpwardTrend(
                        start_date=data.index[trend_start_idx],
                        end_date=data.index[trend_end_idx],
                        duration=consecutive_positive_days,
                        start_price=start_price,
                        end_price=end_price,
                        total_return=total_return,
                        peak_price=peak_price,
                        trough_price=trough_price,
                        drawdown_pct=drawdown_pct,
                        has_drawdown=(drawdown_pct < 0)
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
    
    def calculate_drawdown_for_trend(
        self, 
        data: pd.DataFrame, 
        start_idx: int, 
        end_idx: int
    ) -> Tuple[float, float, float]:
        """
        Calculate the maximum drawdown during a specific trend period.
        
        Args:
            data: DataFrame with price data
            start_idx: Starting index of the trend
            end_idx: Ending index of the trend
            
        Returns:
            Tuple of (peak_price, trough_price, drawdown_pct)
        """
        # Get high and low prices during the trend period
        trend_highs = data['High'].iloc[start_idx:end_idx+1]
        trend_lows = data['Low'].iloc[start_idx:end_idx+1]
        
        # Track the running peak and maximum drawdown
        peak_price = trend_highs.iloc[0]
        max_drawdown_pct = 0.0
        trough_price = peak_price
        
        for i in range(len(trend_highs)):
            current_high = trend_highs.iloc[i]
            current_low = trend_lows.iloc[i]
            
            # Update peak if we've reached a new high
            if current_high > peak_price:
                peak_price = current_high
            
            # Calculate drawdown from peak to current low
            drawdown = (current_low - peak_price) / peak_price
            
            # Update maximum drawdown if this is worse (more negative)
            if drawdown < max_drawdown_pct:
                max_drawdown_pct = drawdown
                trough_price = current_low
        
        return peak_price, trough_price, max_drawdown_pct
    
    def calculate_statistics(self, trends: List[UpwardTrend]) -> DrawdownStatistics:
        """
        Calculate statistical measures of drawdowns across all trends.
        
        Args:
            trends: List of identified upward trends
            
        Returns:
            DrawdownStatistics object with summary metrics
        """
        print("\n📈 Calculating statistics...")
        
        if not trends:
            # Return empty statistics if no trends found
            return DrawdownStatistics(
                total_trends_analyzed=0,
                trends_with_drawdowns=0,
                drawdown_percentage=0.0,
                average_drawdown=0.0,
                average_drawdown_with_dd=0.0,
                max_drawdown=0.0,
                min_drawdown=0.0,
                median_drawdown=0.0,
                std_drawdown=0.0,
                drawdowns_by_duration={}
            )
        
        # Filter trends with actual drawdowns
        trends_with_drawdowns = [t for t in trends if t.has_drawdown]
        
        # Overall statistics
        total_trends = len(trends)
        num_with_drawdowns = len(trends_with_drawdowns)
        drawdown_percentage = (num_with_drawdowns / total_trends * 100) if total_trends > 0 else 0
        
        # Calculate average across ALL trends (convert to positive percentages for display)
        all_drawdowns = [abs(t.drawdown_pct) for t in trends]
        average_drawdown = np.mean(all_drawdowns) if all_drawdowns else 0.0
        
        # Calculate average for trends with drawdowns only
        dd_only = [abs(t.drawdown_pct) for t in trends if t.has_drawdown]
        average_drawdown_with_dd = np.mean(dd_only) if dd_only else 0.0
        
        # Min, max, median, std
        max_drawdown = max(dd_only) if dd_only else 0.0
        min_drawdown = min(dd_only) if dd_only else 0.0
        median_drawdown = np.median(dd_only) if dd_only else 0.0
        std_drawdown = np.std(dd_only) if dd_only else 0.0
        
        # Group by duration
        drawdowns_by_duration = {}
        for duration in range(3, 11):
            duration_trends = [t for t in trends if t.duration == duration]
            if duration_trends:
                avg_dd = np.mean([abs(t.drawdown_pct) for t in duration_trends])
                drawdowns_by_duration[duration] = avg_dd
            else:
                drawdowns_by_duration[duration] = 0.0
        
        print(f"✅ Statistics calculated")
        print(f"   Total trends: {total_trends}")
        print(f"   Trends with drawdowns: {num_with_drawdowns} ({drawdown_percentage:.1f}%)")
        print(f"   Average drawdown: {average_drawdown*100:.3f}%")
        
        return DrawdownStatistics(
            total_trends_analyzed=total_trends,
            trends_with_drawdowns=num_with_drawdowns,
            drawdown_percentage=drawdown_percentage,
            average_drawdown=average_drawdown,
            average_drawdown_with_dd=average_drawdown_with_dd,
            max_drawdown=max_drawdown,
            min_drawdown=min_drawdown,
            median_drawdown=median_drawdown,
            std_drawdown=std_drawdown,
            drawdowns_by_duration=drawdowns_by_duration
        )
    
    def run_analysis(self) -> DrawdownAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            DrawdownAnalysisResult with all findings
        """
        print("🚀 Starting Upward Trend Drawdown Analysis...")
        
        # Load data
        data = self.load_data()
        
        # Identify upward trends
        trends = self.identify_upward_trends(data)
        
        # Calculate statistics
        statistics = self.calculate_statistics(trends)
        
        # Create result object
        result = DrawdownAnalysisResult(
            analysis_period_start=data.index[0].strftime('%Y-%m-%d'),
            analysis_period_end=data.index[-1].strftime('%Y-%m-%d'),
            total_trading_days=len(data),
            upward_trends=trends,
            statistics=statistics
        )
        
        print("\n✅ Analysis complete!")
        
        return result
    
    def generate_report(self, result: DrawdownAnalysisResult) -> str:
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
        report.append("UPWARD TREND DRAWDOWN ANALYSIS REPORT")
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
        
        # Drawdown analysis
        report.append("DRAWDOWN ANALYSIS:")
        trends_without = stats.total_trends_analyzed - stats.trends_with_drawdowns
        pct_without = (trends_without / stats.total_trends_analyzed * 100) if stats.total_trends_analyzed > 0 else 0
        report.append(f"  Trends with Drawdowns: {stats.trends_with_drawdowns} ({stats.drawdown_percentage:.1f}%)")
        report.append(f"  Trends without Drawdowns: {trends_without} ({pct_without:.1f}%)")
        report.append("")
        
        # Drawdown statistics (across all trends)
        report.append("DRAWDOWN STATISTICS (across all trends):")
        report.append(f"  Average Drawdown: {stats.average_drawdown*100:.3f}%")
        report.append(f"  Median Drawdown: {stats.median_drawdown*100:.3f}%")
        report.append(f"  Standard Deviation: {stats.std_drawdown*100:.3f}%")
        report.append(f"  Maximum Drawdown: {stats.max_drawdown*100:.3f}%")
        report.append(f"  Minimum Drawdown: {stats.min_drawdown*100:.3f}%")
        report.append("")
        
        # Drawdown statistics (only trends with drawdowns)
        if stats.trends_with_drawdowns > 0:
            report.append("DRAWDOWN STATISTICS (only trends with drawdowns):")
            report.append(f"  Average Drawdown: {stats.average_drawdown_with_dd*100:.3f}%")
            report.append("")
        
        # Average drawdown by trend duration
        report.append("AVERAGE DRAWDOWN BY TREND DURATION:")
        for duration in range(3, 11):
            avg_dd = stats.drawdowns_by_duration.get(duration, 0.0)
            report.append(f"  {duration}-day trends: {avg_dd*100:.3f}%")
        report.append("")
        
        # Top 5 largest drawdowns
        trends_with_dd = [t for t in trends if t.has_drawdown]
        if trends_with_dd:
            sorted_trends = sorted(trends_with_dd, key=lambda t: t.drawdown_pct)
            top_5 = sorted_trends[:5]
            
            report.append("TOP 5 LARGEST DRAWDOWNS:")
            for i, trend in enumerate(top_5, 1):
                report.append(
                    f"  {i}. {trend.start_date.strftime('%Y-%m-%d')}: "
                    f"{trend.drawdown_pct*100:.2f}% ({trend.duration}-day trend)"
                )
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_results(self, result: DrawdownAnalysisResult, save_path: str = None):
        """
        Create visualizations of the analysis results.
        
        Args:
            result: Complete analysis results
            save_path: Optional path to save the plot
        """
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: SPY Price with Upward Trends and Drawdowns (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_with_trends(ax1, result)
        
        # Plot 2: Drawdown Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown_distribution(ax2, result)
        
        # Plot 3: Average Drawdown by Trend Duration
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_by_duration(ax3, result)
        
        # Plot 4: Cumulative Trend Analysis (spans 2 columns)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_cumulative_trends(ax4, result)
        
        fig.suptitle(
            f'{self.symbol} Upward Trend Drawdown Analysis\n'
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
    
    def _plot_price_with_trends(self, ax, result: DrawdownAnalysisResult):
        """Plot SPY price with upward trends and drawdowns highlighted."""
        # Plot price
        ax.plot(self.data.index, self.data['Close'], 
                label=f'{self.symbol} Close', color='blue', alpha=0.7, linewidth=1.5)
        
        # Highlight upward trends
        for trend in result.upward_trends:
            ax.axvspan(trend.start_date, trend.end_date, 
                      alpha=0.2, color='green', label='_nolegend_')
        
        # Mark drawdown trough points
        trends_with_dd = [t for t in result.upward_trends if t.has_drawdown]
        if trends_with_dd:
            # Find the date for each trough
            for trend in trends_with_dd:
                # Find the index where trough occurred
                trend_data = self.data.loc[trend.start_date:trend.end_date]
                trough_date = trend_data[trend_data['Low'] == trend.trough_price].index[0]
                ax.scatter(trough_date, trend.trough_price, 
                          color='red', s=50, zorder=5, alpha=0.7, label='_nolegend_')
        
        # Add annotations for significant drawdowns (> 2%)
        significant_dd = [t for t in trends_with_dd if abs(t.drawdown_pct) > 0.02]
        for trend in significant_dd[:5]:  # Annotate top 5
            trend_data = self.data.loc[trend.start_date:trend.end_date]
            trough_date = trend_data[trend_data['Low'] == trend.trough_price].index[0]
            ax.annotate(
                f'{trend.drawdown_pct*100:.1f}%',
                xy=(trough_date, trend.trough_price),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=8,
                color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
            )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('SPY Price with Upward Trends (Green) and Drawdowns (Red)')
        ax.grid(True, alpha=0.3)
        ax.legend(['SPY Close', 'Upward Trend', 'Drawdown Trough'], loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_drawdown_distribution(self, ax, result: DrawdownAnalysisResult):
        """Plot histogram of drawdown distribution."""
        drawdowns = [abs(t.drawdown_pct) * 100 for t in result.upward_trends if t.has_drawdown]
        
        if not drawdowns:
            ax.text(0.5, 0.5, 'No drawdowns found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Drawdown Distribution')
            return
        
        ax.hist(drawdowns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add vertical lines for average and median
        avg_dd = result.statistics.average_drawdown_with_dd * 100
        median_dd = result.statistics.median_drawdown * 100
        
        ax.axvline(avg_dd, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dd:.2f}%')
        ax.axvline(median_dd, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_dd:.2f}%')
        
        ax.set_xlabel('Drawdown (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Drawdown Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_drawdown_by_duration(self, ax, result: DrawdownAnalysisResult):
        """Plot average drawdown by trend duration."""
        durations = list(range(3, 11))
        avg_drawdowns = [result.statistics.drawdowns_by_duration.get(d, 0) * 100 
                        for d in durations]
        
        # Calculate standard deviations for error bars
        std_devs = []
        for duration in durations:
            duration_trends = [t for t in result.upward_trends if t.duration == duration]
            if duration_trends:
                dd_values = [abs(t.drawdown_pct) * 100 for t in duration_trends]
                std_devs.append(np.std(dd_values) if len(dd_values) > 1 else 0)
            else:
                std_devs.append(0)
        
        bars = ax.bar(durations, avg_drawdowns, color='steelblue', alpha=0.7, 
                     yerr=std_devs, capsize=5, edgecolor='black')
        
        # Add overall average line
        overall_avg = result.statistics.average_drawdown * 100
        ax.axhline(overall_avg, color='red', linestyle='--', linewidth=2, 
                  label=f'Overall Avg: {overall_avg:.2f}%')
        
        ax.set_xlabel('Trend Duration (days)')
        ax.set_ylabel('Average Drawdown (%)')
        ax.set_title('Average Drawdown by Trend Duration')
        ax.set_xticks(durations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_cumulative_trends(self, ax, result: DrawdownAnalysisResult):
        """Plot cumulative number of upward trends over time."""
        # Sort trends by start date
        sorted_trends = sorted(result.upward_trends, key=lambda t: t.start_date)
        
        if not sorted_trends:
            ax.text(0.5, 0.5, 'No trends found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cumulative Upward Trends')
            return
        
        # Create cumulative count by duration
        duration_data = {d: {'dates': [], 'counts': []} for d in range(3, 11)}
        
        for duration in range(3, 11):
            duration_trends = [t for t in sorted_trends if t.duration == duration]
            cumulative = 0
            for trend in duration_trends:
                cumulative += 1
                duration_data[duration]['dates'].append(trend.start_date)
                duration_data[duration]['counts'].append(cumulative)
        
        # Plot each duration
        colors = plt.cm.viridis(np.linspace(0, 1, 8))
        for i, duration in enumerate(range(3, 11)):
            if duration_data[duration]['dates']:
                ax.step(duration_data[duration]['dates'], 
                       duration_data[duration]['counts'],
                       where='post',
                       label=f'{duration} days',
                       color=colors[i],
                       linewidth=1.5)
        
        # Plot total cumulative
        all_dates = [t.start_date for t in sorted_trends]
        cumulative_total = list(range(1, len(sorted_trends) + 1))
        ax.plot(all_dates, cumulative_total, 
               color='black', linewidth=2.5, label='Total', linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Number of Trends')
        ax.set_title('Cumulative Upward Trends Over Time')
        ax.legend(loc='best', ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def main():
    """
    Main function to run the upward trend drawdown analysis.
    """
    print("🚀 Starting Upward Trend Drawdown Analysis...")
    
    # Initialize analyzer (will use last 12 months by default)
    analyzer = UpwardTrendDrawdownAnalyzer(symbol='SPY', analysis_period_months=12)
    
    # Run analysis
    result = analyzer.run_analysis()
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print("\n" + report)
    
    # Create plots
    analyzer.plot_results(result, save_path='upward_trend_drawdown_analysis.png')
    
    print("\n✅ Analysis complete!")
    
    return result


if __name__ == "__main__":
    main()

