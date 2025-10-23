"""
Daily Drawdown Likelihood Analysis

This module analyzes the likelihood of drawdowns occurring on each specific day
of an upward trend (days 1-10).

An upward trend is defined as 3-10 consecutive days of positive returns.
A daily drawdown is defined as an intraday decline where Low < max(Open, Previous Close).

The analysis provides insights into whether drawdowns are more likely to occur
early in a trend, late in a trend, or uniformly distributed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Import data retrieval classes
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from common.data_retriever import DataRetriever


@dataclass(frozen=True)
class DailyDrawdown:
    """Represents whether a drawdown occurred on a specific day."""
    date: pd.Timestamp
    day_position: int  # Position within the trend (1-10)
    had_drawdown: bool
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    prev_close: float
    drawdown_magnitude: float  # Percentage drawdown if occurred
    drawdown_type: str  # 'intraday' or 'gap_down' or 'none'


@dataclass(frozen=True)
class TrendWithDailyDrawdowns:
    """Represents an upward trend with daily drawdown information."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int  # Number of days (3-10)
    start_price: float
    end_price: float
    total_return: float
    daily_drawdowns: Tuple[DailyDrawdown, ...]  # One entry per day in the trend
    total_days_with_drawdowns: int
    drawdown_frequency: float  # Percentage of days with drawdowns


@dataclass(frozen=True)
class DailyLikelihoodStatistics:
    """Statistics about drawdown likelihood by day position."""
    day_position: int  # 1-10
    total_trends_reaching_day: int  # How many trends made it to this day
    trends_with_drawdown_on_day: int  # How many had a drawdown on this day
    likelihood_percentage: float  # Probability of drawdown on this day
    average_drawdown_magnitude: float  # Average size of drawdowns on this day
    max_drawdown_magnitude: float
    min_drawdown_magnitude: float


@dataclass(frozen=True)
class DailyDrawdownAnalysisResult:
    """Complete results of the daily drawdown likelihood analysis."""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    total_trends_analyzed: int
    trends_with_daily_data: Tuple[TrendWithDailyDrawdowns, ...]
    likelihood_by_day_position: Dict[int, DailyLikelihoodStatistics]
    overall_daily_likelihood: float  # Average likelihood across all days


class DailyDrawdownLikelihoodAnalyzer:
    """
    Analyzes the likelihood of drawdowns occurring on each specific day
    of an upward trend (days 1-10).
    
    This analyzer provides insights into whether drawdowns are more likely
    to occur early in a trend, late in a trend, or uniformly distributed.
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
            DataFrame with OHLCV data and calculated returns
        """
        print(f"📊 Loading {self.symbol} data from {self.start_date}...")
        
        # Fetch data using the existing data retriever
        self.data = self.data_retriever.fetch_data_for_period(
            self.start_date, 
            'daily_likelihood_analysis'
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
    
    def identify_upward_trends(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identify all upward trends (3-10 consecutive days of positive returns).
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of tuples (start_index, end_index) for each trend
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
                
                # If trend is between 3-10 days, record it
                if 3 <= consecutive_positive_days <= 10:
                    trend_end_idx = trend_start_idx + consecutive_positive_days - 1
                    trends.append((trend_start_idx, trend_end_idx))
                
                # Move index past this trend
                i = j
            else:
                i += 1
        
        print(f"✅ Identified {len(trends)} upward trends")
        
        return trends
    
    def check_daily_drawdown(self, data: pd.DataFrame, day_idx: int) -> Tuple[bool, float, str]:
        """
        Check if a drawdown occurred on a specific day.
        
        A drawdown occurs if:
        - Low < max(Open, Previous Close) (intraday decline from reference point)
        
        Args:
            data: DataFrame with OHLC data
            day_idx: Index of the day to check
            
        Returns:
            Tuple of (had_drawdown, drawdown_magnitude, drawdown_type)
        """
        if day_idx == 0:
            return False, 0.0, 'none'
        
        open_price = data['Open'].iloc[day_idx]
        high_price = data['High'].iloc[day_idx]
        low_price = data['Low'].iloc[day_idx]
        close_price = data['Close'].iloc[day_idx]
        prev_close = data['Close'].iloc[day_idx - 1]
        
        # Reference point: the higher of open or previous close
        reference_point = max(open_price, prev_close)
        
        # Check if low dropped below reference
        if low_price < reference_point:
            drawdown_magnitude = (reference_point - low_price) / reference_point
            
            # Classify type based on opening vs previous close
            if low_price < prev_close and open_price >= prev_close:
                drawdown_type = 'intraday'  # Opened at/above prev close but fell below
            elif open_price < prev_close:
                drawdown_type = 'gap_down'  # Opened below previous close
            else:
                drawdown_type = 'intraday'  # Low < Open but Low >= Prev Close
            
            return True, drawdown_magnitude, drawdown_type
        else:
            return False, 0.0, 'none'
    
    def analyze_trend_daily_drawdowns(
        self, 
        data: pd.DataFrame, 
        start_idx: int, 
        end_idx: int
    ) -> TrendWithDailyDrawdowns:
        """
        Analyze daily drawdowns for a specific trend.
        
        Args:
            data: DataFrame with price data
            start_idx: Starting index of the trend
            end_idx: Ending index of the trend
            
        Returns:
            TrendWithDailyDrawdowns object with complete daily analysis
        """
        duration = end_idx - start_idx + 1
        daily_drawdowns = []
        days_with_drawdowns = 0
        
        for day_offset in range(duration):
            day_idx = start_idx + day_offset
            day_position = day_offset + 1  # 1-based position
            
            # Check for drawdown on this day
            had_drawdown, magnitude, dd_type = self.check_daily_drawdown(data, day_idx)
            
            if had_drawdown:
                days_with_drawdowns += 1
            
            # Create DailyDrawdown object
            daily_dd = DailyDrawdown(
                date=data.index[day_idx],
                day_position=day_position,
                had_drawdown=had_drawdown,
                open_price=data['Open'].iloc[day_idx],
                high_price=data['High'].iloc[day_idx],
                low_price=data['Low'].iloc[day_idx],
                close_price=data['Close'].iloc[day_idx],
                prev_close=data['Close'].iloc[day_idx - 1] if day_idx > 0 else data['Close'].iloc[day_idx],
                drawdown_magnitude=magnitude,
                drawdown_type=dd_type
            )
            daily_drawdowns.append(daily_dd)
        
        # Create TrendWithDailyDrawdowns object
        trend = TrendWithDailyDrawdowns(
            start_date=data.index[start_idx],
            end_date=data.index[end_idx],
            duration=duration,
            start_price=data['Close'].iloc[start_idx - 1],
            end_price=data['Close'].iloc[end_idx],
            total_return=(data['Close'].iloc[end_idx] - data['Close'].iloc[start_idx - 1]) / data['Close'].iloc[start_idx - 1],
            daily_drawdowns=tuple(daily_drawdowns),
            total_days_with_drawdowns=days_with_drawdowns,
            drawdown_frequency=days_with_drawdowns / duration if duration > 0 else 0.0
        )
        
        return trend
    
    def calculate_likelihood_by_day_position(
        self, 
        trends: List[TrendWithDailyDrawdowns]
    ) -> Dict[int, DailyLikelihoodStatistics]:
        """
        Calculate likelihood that the upward trend ends after each day position.
        
        For each day position N, calculates the probability that the trend ends
        after day N (i.e., the trend was exactly N days long).
        
        Args:
            trends: List of trends with daily drawdown data
            
        Returns:
            Dictionary mapping day position to likelihood statistics
        """
        print("\n📈 Calculating likelihood by day position...")
        
        likelihood_stats = {}
        
        for day_position in range(1, 11):  # Days 1-10
            # Count trends that reached this day position
            trends_reaching_day = [t for t in trends if t.duration >= day_position]
            total_reaching = len(trends_reaching_day)
            
            if total_reaching == 0:
                continue
            
            # Count trends that ended exactly at this day position
            # (i.e., duration equals day_position)
            trends_ending_at_day = [t for t in trends if t.duration == day_position]
            num_ending = len(trends_ending_at_day)
            
            likelihood_pct = (num_ending / total_reaching * 100) if total_reaching > 0 else 0.0
            
            # For trends that ended at this position, calculate the "drawdown" magnitude
            # which is the negative return on the day after the trend ended
            if trends_ending_at_day:
                # Get the last day's return (which would be the first negative return)
                # Since we don't store the negative return that ended the trend,
                # we'll use the trend's final positive return as a proxy
                magnitudes = [t.total_return / t.duration for t in trends_ending_at_day]
                avg_magnitude = np.mean(magnitudes)
                max_magnitude = max(magnitudes)
                min_magnitude = min(magnitudes)
            else:
                avg_magnitude = 0.0
                max_magnitude = 0.0
                min_magnitude = 0.0
            
            # Create statistics object
            stats = DailyLikelihoodStatistics(
                day_position=day_position,
                total_trends_reaching_day=total_reaching,
                trends_with_drawdown_on_day=num_ending,
                likelihood_percentage=likelihood_pct,
                average_drawdown_magnitude=avg_magnitude,
                max_drawdown_magnitude=max_magnitude,
                min_drawdown_magnitude=min_magnitude
            )
            
            likelihood_stats[day_position] = stats
            
            print(f"   Day {day_position}: {likelihood_pct:.1f}% likelihood ({num_ending}/{total_reaching} trends ended)")
        
        return likelihood_stats
    
    def run_analysis(self) -> DailyDrawdownAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            DailyDrawdownAnalysisResult with all findings
        """
        print("🚀 Starting Daily Drawdown Likelihood Analysis...")
        
        # Load data
        data = self.load_data()
        
        # Identify upward trends
        trend_indices = self.identify_upward_trends(data)
        
        # Analyze each trend for daily drawdowns
        print("\n🔬 Analyzing daily drawdowns for each trend...")
        trends_with_daily_data = []
        
        for start_idx, end_idx in trend_indices:
            trend = self.analyze_trend_daily_drawdowns(data, start_idx, end_idx)
            trends_with_daily_data.append(trend)
        
        print(f"✅ Analyzed {len(trends_with_daily_data)} trends")
        
        # Calculate likelihood by day position
        likelihood_stats = self.calculate_likelihood_by_day_position(trends_with_daily_data)
        
        # Calculate overall daily likelihood
        if likelihood_stats:
            overall_likelihood = np.mean([stats.likelihood_percentage for stats in likelihood_stats.values()])
        else:
            overall_likelihood = 0.0
        
        # Create result object
        result = DailyDrawdownAnalysisResult(
            analysis_period_start=data.index[0].strftime('%Y-%m-%d'),
            analysis_period_end=data.index[-1].strftime('%Y-%m-%d'),
            total_trading_days=len(data),
            total_trends_analyzed=len(trends_with_daily_data),
            trends_with_daily_data=tuple(trends_with_daily_data),
            likelihood_by_day_position=likelihood_stats,
            overall_daily_likelihood=overall_likelihood
        )
        
        print("\n✅ Analysis complete!")
        
        return result
    
    def generate_report(self, result: DailyDrawdownAnalysisResult) -> str:
        """
        Generate a text report of the analysis results.
        
        Args:
            result: Complete analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("DAILY DRAWDOWN LIKELIHOOD ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Symbol: {self.symbol}")
        report.append(f"Analysis Period: {result.analysis_period_start} to {result.analysis_period_end}")
        report.append(f"Total Trading Days: {result.total_trading_days}")
        report.append(f"Total Upward Trends Analyzed: {result.total_trends_analyzed}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"  Average Likelihood of Trend Ending: {result.overall_daily_likelihood:.1f}%")
        report.append(f"  (This is the average probability that a trend ends at each day position)")
        report.append("")
        
        # Likelihood by day position
        report.append("LIKELIHOOD BY DAY POSITION:")
        report.append("-" * 80)
        
        for day_pos in range(1, 11):
            if day_pos in result.likelihood_by_day_position:
                stats = result.likelihood_by_day_position[day_pos]
                report.append(f"Day {day_pos}{' (First day of trend)' if day_pos == 1 else ''}:")
                report.append(f"  Trends Reaching Day {day_pos}: {stats.total_trends_reaching_day}")
                report.append(f"  Trends Ending on Day {day_pos}: {stats.trends_with_drawdown_on_day} ({stats.likelihood_percentage:.1f}%)")
                if stats.trends_with_drawdown_on_day > 0:
                    report.append(f"  Average Return per Day: {stats.average_drawdown_magnitude*100:.2f}%")
                report.append("")
        
        # Insights
        if result.likelihood_by_day_position:
            most_likely = max(result.likelihood_by_day_position.values(), 
                            key=lambda s: s.likelihood_percentage)
            least_likely = min(result.likelihood_by_day_position.values(), 
                             key=lambda s: s.likelihood_percentage)
            
            report.append("INSIGHTS:")
            report.append(f"  Most Vulnerable Day (highest likelihood of trend ending): Day {most_likely.day_position} ({most_likely.likelihood_percentage:.1f}%)")
            report.append(f"  Most Stable Day (lowest likelihood of trend ending): Day {least_likely.day_position} ({least_likely.likelihood_percentage:.1f}%)")
            report.append("")
        
        # Trend duration breakdown
        duration_counts = {}
        for trend in result.trends_with_daily_data:
            duration_counts[trend.duration] = duration_counts.get(trend.duration, 0) + 1
        
        report.append("TREND DURATION BREAKDOWN:")
        for duration in range(3, 11):
            count = duration_counts.get(duration, 0)
            report.append(f"  {duration}-day trends: {count} trends")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_results(self, result: DailyDrawdownAnalysisResult, save_path: str = None):
        """
        Create visualizations of the analysis results.
        
        Args:
            result: Complete analysis results
            save_path: Optional path to save the plot
        """
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: SPY Price with Drawdown Days Highlighted (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_with_drawdown_days(ax1, result)
        
        # Plot 2: Likelihood by Day Position (Bar Chart)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_likelihood_bar_chart(ax2, result)
        
        # Plot 3: Average Drawdown Magnitude by Day Position
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_magnitude_by_day(ax3, result)
        
        # Plot 4: Heatmap - Likelihood by Day Position and Trend Duration
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_likelihood_heatmap(ax4, result)
        
        # Plot 5: Cumulative Likelihood Curve
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_cumulative_likelihood(ax5, result)
        
        fig.suptitle(
            f'{self.symbol} Daily Drawdown Likelihood Analysis\n'
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
    
    def _plot_price_with_drawdown_days(self, ax, result: DailyDrawdownAnalysisResult):
        """Plot SPY price with drawdown days highlighted."""
        # Plot price
        ax.plot(self.data.index, self.data['Close'], 
                label=f'{self.symbol} Close', color='blue', alpha=0.7, linewidth=1.5)
        
        # Highlight upward trends
        for trend in result.trends_with_daily_data:
            ax.axvspan(trend.start_date, trend.end_date, 
                      alpha=0.15, color='green', label='_nolegend_')
        
        # Mark days with drawdowns
        for trend in result.trends_with_daily_data:
            for daily_dd in trend.daily_drawdowns:
                if daily_dd.had_drawdown:
                    # Marker size proportional to drawdown magnitude
                    marker_size = 30 + daily_dd.drawdown_magnitude * 500
                    ax.scatter(daily_dd.date, daily_dd.low_price,
                             color='red', s=marker_size, alpha=0.6, 
                             zorder=5, label='_nolegend_')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('SPY Price with Upward Trends (Green) and Drawdown Days (Red)')
        ax.grid(True, alpha=0.3)
        ax.legend(['SPY Close', 'Upward Trend', 'Drawdown Day'], loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_likelihood_bar_chart(self, ax, result: DailyDrawdownAnalysisResult):
        """Plot likelihood that trend ends at each day position."""
        day_positions = sorted(result.likelihood_by_day_position.keys())
        likelihoods = [result.likelihood_by_day_position[dp].likelihood_percentage 
                      for dp in day_positions]
        
        # Color gradient from green to red
        colors = plt.cm.RdYlGn_r(np.array(likelihoods) / 100)
        
        bars = ax.bar(day_positions, likelihoods, color=colors, alpha=0.7, edgecolor='black')
        
        # Add average line
        ax.axhline(result.overall_daily_likelihood, color='blue', linestyle='--', 
                  linewidth=2, label=f'Average: {result.overall_daily_likelihood:.1f}%')
        
        ax.set_xlabel('Day Position in Trend')
        ax.set_ylabel('Probability of Trend Ending (%)')
        ax.set_title('Likelihood of Upward Trend Ending at Each Day Position')
        ax.set_xticks(day_positions)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_magnitude_by_day(self, ax, result: DailyDrawdownAnalysisResult):
        """Plot average drawdown magnitude by day position."""
        day_positions = sorted(result.likelihood_by_day_position.keys())
        magnitudes = [result.likelihood_by_day_position[dp].average_drawdown_magnitude * 100
                     for dp in day_positions]
        
        ax.plot(day_positions, magnitudes, marker='o', linewidth=2, 
               markersize=8, color='darkred')
        
        ax.set_xlabel('Day Position in Trend')
        ax.set_ylabel('Average Drawdown Magnitude (%)')
        ax.set_title('Average Drawdown Size by Day Position')
        ax.set_xticks(day_positions)
        ax.grid(True, alpha=0.3)
    
    def _plot_likelihood_heatmap(self, ax, result: DailyDrawdownAnalysisResult):
        """Plot heatmap: For trends that survived through day N, what % ended on day M?"""
        # Matrix: rows = "survived through day N", cols = "ended on day M"
        survived_days = range(1, 10)  # Can survive through days 1-9
        ending_days = range(3, 11)    # Can end on days 3-10
        
        # Initialize matrix with NaN
        matrix = np.full((len(survived_days), len(ending_days)), np.nan)
        
        # For each "survived through day N"
        for i, survived_day in enumerate(survived_days):
            # Get trends that survived through this day (duration >= survived_day)
            trends_surviving = [t for t in result.trends_with_daily_data 
                               if t.duration >= survived_day]
            
            if not trends_surviving:
                continue
            
            total_surviving = len(trends_surviving)
            
            # For each possible ending day
            for j, ending_day in enumerate(ending_days):
                if ending_day <= survived_day:
                    # Can't end before or on the day we survived through
                    matrix[i, j] = np.nan
                else:
                    # Count trends that ended exactly on this day
                    trends_ending_here = [t for t in trends_surviving 
                                         if t.duration == ending_day]
                    count_ending = len(trends_ending_here)
                    
                    # Calculate percentage
                    percentage = (count_ending / total_surviving) * 100
                    matrix[i, j] = percentage
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Likelihood (%)'}, ax=ax,
                   xticklabels=[f'D{d}' for d in ending_days],
                   yticklabels=[f'≥D{d}' for d in survived_days],
                   vmin=0, vmax=100, mask=np.isnan(matrix))
        
        ax.set_xlabel('Ending Day Position')
        ax.set_ylabel('Trends That Survived Through')
        ax.set_title('Likelihood of Ending: For Trends Surviving ≥N Days, % Ending on Day M')
    
    def _plot_cumulative_likelihood(self, ax, result: DailyDrawdownAnalysisResult):
        """Plot cumulative probability that trends have ended by day N."""
        day_positions = sorted(result.likelihood_by_day_position.keys())
        
        # Calculate cumulative probability (percentage of all trends that ended by day N)
        cumulative_probs = []
        total_trends = len(result.trends_with_daily_data)
        
        for target_day in day_positions:
            # Count how many trends ended on or before this day
            trends_ended_by_day = len([t for t in result.trends_with_daily_data 
                                       if t.duration <= target_day])
            
            cumulative_prob = (trends_ended_by_day / total_trends) * 100
            cumulative_probs.append(cumulative_prob)
        
        ax.plot(day_positions, cumulative_probs, marker='o', linewidth=2.5,
               markersize=8, color='darkblue')
        ax.fill_between(day_positions, cumulative_probs, alpha=0.3)
        
        ax.set_xlabel('Day Position in Trend')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title('Cumulative % of Trends That Have Ended by Day N')
        ax.set_xticks(day_positions)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key milestones
        for i, (day, prob) in enumerate(zip(day_positions, cumulative_probs)):
            if day in [3, 5, 7, 10] or i == len(day_positions) - 1:
                ax.annotate(f'{prob:.0f}%', 
                          xy=(day, prob),
                          xytext=(0, 10),
                          textcoords='offset points',
                          ha='center',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def main():
    """
    Main function to run the daily drawdown likelihood analysis.
    """
    print("🚀 Starting Daily Drawdown Likelihood Analysis...")
    
    # Initialize analyzer (will use last 12 months by default)
    analyzer = DailyDrawdownLikelihoodAnalyzer(symbol='SPY', analysis_period_months=12)
    
    # Run analysis
    result = analyzer.run_analysis()
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print("\n" + report)
    
    # Create plots
    analyzer.plot_results(result, save_path='daily_drawdown_likelihood_analysis.png')
    
    print("\n✅ Analysis complete!")
    
    return result


if __name__ == "__main__":
    main()

