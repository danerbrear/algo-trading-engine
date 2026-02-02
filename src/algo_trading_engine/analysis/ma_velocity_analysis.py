"""
Moving Average Velocity/Elasticity Analysis

This module analyzes the elasticity/velocity (short term MA / longer term MA) that best signals
upward and downward trends in SPY over the last 6 months.

The analysis focuses purely on moving average analysis from SPY price changes without any
trading strategy, position management, or actual trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import data retrieval classes
from algo_trading_engine.common.data_retriever import DataRetriever


@dataclass
class TrendSignal:
    """Represents a trend signal from MA velocity"""
    signal_date: pd.Timestamp
    ma_velocity: float
    short_ma: int
    long_ma: int
    signal_type: str  # 'up' or 'down'
    success: bool
    trend_duration: int
    trend_return: float


@dataclass
class MAVelocityResult:
    """Results for a specific MA velocity combination"""
    short_ma: int
    long_ma: int
    velocity: float
    success_rate: float
    total_signals: int
    successful_signals: int
    avg_trend_duration: float
    avg_trend_return: float


class MAVelocityAnalyzer:
    """
    Analyzes moving average velocity/elasticity for SPY trend signals.
    
    Velocity is defined as short_ma / long_ma ratio.
    The goal is to find MA combinations that best signal upward and downward trends.
    """
    
    def __init__(self, symbol: str = 'SPY'):
        """
        Initialize the MA Velocity Analyzer.
        
        Args:
            symbol: Stock symbol to analyze (default: 'SPY')
        """
        # Calculate start date as 6 months ago from today
        today = datetime.now()
        six_months_ago = today - timedelta(days=180)
        self.start_date = six_months_ago.strftime('%Y-%m-%d')
        self.symbol = symbol
        self.data_retriever = DataRetriever(symbol=symbol, lstm_start_date=self.start_date)
        self.data = None
        self.trend_signals = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load SPY daily close price data for the last 6 months.
        
        Returns:
            DataFrame with SPY data including Close prices
        """
        print(f"📊 Loading {self.symbol} data from {self.start_date}...")
        
        # Fetch data using the existing data retriever
        self.data = self.data_retriever.fetch_data_for_period(self.start_date)
        
        # Ensure we have Close prices
        if 'Close' not in self.data.columns:
            raise ValueError("No Close price data available")
            
        print(f"✅ Loaded {len(self.data)} days of {self.symbol} data")
        print(f"   Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def calculate_moving_averages(self, data: pd.DataFrame, short_periods: List[int], 
                                long_periods: List[int]) -> pd.DataFrame:
        """
        Calculate moving averages for various periods.
        
        Args:
            data: DataFrame with Close prices
            short_periods: List of short MA periods to test
            long_periods: List of long MA periods to test
            
        Returns:
            DataFrame with calculated moving averages
        """
        result_data = data.copy()
        
        # Calculate all short MAs
        for period in short_periods:
            result_data[f'SMA_{period}'] = result_data['Close'].rolling(
                window=period, min_periods=period
            ).mean()
        
        # Calculate all long MAs
        for period in long_periods:
            result_data[f'SMA_{period}'] = result_data['Close'].rolling(
                window=period, min_periods=period
            ).mean()
        
        return result_data
    
    def identify_trend_signals(self, data: pd.DataFrame, short_periods: List[int], 
                             long_periods: List[int]) -> List[TrendSignal]:
        """
        Identify trend signals based on MA velocity changes.
        
        Args:
            data: DataFrame with Close prices and MAs
            short_periods: List of short MA periods
            long_periods: List of long MA periods
            
        Returns:
            List of TrendSignal objects
        """
        signals = []
        
        for short_ma in short_periods:
            for long_ma in long_periods:
                if short_ma >= long_ma:
                    continue
                    
                # Calculate MA velocity
                velocity_col = f'MA_Velocity_{short_ma}_{long_ma}'
                data[velocity_col] = data[f'SMA_{short_ma}'] / data[f'SMA_{long_ma}']
                
                # Calculate velocity changes to identify signals
                velocity_changes = data[velocity_col].diff()
                
                # Look for upward trend signals (velocity increases)
                for i in range(1, len(data) - 3):  # Need at least 3 days after signal
                    if velocity_changes.iloc[i] > 0:  # Velocity increased
                        # Check if this leads to an upward trend of at least 3 days
                        success, duration, trend_return = self._check_trend_success(
                            data, i, 'up', min_duration=3, max_duration=60
                        )
                        
                        signal = TrendSignal(
                            signal_date=data.index[i],
                            ma_velocity=data[velocity_col].iloc[i],
                            short_ma=short_ma,
                            long_ma=long_ma,
                            signal_type='up',
                            success=success,
                            trend_duration=duration,
                            trend_return=trend_return
                        )
                        signals.append(signal)
                
                # Look for downward trend signals (velocity decreases)
                for i in range(1, len(data) - 3):  # Need at least 3 days after signal
                    if velocity_changes.iloc[i] < 0:  # Velocity decreased
                        # Check if this leads to a downward trend of at least 3 days
                        success, duration, trend_return = self._check_trend_success(
                            data, i, 'down', min_duration=3, max_duration=60
                        )
                        
                        signal = TrendSignal(
                            signal_date=data.index[i],
                            ma_velocity=data[velocity_col].iloc[i],
                            short_ma=short_ma,
                            long_ma=long_ma,
                            signal_type='down',
                            success=success,
                            trend_duration=duration,
                            trend_return=trend_return
                        )
                        signals.append(signal)
        
        self.trend_signals = signals
        print(f"✅ Identified {len(signals)} trend signals")
        print(f"   Upward signals: {len([s for s in signals if s.signal_type == 'up'])}")
        print(f"   Downward signals: {len([s for s in signals if s.signal_type == 'down'])}")
        
        return signals
    
    def _check_trend_success(self, data: pd.DataFrame, signal_index: int, 
                           trend_type: str, min_duration: int = 3, 
                           max_duration: int = 60) -> Tuple[bool, int, float]:
        """
        Check if a trend signal leads to a successful trend.
        
        Args:
            data: DataFrame with price data
            signal_index: Index of the signal
            trend_type: 'up' or 'down'
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days
            
        Returns:
            Tuple of (success, duration, return)
        """
        start_price = data['Close'].iloc[signal_index]
        
        # Look for trend continuation
        for duration in range(min_duration, min(max_duration + 1, len(data) - signal_index)):
            end_index = signal_index + duration
            if end_index >= len(data):
                break
                
            end_price = data['Close'].iloc[end_index]
            total_return = (end_price - start_price) / start_price
            
            if trend_type == 'up':
                if total_return > 0:
                    # Check if this is a sustained upward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(signal_index + 1, end_index):
                        current_price = data['Close'].iloc[j]
                        current_return = (current_price - start_price) / start_price
                        if current_return < -0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break
                    
                    if trend_sustained:
                        return True, duration, total_return
            else:  # down trend
                if total_return < 0:
                    # Check if this is a sustained downward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(signal_index + 1, end_index):
                        current_price = data['Close'].iloc[j]
                        current_return = (current_price - start_price) / start_price
                        if current_return > 0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break
                    
                    if trend_sustained:
                        return True, duration, total_return
        
        return False, 0, 0.0
    
    def calculate_success_rates(self, signals: List[TrendSignal]) -> Dict[str, List[MAVelocityResult]]:
        """
        Calculate success rates for different MA combinations.
        
        Args:
            signals: List of trend signals
            
        Returns:
            Dictionary with results for upward and downward trends
        """
        results = {'up': [], 'down': []}
        
        # Group signals by MA combination and trend type
        signal_groups = {}
        for signal in signals:
            key = (signal.short_ma, signal.long_ma, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Calculate success rates for each combination
        for (short_ma, long_ma, trend_type), group_signals in signal_groups.items():
            if len(group_signals) < 3:  # Need at least 3 signals for meaningful analysis
                continue
            
            successful_signals = [s for s in group_signals if s.success]
            success_rate = len(successful_signals) / len(group_signals)
            
            if successful_signals:
                avg_duration = np.mean([s.trend_duration for s in successful_signals])
                avg_return = np.mean([s.trend_return for s in successful_signals])
            else:
                avg_duration = 0.0
                avg_return = 0.0
            
            result = MAVelocityResult(
                short_ma=short_ma,
                long_ma=long_ma,
                velocity=short_ma / long_ma,
                success_rate=success_rate,
                total_signals=len(group_signals),
                successful_signals=len(successful_signals),
                avg_trend_duration=avg_duration,
                avg_trend_return=avg_return
            )
            
            results[trend_type].append(result)
        
        return results
    
    def find_optimal_ma_combinations(self, short_periods: List[int] = None, 
                                   long_periods: List[int] = None) -> Dict[str, MAVelocityResult]:
        """
        Find the optimal MA combinations for upward and downward trends.
        
        Args:
            short_periods: List of short MA periods to test (default: [5, 10, 15, 20])
            long_periods: List of long MA periods to test (default: [30, 50, 100, 200])
            
        Returns:
            Dictionary with optimal MA combinations for each trend type
        """
        if short_periods is None:
            short_periods = [3, 5, 10, 15, 20]
        if long_periods is None:
            long_periods = [20, 25, 30, 50, 100, 200]
        
        # Load data if not already loaded
        if self.data is None:
            self.load_data()
        
        # Calculate moving averages
        data_with_ma = self.calculate_moving_averages(self.data, short_periods, long_periods)
        
        # Identify trend signals
        signals = self.identify_trend_signals(data_with_ma, short_periods, long_periods)
        
        # Calculate success rates
        results = self.calculate_success_rates(signals)
        
        # Find optimal combinations (highest success rate)
        optimal_combinations = {}
        
        for trend_type in ['up', 'down']:
            if results[trend_type]:
                # Sort by success rate (higher is better)
                sorted_results = sorted(results[trend_type], 
                                      key=lambda x: x.success_rate, 
                                      reverse=True)
                optimal_combinations[trend_type] = sorted_results[0]
        
        return optimal_combinations
    
    def generate_report(self, optimal_combinations: Dict[str, MAVelocityResult]) -> str:
        """
        Generate a comprehensive report of the analysis results.
        
        Args:
            optimal_combinations: Dictionary with optimal MA combinations
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("MOVING AVERAGE VELOCITY/ELASTICITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Symbol: {self.symbol}")
        report.append(f"Analysis Period: {self.start_date} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"Total Trading Days: {len(self.data)}")
        report.append(f"Total Trend Signals: {len(self.trend_signals)}")
        report.append("")
        
        # Summary of signals
        up_signals = [s for s in self.trend_signals if s.signal_type == 'up']
        down_signals = [s for s in self.trend_signals if s.signal_type == 'down']
        
        report.append("SIGNAL SUMMARY:")
        report.append(f"  Upward Trend Signals: {len(up_signals)}")
        report.append(f"  Downward Trend Signals: {len(down_signals)}")
        if up_signals:
            up_success_rate = len([s for s in up_signals if s.success]) / len(up_signals)
            report.append(f"  Upward Signal Success Rate: {up_success_rate:.1%}")
        if down_signals:
            down_success_rate = len([s for s in down_signals if s.success]) / len(down_signals)
            report.append(f"  Downward Signal Success Rate: {down_success_rate:.1%}")
        report.append("")
        
        # Longest trends analysis
        successful_up_signals = [s for s in up_signals if s.success]
        successful_down_signals = [s for s in down_signals if s.success]
        
        report.append("LONGEST TRENDS:")
        if successful_up_signals:
            longest_up_trend = max(successful_up_signals, key=lambda x: x.trend_duration)
            report.append(f"  Longest Upward Trend: {longest_up_trend.trend_duration} days")
            report.append(f"    Start Date: {longest_up_trend.signal_date.strftime('%Y-%m-%d')}")
            report.append(f"    Total Return: {longest_up_trend.trend_return:.2%}")
            report.append(f"    MA Combination: SMA {longest_up_trend.short_ma}/{longest_up_trend.long_ma}")
        else:
            report.append(f"  Longest Upward Trend: No successful upward trends found")
            
        if successful_down_signals:
            longest_down_trend = max(successful_down_signals, key=lambda x: x.trend_duration)
            report.append(f"  Longest Downward Trend: {longest_down_trend.trend_duration} days")
            report.append(f"    Start Date: {longest_down_trend.signal_date.strftime('%Y-%m-%d')}")
            report.append(f"    Total Return: {longest_down_trend.trend_return:.2%}")
            report.append(f"    MA Combination: SMA {longest_down_trend.short_ma}/{longest_down_trend.long_ma}")
        else:
            report.append(f"  Longest Downward Trend: No successful downward trends found")
        report.append("")
        
        # Optimal combinations
        report.append("OPTIMAL MOVING AVERAGE COMBINATIONS:")
        report.append("-" * 50)
        
        for trend_type, result in optimal_combinations.items():
            trend_name = "UPWARD" if trend_type == 'up' else "DOWNWARD"
            report.append(f"{trend_name} TREND SIGNALS:")
            report.append(f"  Short MA: {result.short_ma} days")
            report.append(f"  Long MA: {result.long_ma} days")
            report.append(f"  Velocity Ratio: {result.velocity:.3f}")
            report.append(f"  Success Rate: {result.success_rate:.1%}")
            report.append(f"  Total Signals: {result.total_signals}")
            report.append(f"  Successful Signals: {result.successful_signals}")
            report.append(f"  Average Trend Duration: {result.avg_trend_duration:.1f} days")
            report.append(f"  Average Trend Return: {result.avg_trend_return:.2%}")
            report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, optimal_combinations: Dict[str, MAVelocityResult], 
                    save_path: Optional[str] = None):
        """
        Create visualization plots for the analysis results.
        
        Args:
            optimal_combinations: Dictionary with optimal MA combinations
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Moving Average Velocity Analysis - {self.symbol}', fontsize=16)
        
        # Plot 1: Price and optimal MAs for upward trends
        if 'up' in optimal_combinations:
            ax1 = axes[0, 0]
            up_result = optimal_combinations['up']
            ax1.plot(self.data.index, self.data['Close'], label=f'{self.symbol} Close', alpha=0.7)
            
            # Check if MA columns exist, if not calculate them
            short_ma_col = f'SMA_{up_result.short_ma}'
            long_ma_col = f'SMA_{up_result.long_ma}'
            
            if short_ma_col not in self.data.columns:
                self.data[short_ma_col] = self.data['Close'].rolling(window=up_result.short_ma, min_periods=up_result.short_ma).mean()
            if long_ma_col not in self.data.columns:
                self.data[long_ma_col] = self.data['Close'].rolling(window=up_result.long_ma, min_periods=up_result.long_ma).mean()
            
            ax1.plot(self.data.index, self.data[short_ma_col], 
                    label=f'SMA {up_result.short_ma}', alpha=0.8)
            ax1.plot(self.data.index, self.data[long_ma_col], 
                    label=f'SMA {up_result.long_ma}', alpha=0.8)
            ax1.set_title(f'Optimal MAs for Upward Trends\nSuccess Rate: {up_result.success_rate:.1%}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price and optimal MAs for downward trends
        if 'down' in optimal_combinations:
            ax2 = axes[0, 1]
            down_result = optimal_combinations['down']
            ax2.plot(self.data.index, self.data['Close'], label=f'{self.symbol} Close', alpha=0.7)
            
            # Check if MA columns exist, if not calculate them
            short_ma_col = f'SMA_{down_result.short_ma}'
            long_ma_col = f'SMA_{down_result.long_ma}'
            
            if short_ma_col not in self.data.columns:
                self.data[short_ma_col] = self.data['Close'].rolling(window=down_result.short_ma, min_periods=down_result.short_ma).mean()
            if long_ma_col not in self.data.columns:
                self.data[long_ma_col] = self.data['Close'].rolling(window=down_result.long_ma, min_periods=down_result.long_ma).mean()
            
            ax2.plot(self.data.index, self.data[short_ma_col], 
                    label=f'SMA {down_result.short_ma}', alpha=0.8)
            ax2.plot(self.data.index, self.data[long_ma_col], 
                    label=f'SMA {down_result.long_ma}', alpha=0.8)
            ax2.set_title(f'Optimal MAs for Downward Trends\nSuccess Rate: {down_result.success_rate:.1%}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Signal points on price chart
        ax3 = axes[1, 0]
        ax3.plot(self.data.index, self.data['Close'], label=f'{self.symbol} Close', alpha=0.7)
        
        # Plot successful signals
        successful_signals = [s for s in self.trend_signals if s.success]
        for signal in successful_signals:
            color = 'green' if signal.signal_type == 'up' else 'red'
            ax3.scatter(signal.signal_date, self.data.loc[signal.signal_date, 'Close'], 
                       color=color, s=50, alpha=0.7)
        
        ax3.set_title('Successful Trend Signals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Failed Trend Signals
        ax4 = axes[1, 1]
        ax4.plot(self.data.index, self.data['Close'], label=f'{self.symbol} Close', alpha=0.7)
        
        # Plot failed signals
        failed_signals = [s for s in self.trend_signals if not s.success]
        for signal in failed_signals:
            color = 'green' if signal.signal_type == 'up' else 'red'
            ax4.scatter(signal.signal_date, self.data.loc[signal.signal_date, 'Close'], 
                       color=color, s=50, alpha=0.7)
        
        ax4.set_title('Failed Trend Signals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to run the MA velocity analysis.
    """
    print("🚀 Starting Moving Average Velocity Analysis...")
    
    # Initialize analyzer (will use last 6 months by default)
    analyzer = MAVelocityAnalyzer(symbol='SPY')
    
    # Define MA periods to test
    short_periods = [5, 10, 15, 20, 25]
    long_periods = [30, 50, 100, 150, 200]
    
    # Find optimal combinations
    optimal_combinations = analyzer.find_optimal_ma_combinations(short_periods, long_periods)
    
    # Generate and print report
    report = analyzer.generate_report(optimal_combinations)
    print(report)
    
    # Create plots
    analyzer.plot_results(optimal_combinations, save_path='ma_velocity_analysis.png')
    
    print("\n✅ Analysis complete!")
    
    return optimal_combinations


if __name__ == "__main__":
    main()
