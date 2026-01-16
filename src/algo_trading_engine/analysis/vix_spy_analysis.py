"""
VIX Big Move and SPY Return Analysis

This module analyzes SPY returns following big moves in VIX.
A "big move" is defined as a VIX log return greater than 2 standard deviations
from its mean over 1-3 days.

The analysis calculates:
- Average SPY returns for 3, 6, and 12 days following big VIX moves
- Frequency distribution of SPY log returns (rounded to nearest 0.2%)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import data retrieval classes
from algo_trading_engine.common.data_retriever import DataRetriever


@dataclass
class BigMoveEvent:
    """Represents a big move event in VIX"""
    date: pd.Timestamp
    vix_log_return: float
    days_in_move: int  # 1, 2, or 3 days
    vix_std_devs: float  # How many std deviations above mean
    spy_return_3d: float
    spy_return_6d: float
    spy_return_12d: float


@dataclass
class ReturnStats:
    """Statistics for SPY returns following big VIX moves"""
    days_forward: int
    average_return: float
    median_return: float
    total_events: int
    frequency_distribution: Dict[float, int]  # Return bucket -> count


class VIXSPYAnalyzer:
    """
    Analyzes SPY returns following big moves in VIX.
    
    Big move definition: VIX log return > 2 standard deviations from mean
    over 1-3 day periods.
    """
    
    def __init__(self, lookback_years: int = 5):
        """
        Initialize the VIX-SPY Analyzer.
        
        Args:
            lookback_years: Number of years of historical data to analyze (default: 5)
        """
        # Calculate start date
        today = datetime.now()
        start_date = today - timedelta(days=lookback_years * 365)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.lookback_years = lookback_years
        
        # Data retrievers for VIX and SPY
        self.vix_retriever = DataRetriever(symbol='^VIX', lstm_start_date=self.start_date)
        self.spy_retriever = DataRetriever(symbol='SPY', lstm_start_date=self.start_date)
        
        self.vix_data = None
        self.spy_data = None
        self.big_move_events = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load VIX and SPY daily close price data.
        
        Returns:
            Tuple of (VIX DataFrame, SPY DataFrame)
        """
        print(f"Loading VIX and SPY data from {self.start_date}...")
        
        # Fetch VIX data
        self.vix_data = self.vix_retriever.fetch_data_for_period(self.start_date, 'vix_analysis')
        if 'Close' not in self.vix_data.columns:
            raise ValueError("No Close price data available for VIX")
        
        # Fetch SPY data
        self.spy_data = self.spy_retriever.fetch_data_for_period(self.start_date, 'vix_analysis')
        if 'Close' not in self.spy_data.columns:
            raise ValueError("No Close price data available for SPY")
        
        # Align data on common dates
        common_dates = self.vix_data.index.intersection(self.spy_data.index)
        self.vix_data = self.vix_data.loc[common_dates]
        self.spy_data = self.spy_data.loc[common_dates]
        
        print(f"Loaded {len(self.vix_data)} days of aligned VIX and SPY data")
        print(f"   Date range: {self.vix_data.index[0]} to {self.vix_data.index[-1]}")
        
        return self.vix_data, self.spy_data
    
    def calculate_log_returns(self, data: pd.DataFrame, periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculate log returns for specified periods.
        
        Args:
            data: DataFrame with Close prices
            periods: List of periods for log return calculation
            
        Returns:
            DataFrame with log returns for each period
        """
        result = data.copy()
        
        for period in periods:
            # Log return = ln(Price_t / Price_{t-period})
            result[f'log_return_{period}d'] = np.log(result['Close'] / result['Close'].shift(period))
        
        return result
    
    def identify_big_moves(self, vix_data: pd.DataFrame, std_threshold: float = 2.0) -> List[Tuple[pd.Timestamp, int, float, float]]:
        """
        Identify big moves in VIX (log returns > std_threshold standard deviations).
        
        Args:
            vix_data: DataFrame with VIX data and log returns
            std_threshold: Number of standard deviations to define a big move
            
        Returns:
            List of tuples: (date, period, log_return, std_devs_above_mean)
        """
        big_moves = []
        
        for period in [1, 2, 3]:
            col = f'log_return_{period}d'
            if col not in vix_data.columns:
                continue
            
            # Calculate mean and std for this period's log returns
            log_returns = vix_data[col].dropna()
            mean_return = log_returns.mean()
            std_return = log_returns.std()
            
            print(f"\nVIX {period}-day log return statistics:")
            print(f"   Mean: {mean_return:.6f}")
            print(f"   Std Dev: {std_return:.6f}")
            print(f"   Threshold (mean + {std_threshold} * std): {mean_return + std_threshold * std_return:.6f}")
            
            # Identify dates where log return exceeds threshold
            threshold = mean_return + std_threshold * std_return
            big_move_mask = vix_data[col] > threshold
            big_move_dates = vix_data[big_move_mask].index
            
            print(f"   Found {len(big_move_dates)} big moves")
            
            for date in big_move_dates:
                log_return = vix_data.loc[date, col]
                std_devs = (log_return - mean_return) / std_return
                big_moves.append((date, period, log_return, std_devs))
        
        print(f"\nTotal big moves identified: {len(big_moves)}")
        return big_moves
    
    def calculate_forward_spy_returns(self, date: pd.Timestamp, forward_days: List[int] = [3, 6, 12]) -> Dict[int, float]:
        """
        Calculate SPY log returns for specified forward periods from a given date.
        
        Args:
            date: Starting date
            forward_days: List of forward periods to calculate returns
            
        Returns:
            Dictionary mapping forward days to log returns (None if data not available)
        """
        forward_returns = {}
        
        try:
            start_price = self.spy_data.loc[date, 'Close']
            start_idx = self.spy_data.index.get_loc(date)
        except (KeyError, IndexError):
            # Date not in data
            return {days: None for days in forward_days}
        
        for days in forward_days:
            target_idx = start_idx + days
            
            if target_idx < len(self.spy_data):
                target_date = self.spy_data.index[target_idx]
                end_price = self.spy_data.loc[target_date, 'Close']
                log_return = np.log(end_price / start_price)
                forward_returns[days] = log_return
            else:
                forward_returns[days] = None
        
        return forward_returns
    
    def analyze_big_moves(self) -> List[BigMoveEvent]:
        """
        Analyze all big moves and calculate corresponding SPY returns.
        
        Returns:
            List of BigMoveEvent objects
        """
        # Calculate log returns for VIX
        self.vix_data = self.calculate_log_returns(self.vix_data, periods=[1, 2, 3])
        
        # Identify big moves
        big_moves = self.identify_big_moves(self.vix_data)
        
        print(f"\nCalculating SPY returns for {len(big_moves)} big move events...")
        
        # Calculate SPY forward returns for each big move
        events = []
        for date, period, vix_log_return, std_devs in big_moves:
            forward_returns = self.calculate_forward_spy_returns(date, forward_days=[3, 6, 12])
            
            # Only create event if we have all forward returns
            if all(r is not None for r in forward_returns.values()):
                event = BigMoveEvent(
                    date=date,
                    vix_log_return=vix_log_return,
                    days_in_move=period,
                    vix_std_devs=std_devs,
                    spy_return_3d=forward_returns[3],
                    spy_return_6d=forward_returns[6],
                    spy_return_12d=forward_returns[12]
                )
                events.append(event)
        
        self.big_move_events = events
        print(f"Analyzed {len(events)} complete big move events")
        
        return events
    
    def round_to_nearest(self, value: float, increment: float = 0.002) -> float:
        """
        Round a value to the nearest increment (default 0.2% = 0.002).
        
        Args:
            value: Value to round
            increment: Rounding increment
            
        Returns:
            Rounded value
        """
        return round(value / increment) * increment
    
    def calculate_return_statistics(self, events: List[BigMoveEvent]) -> Dict[int, ReturnStats]:
        """
        Calculate statistics for SPY returns at different forward periods.
        
        Args:
            events: List of BigMoveEvent objects
            
        Returns:
            Dictionary mapping forward days to ReturnStats
        """
        stats = {}
        
        for days in [3, 6, 12]:
            # Extract returns for this period
            if days == 3:
                returns = [e.spy_return_3d for e in events]
            elif days == 6:
                returns = [e.spy_return_6d for e in events]
            else:  # 12
                returns = [e.spy_return_12d for e in events]
            
            # Calculate statistics
            avg_return = np.mean(returns)
            median_return = np.median(returns)
            
            # Round returns to nearest 0.2% and count frequencies
            rounded_returns = [self.round_to_nearest(r, 0.002) for r in returns]
            frequency_dist = Counter(rounded_returns)
            
            # Sort frequency distribution
            frequency_dist = dict(sorted(frequency_dist.items()))
            
            stats[days] = ReturnStats(
                days_forward=days,
                average_return=avg_return,
                median_return=median_return,
                total_events=len(returns),
                frequency_distribution=frequency_dist
            )
        
        return stats
    
    def plot_results(self, stats: Dict[int, ReturnStats], save_path: str = None):
        """
        Create comprehensive visualizations of the VIX-SPY analysis results.
        
        Args:
            stats: Dictionary of ReturnStats for each forward period
            save_path: Optional path to save the plot (if None, will display)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Plot 1: Average Returns Bar Chart
        ax_avg = fig.add_subplot(gs[0, :])
        days_list = [3, 6, 12]
        avg_returns = [stats[d].average_return * 100 for d in days_list]
        median_returns = [stats[d].median_return * 100 for d in days_list]
        
        x = np.arange(len(days_list))
        width = 0.35
        
        bars1 = ax_avg.bar(x - width/2, avg_returns, width, label='Average', color=colors, alpha=0.8)
        bars2 = ax_avg.bar(x + width/2, median_returns, width, label='Median', 
                          color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
        
        ax_avg.set_xlabel('Days After Big VIX Move', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('SPY Return (%)', fontsize=12, fontweight='bold')
        ax_avg.set_title('Average SPY Returns Following Big VIX Moves\n(VIX log return > 2 std deviations)', 
                        fontsize=14, fontweight='bold', pad=20)
        ax_avg.set_xticks(x)
        ax_avg.set_xticklabels([f'{d} Days' for d in days_list])
        ax_avg.legend(fontsize=10)
        ax_avg.grid(axis='y', alpha=0.3, linestyle='--')
        ax_avg.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_avg.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}%',
                          ha='center', va='bottom' if height >= 0 else 'top',
                          fontsize=9, fontweight='bold')
        
        # Plots 2-4: Frequency Distribution Histograms
        for idx, days in enumerate([3, 6, 12]):
            ax = fig.add_subplot(gs[1, idx])
            stat = stats[days]
            
            # Convert to percentage and prepare data
            returns_pct = np.array(list(stat.frequency_distribution.keys())) * 100
            frequencies = np.array(list(stat.frequency_distribution.values()))
            
            # Create histogram
            bars = ax.bar(returns_pct, frequencies, width=0.18, color=colors[idx], 
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add vertical line for mean and median
            ax.axvline(stat.average_return * 100, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {stat.average_return*100:.2f}%')
            ax.axvline(stat.median_return * 100, color='blue', linestyle='--', 
                      linewidth=2, label=f'Median: {stat.median_return*100:.2f}%')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('SPY Return (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency (Count)', fontsize=10, fontweight='bold')
            ax.set_title(f'{days}-Day Forward Returns\n(n={stat.total_events} events)', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Overall title
        fig.suptitle('VIX Big Move → SPY Return Analysis\n' + 
                    f'Analysis Period: {self.start_date} to {self.vix_data.index[-1].strftime("%Y-%m-%d")}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()
    
    def print_results(self, stats: Dict[int, ReturnStats]):
        """
        Print analysis results in a readable format.
        
        Args:
            stats: Dictionary of ReturnStats for each forward period
        """
        print("\n" + "="*80)
        print("VIX BIG MOVE -> SPY RETURN ANALYSIS")
        print("="*80)
        print(f"\nAnalysis Period: {self.start_date} to {self.vix_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Big Move Definition: VIX log return > 2 std deviations from mean (1-3 days)")
        print(f"Total Big Move Events Analyzed: {len(self.big_move_events)}")
        
        for days in [3, 6, 12]:
            stat = stats[days]
            print("\n" + "-"*80)
            print(f"SPY RETURNS {days} DAYS AFTER BIG VIX MOVE")
            print("-"*80)
            print(f"Average Return: {stat.average_return:.4f} ({stat.average_return*100:.2f}%)")
            print(f"Median Return:  {stat.median_return:.4f} ({stat.median_return*100:.2f}%)")
            print(f"Total Events:   {stat.total_events}")
            
            print(f"\nFrequency Distribution (rounded to nearest 0.2%):")
            print(f"{'Return Bucket':<20} {'Count':<10} {'Frequency %':<15}")
            print("-"*50)
            
            for return_bucket, count in stat.frequency_distribution.items():
                frequency_pct = (count / stat.total_events) * 100
                return_pct = return_bucket * 100
                print(f"{return_pct:>6.1f}%{'':<13} {count:<10} {frequency_pct:>6.2f}%")
        
        print("\n" + "="*80)
    
    def run_analysis(self, create_plot: bool = True, save_plot_path: str = None):
        """
        Run the complete VIX-SPY analysis pipeline.
        
        Args:
            create_plot: Whether to create visualizations (default: True)
            save_plot_path: Optional path to save the plot (if None, will display)
        """
        print("Starting VIX Big Move -> SPY Return Analysis\n")
        
        # Load data
        self.load_data()
        
        # Analyze big moves and calculate SPY returns
        events = self.analyze_big_moves()
        
        if not events:
            print("No big move events found with complete data")
            return
        
        # Calculate statistics
        stats = self.calculate_return_statistics(events)
        
        # Print results
        self.print_results(stats)
        
        # Create visualizations
        if create_plot:
            print("\nGenerating visualizations...")
            self.plot_results(stats, save_path=save_plot_path)
        
        print("\nAnalysis complete!")
        
        return stats


def main(create_plot: bool = True, save_plot: bool = True):
    """
    Main entry point for VIX-SPY analysis.
    
    Args:
        create_plot: Whether to create visualizations (default: True)
        save_plot: Whether to save the plot to file (default: True)
    """
    analyzer = VIXSPYAnalyzer(lookback_years=5)
    
    # Determine save path
    save_path = None
    if create_plot and save_plot:
        save_path = 'predictions/vix_spy_analysis.png'
        print(f"\nPlot will be saved to: {save_path}")
    
    stats = analyzer.run_analysis(create_plot=create_plot, save_plot_path=save_path)
    return analyzer, stats


if __name__ == '__main__':
    analyzer, stats = main()
