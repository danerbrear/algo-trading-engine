"""
Bullish Market Regime Drawdown Analysis

This module analyzes drawdowns that occur during upward trends in SPY,
categorized by bullish market regimes identified by HMM.

The analysis answers: What bullish market regime experiences the most 
amount of drawdowns following an upward trend?

An upward trend is defined as 3-10 consecutive days of positive returns.
A drawdown is a decline in price from a peak during the trend period.
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
from common.models import MarketStateType


@dataclass(frozen=True)
class UpwardTrendWithRegime:
    """Represents an upward trend with associated market regime"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int  # 3-10 days
    start_price: float
    end_price: float
    total_return: float
    trend_ending_drawdown_pct: float  # Drawdown when trend ends
    trend_ending_price: float  # Price when trend ends (first negative day)
    market_regime: MarketStateType  # HMM-identified regime
    regime_confidence: float  # HMM confidence for this regime


@dataclass(frozen=True)
class RegimeDrawdownStatistics:
    """Statistical summary of drawdowns by market regime"""
    regime: MarketStateType
    total_trends: int
    min_drawdown: float
    mean_drawdown: float
    median_drawdown: float
    max_drawdown: float
    std_drawdown: float
    drawdowns_by_duration: Dict[int, float]  # Average drawdown by trend duration


@dataclass(frozen=True)
class BullishRegimeDrawdownAnalysisResult:
    """Complete results of the bullish regime drawdown analysis"""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    total_trends_analyzed: int
    trends_by_regime: Dict[MarketStateType, List[UpwardTrendWithRegime]]
    regime_statistics: Dict[MarketStateType, RegimeDrawdownStatistics]
    overall_average_drawdown: float
    regime_with_most_trends: MarketStateType
    regime_with_highest_mean_drawdown: MarketStateType


class BullishRegimeDrawdownAnalyzer:
    """Analyzes drawdowns during upward trends by bullish market regime"""
    
    def __init__(self, symbol: str = 'SPY', analysis_period_months: int = 12):
        self.symbol = symbol
        self.analysis_period_months = analysis_period_months
        self.hmm_training_months = 24  # 2 years for HMM training
        self.data_retriever = DataRetriever(symbol=symbol, use_free_tier=True, quiet_mode=True)
        self.hmm_model = None
        self.hmm_training_data = None
        self.analysis_data = None
    
    def load_data_and_train_hmm(self) -> pd.DataFrame:
        """Load data and train HMM model for market regime classification"""
        print(f"\n📊 Loading SPY data for HMM training ({self.hmm_training_months} months) and analysis ({self.analysis_period_months} months)...")
        
        # Calculate dates
        end_date = datetime.now()
        analysis_start_date = end_date - timedelta(days=self.analysis_period_months * 30)
        hmm_training_start_date = analysis_start_date - timedelta(days=self.hmm_training_months * 30)
        
        # Step 1: Load HMM training data (2 years before analysis period)
        print(f"\n🎯 Phase 1: Loading HMM training data from {hmm_training_start_date.strftime('%Y-%m-%d')} to {analysis_start_date.strftime('%Y-%m-%d')}")
        self.hmm_training_data = self.data_retriever.fetch_data_for_period(
            hmm_training_start_date.strftime('%Y-%m-%d'), 
            'hmm'
        )
        
        # Calculate features for HMM training data
        self.data_retriever.calculate_features_for_data(self.hmm_training_data)
        
        # Step 2: Train HMM model on training data
        print(f"\n🎯 Phase 2: Training HMM model on {len(self.hmm_training_data)} training samples...")
        from src.model.market_state_classifier import MarketStateClassifier
        self.hmm_model = MarketStateClassifier()
        self.hmm_model.train_hmm_model(self.hmm_training_data)
        
        # Step 3: Load analysis data (specified period)
        print(f"\n📊 Phase 3: Loading analysis data from {analysis_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        self.analysis_data = self.data_retriever.fetch_data_for_period(
            analysis_start_date.strftime('%Y-%m-%d'), 
            'hmm'
        )
        
        # Calculate features for analysis data
        self.data_retriever.calculate_features_for_data(self.analysis_data)
        
        # Step 4: Apply trained HMM to analysis data (not training data)
        print(f"\n🔮 Phase 4: Applying trained HMM to analysis data...")
        self.analysis_data['Market_State'] = self.hmm_model.predict_states(self.analysis_data)
        
        return self.analysis_data
    
    def identify_upward_trends_with_regimes(self, data: pd.DataFrame) -> List[UpwardTrendWithRegime]:
        """Identify upward trends and classify their market regimes"""
        print("\n🔍 Identifying upward trends and classifying market regimes...")
        
        trends = []
        i = 1  # Start at 1 because returns start at index 1
        
        while i < len(data) - 2:  # Need at least 3 days for minimum trend
            # Check if current day has positive return
            if data['Returns'].iloc[i] > 0:
                trend_start_idx = i
                consecutive_positive_days = 0
                
                # Count consecutive positive returns
                j = i
                while j < len(data) and data['Returns'].iloc[j] > 0:
                    consecutive_positive_days += 1
                    j += 1
                    
                    # Cap at 10 days max
                    if consecutive_positive_days >= 10:
                        break
                
                # If trend is between 3-10 days, analyze it
                if 3 <= consecutive_positive_days <= 10:
                    trend_end_idx = trend_start_idx + consecutive_positive_days - 1
                    
                    # Calculate drawdown that occurs when trend ends
                    trend_ending_drawdown_pct, trend_ending_price = self.calculate_trend_ending_drawdown(
                        data, trend_end_idx
                    )
                    
                    # Get market regime for this trend (use middle of trend)
                    middle_idx = trend_start_idx + (consecutive_positive_days // 2)
                    market_state_id = data['Market_State'].iloc[middle_idx]
                    market_regime = self._map_state_id_to_regime(market_state_id)
                    
                    # Only include bullish regimes
                    if market_regime in [MarketStateType.LOW_VOLATILITY_UPTREND, 
                                       MarketStateType.MOMENTUM_UPTREND, 
                                       MarketStateType.HIGH_VOLATILITY_RALLY]:
                        
                        trend = UpwardTrendWithRegime(
                            start_date=data.index[trend_start_idx],
                            end_date=data.index[trend_end_idx],
                            duration=consecutive_positive_days,
                            start_price=data['Close'].iloc[trend_start_idx],
                            end_price=data['Close'].iloc[trend_end_idx],
                            total_return=((data['Close'].iloc[trend_end_idx] / data['Close'].iloc[trend_start_idx]) - 1) * 100,
                            trend_ending_drawdown_pct=trend_ending_drawdown_pct,
                            trend_ending_price=trend_ending_price,
                            market_regime=market_regime,
                            regime_confidence=0.8  # Placeholder - could be enhanced with actual confidence
                        )
                        trends.append(trend)
                
                # Move index past this trend
                i = j
            else:
                i += 1
        
        print(f"✅ Identified {len(trends)} upward trends in bullish regimes")
        return trends
    
    def calculate_trend_ending_drawdown(self, data: pd.DataFrame, trend_end_idx: int) -> Tuple[float, float]:
        """Calculate drawdown that occurs when the upward trend ends"""
        # Get the last day of the upward trend (last positive return day)
        trend_end_price = data['Close'].iloc[trend_end_idx]
        
        # Get the first day after the trend ends (first negative return day)
        if trend_end_idx + 1 < len(data):
            trend_ending_price = data['Close'].iloc[trend_end_idx + 1]
            # Calculate drawdown from trend end to trend ending
            drawdown_pct = ((trend_end_price - trend_ending_price) / trend_end_price) * 100
        else:
            # No data after trend end
            trend_ending_price = trend_end_price
            drawdown_pct = 0.0
        
        return drawdown_pct, trend_ending_price
    
    def _map_state_id_to_regime(self, state_id: int) -> MarketStateType:
        """Map HMM state ID to market regime type"""
        # This mapping should be based on the actual HMM model characteristics
        # For now, using a simple mapping - this could be enhanced with actual regime analysis
        regime_mapping = {
            0: MarketStateType.LOW_VOLATILITY_UPTREND,
            1: MarketStateType.MOMENTUM_UPTREND,
            2: MarketStateType.CONSOLIDATION,
            3: MarketStateType.HIGH_VOLATILITY_DOWNTREND,
            4: MarketStateType.HIGH_VOLATILITY_RALLY,
        }
        
        # Handle cases where we have more or fewer states
        if state_id in regime_mapping:
            return regime_mapping[state_id]
        else:
            # Default to consolidation for unknown states
            return MarketStateType.CONSOLIDATION
    
    def calculate_regime_statistics(self, trends: List[UpwardTrendWithRegime]) -> Dict[MarketStateType, RegimeDrawdownStatistics]:
        """Calculate drawdown statistics grouped by market regime"""
        print("\n📊 Calculating regime-specific drawdown statistics...")
        
        # Group trends by regime
        trends_by_regime = {}
        for trend in trends:
            if trend.market_regime not in trends_by_regime:
                trends_by_regime[trend.market_regime] = []
            trends_by_regime[trend.market_regime].append(trend)
        
        regime_stats = {}
        for regime, regime_trends in trends_by_regime.items():
            if not regime_trends:
                continue
                
            # Calculate statistics for this regime
            drawdowns = [t.trend_ending_drawdown_pct for t in regime_trends]
            
            # Calculate drawdowns by duration
            drawdowns_by_duration = {}
            for duration in range(3, 11):  # 3-10 days
                duration_trends = [t for t in regime_trends if t.duration == duration]
                if duration_trends:
                    drawdowns_by_duration[duration] = np.mean([t.trend_ending_drawdown_pct for t in duration_trends])
            
            stats = RegimeDrawdownStatistics(
                regime=regime,
                total_trends=len(regime_trends),
                min_drawdown=min(drawdowns),
                mean_drawdown=np.mean(drawdowns),
                median_drawdown=np.median(drawdowns),
                max_drawdown=max(drawdowns),
                std_drawdown=np.std(drawdowns),
                drawdowns_by_duration=drawdowns_by_duration
            )
            regime_stats[regime] = stats
        
        return regime_stats
    
    def run_analysis(self) -> BullishRegimeDrawdownAnalysisResult:
        """Execute the complete analysis workflow"""
        # Load data and train HMM (separate training and analysis periods)
        analysis_data = self.load_data_and_train_hmm()
        
        # Identify upward trends with regime classification
        trends = self.identify_upward_trends_with_regimes(analysis_data)
        
        # Calculate regime-specific statistics
        regime_stats = self.calculate_regime_statistics(trends)
        
        # Group trends by regime
        trends_by_regime = {}
        for trend in trends:
            if trend.market_regime not in trends_by_regime:
                trends_by_regime[trend.market_regime] = []
            trends_by_regime[trend.market_regime].append(trend)
        
        # Calculate overall statistics
        overall_avg_drawdown = np.mean([t.trend_ending_drawdown_pct for t in trends]) if trends else 0.0
        regime_with_most_trends = max(regime_stats.keys(), 
                                    key=lambda r: regime_stats[r].total_trends) if regime_stats else MarketStateType.CONSOLIDATION
        regime_with_highest_mean_drawdown = max(regime_stats.keys(), 
                                              key=lambda r: regime_stats[r].mean_drawdown) if regime_stats else MarketStateType.CONSOLIDATION
        
        # Create result object
        result = BullishRegimeDrawdownAnalysisResult(
            analysis_period_start=analysis_data.index[0].strftime('%Y-%m-%d'),
            analysis_period_end=analysis_data.index[-1].strftime('%Y-%m-%d'),
            total_trading_days=len(analysis_data),
            total_trends_analyzed=len(trends),
            trends_by_regime=trends_by_regime,
            regime_statistics=regime_stats,
            overall_average_drawdown=overall_avg_drawdown,
            regime_with_most_trends=regime_with_most_trends,
            regime_with_highest_mean_drawdown=regime_with_highest_mean_drawdown
        )
        
        return result
    
    def generate_report(self, result: BullishRegimeDrawdownAnalysisResult) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("BULLISH MARKET REGIME DRAWDOWN ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Period: {result.analysis_period_start} to {result.analysis_period_end}")
        report.append(f"Total Trading Days: {result.total_trading_days:,}")
        report.append(f"Total Upward Trends Analyzed: {result.total_trends_analyzed:,}")
        report.append("")
        
        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        report.append(f"Average Drawdown Across All Trends: {result.overall_average_drawdown:.2f}%")
        report.append(f"Regime with Most Trends: {result.regime_with_most_trends.value}")
        report.append(f"Regime with Highest Mean Drawdown: {result.regime_with_highest_mean_drawdown.value}")
        report.append("")
        
        # Regime-specific analysis
        report.append("REGIME-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        
        for regime, stats in result.regime_statistics.items():
            report.append(f"\n{regime.value.upper().replace('_', ' ')}")
            report.append(f"  Total Trends: {stats.total_trends:,}")
            report.append(f"  Min Drawdown: {stats.min_drawdown:.2f}%")
            report.append(f"  Mean Drawdown: {stats.mean_drawdown:.2f}%")
            report.append(f"  Median Drawdown: {stats.median_drawdown:.2f}%")
            report.append(f"  Max Drawdown: {stats.max_drawdown:.2f}%")
            report.append(f"  Std Dev Drawdown: {stats.std_drawdown:.2f}%")
        
        return "\n".join(report)
    
    def plot_results(self, result: BullishRegimeDrawdownAnalysisResult, save_path: str = None):
        """Create visualization plots for the analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bullish Market Regime Drawdown Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price with Trend Periods and Drawdowns
        self._plot_price_with_trends_and_drawdowns(axes[0, 0], result)
        
        # Plot 2: Drawdown Distribution by Regime
        self._plot_drawdown_distribution_by_regime(axes[0, 1], result)
        
        # Plot 3: Number of Trends by Regime
        self._plot_trends_by_regime(axes[1, 0], result)
        
        # Plot 4: Average Drawdown by Trend Duration and Regime
        self._plot_drawdown_by_duration_and_regime(axes[1, 1], result)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_price_with_trends_and_drawdowns(self, ax, result: BullishRegimeDrawdownAnalysisResult):
        """Plot SPY price with trend periods and drawdowns highlighted"""
        # This would need access to the actual price data
        # For now, create a placeholder plot
        ax.set_title('SPY Price with Upward Trends and Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.text(0.5, 0.5, 'Price chart with trends and drawdowns\n(Implementation requires price data access)', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_distribution_by_regime(self, ax, result: BullishRegimeDrawdownAnalysisResult):
        """Plot drawdown distribution by regime"""
        regimes = []
        drawdowns = []
        
        for regime, trends in result.trends_by_regime.items():
            regime_drawdowns = [t.trend_ending_drawdown_pct for t in trends]
            regimes.extend([regime.value.replace('_', ' ').title()] * len(regime_drawdowns))
            drawdowns.extend(regime_drawdowns)
        
        if regimes and drawdowns:
            # Create box plot
            regime_data = {}
            for regime, trend_list in result.trends_by_regime.items():
                regime_data[regime.value.replace('_', ' ').title()] = [t.trend_ending_drawdown_pct for t in trend_list]
            
            ax.boxplot(regime_data.values(), labels=regime_data.keys())
            ax.set_title('Drawdown Distribution by Market Regime')
            ax.set_ylabel('Drawdown (%)')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No data available for drawdown distribution', 
                    transform=ax.transAxes, ha='center', va='center')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_trends_by_regime(self, ax, result: BullishRegimeDrawdownAnalysisResult):
        """Plot number of trends by regime"""
        regimes = []
        trend_counts = []
        
        for regime, stats in result.regime_statistics.items():
            regimes.append(regime.value.replace('_', ' ').title())
            trend_counts.append(stats.total_trends)
        
        if regimes and trend_counts:
            bars = ax.bar(regimes, trend_counts, color=['#2E8B57', '#4169E1', '#DC143C'])
            ax.set_title('Number of Trends by Market Regime')
            ax.set_ylabel('Number of Trends')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, trend_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{count}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No data available for trend analysis', 
                    transform=ax.transAxes, ha='center', va='center')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_by_duration_and_regime(self, ax, result: BullishRegimeDrawdownAnalysisResult):
        """Plot average drawdown by trend duration and regime"""
        durations = list(range(3, 11))  # 3-10 days
        regime_colors = {
            MarketStateType.LOW_VOLATILITY_UPTREND: '#2E8B57',
            MarketStateType.MOMENTUM_UPTREND: '#4169E1',
            MarketStateType.HIGH_VOLATILITY_RALLY: '#DC143C'
        }
        
        for regime, stats in result.regime_statistics.items():
            regime_drawdowns = []
            for duration in durations:
                if duration in stats.drawdowns_by_duration:
                    regime_drawdowns.append(stats.drawdowns_by_duration[duration])
                else:
                    regime_drawdowns.append(0)
            
            if any(regime_drawdowns):
                ax.plot(durations, regime_drawdowns, 
                       marker='o', linewidth=2, markersize=6,
                       color=regime_colors.get(regime, '#000000'),
                       label=regime.value.replace('_', ' ').title())
        
        ax.set_title('Average Drawdown by Trend Duration and Regime')
        ax.set_xlabel('Trend Duration (days)')
        ax.set_ylabel('Average Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    """Main function for testing the analyzer"""
    analyzer = BullishRegimeDrawdownAnalyzer(symbol='SPY', analysis_period_months=12)
    result = analyzer.run_analysis()
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print("\n" + report)
    
    # Create visualization
    analyzer.plot_results(result)


if __name__ == "__main__":
    main()
