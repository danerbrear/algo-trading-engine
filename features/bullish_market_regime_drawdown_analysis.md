# Bullish Market Regime Drawdown Analysis

## Feature Description

This analysis answers the question: **What bullish market regime experiences the most amount of drawdowns that end upward trends?**

The analysis identifies upward trends in SPY (3-10 consecutive days of positive returns) and analyzes the drawdowns that occur when these trends end, categorized by the bullish market regime identified by the HMM model.

## Overview

This feature analyzes drawdowns that end upward trends across different bullish market regimes to understand which regime is most prone to trend-ending drawdowns. This provides insights into market behavior patterns that can inform risk management strategies and trend-following approaches.

## Key Design Principles

1. **HMM Market State Integration**: Uses trained HMM model to classify market regimes
2. **Proper Train/Test Split**: HMM trained on 24 months of historical data, then applied to analysis period
3. **Independent Analysis**: Only depends on `/common` folder components
4. **Value Object Pattern**: Uses immutable DTOs and VOs for data representation
5. **Comprehensive Visualization**: Provides both console output and detailed plots
6. **Statistical Rigor**: Calculates meaningful statistics across all trend durations

## Implementation Components

### Core Classes

#### 1. **UpwardTrendWithRegime** (Value Object)
```python
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
```

#### 2. **RegimeDrawdownStatistics** (Value Object)
```python
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
```

#### 3. **BullishRegimeDrawdownAnalysisResult** (Value Object)
```python
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
```

#### 4. **BullishRegimeDrawdownAnalyzer** (Main Analysis Class)
```python
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
        
    def identify_upward_trends_with_regimes(self, data: pd.DataFrame) -> List[UpwardTrendWithRegime]:
        """Identify upward trends and classify their market regimes"""
        
    def calculate_trend_ending_drawdown(self, data: pd.DataFrame, trend_end_idx: int) -> Tuple[float, float]:
        """Calculate drawdown that occurs when the upward trend ends"""
        
    def calculate_regime_statistics(self, trends: List[UpwardTrendWithRegime]) -> Dict[MarketStateType, RegimeDrawdownStatistics]:
        """Calculate drawdown statistics grouped by market regime"""
        
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
        
    def plot_results(self, result: BullishRegimeDrawdownAnalysisResult, save_path: str = None):
        """Create visualization plots for the analysis results"""
```

### Analysis Workflow

#### 1. **Data Preparation and HMM Training**
```python
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
```

#### 2. **Upward Trend Identification with Regime Classification**
```python
def identify_upward_trends_with_regimes(self, data: pd.DataFrame) -> List[UpwardTrendWithRegime]:
    """Identify upward trends and classify their market regimes"""
    print("\n🔍 Identifying upward trends and classifying market regimes...")
    
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
```

#### 3. **Drawdown Calculation**
```python
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
```

#### 4. **Regime-Specific Statistics**
```python
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
```

#### 5. **Report Generation**
```python
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
```

#### 6. **Visualization**
```python
def plot_results(self, result: BullishRegimeDrawdownAnalysisResult, save_path: str = None):
    """Create visualization plots for the analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bullish Market Regime Drawdown Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: SPY Price with Trend Periods and Drawdowns
    self._plot_price_with_trends_and_drawdowns(axes[0, 0], result)
    
    # Plot 2: Drawdown Distribution by Regime
    self._plot_drawdown_distribution_by_regime(axes[0, 1], result)
    
    # Plot 3: Drawdown Frequency by Regime
    self._plot_drawdown_frequency_by_regime(axes[1, 0], result)
    
    # Plot 4: Average Drawdown by Trend Duration and Regime
    self._plot_drawdown_by_duration_and_regime(axes[1, 1], result)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()
```

### Entry Point Script

#### **run_bullish_regime_analysis.py**
```python
#!/usr/bin/env python3
"""
Entry point for Bullish Market Regime Drawdown Analysis

This script runs the analysis to determine which bullish market regime
experiences the most drawdowns during upward trends.
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from analysis.bullish_regime_drawdown_analysis import BullishRegimeDrawdownAnalyzer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze drawdowns during upward trends by bullish market regime"
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )
    
    parser.add_argument(
        '--months', 
        type=int, 
        default=12,
        help='Number of months of data to analyze (default: 12). HMM will be trained on 24 months prior to this period.'
    )
    
    parser.add_argument(
        '--save-plot', 
        type=str, 
        default=None,
        help='Path to save the analysis plot (optional)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("🚀 Starting Bullish Market Regime Drawdown Analysis")
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Period: {args.months} months")
    print(f"HMM Training Period: 24 months prior to analysis period")
    print(f"Total Data Period: {args.months + 24} months")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer
        analyzer = BullishRegimeDrawdownAnalyzer(
            symbol=args.symbol,
            analysis_period_months=args.months
        )
        
        # Run analysis
        result = analyzer.run_analysis()
        
        # Generate and print report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
        # Create visualization
        analyzer.plot_results(result, save_path=args.save_plot)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"Most drawdown-prone bullish regime: {result.most_drawdown_prone_regime.value}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Success Criteria

1. **HMM Integration**: Successfully trains HMM model on 24 months of historical data
2. **Proper Train/Test Split**: HMM trained on separate data from analysis period
3. **Trend Identification**: Correctly identifies upward trends (3-10 consecutive positive days)
4. **Regime Classification**: Accurately classifies trends into bullish market regimes using trained HMM
5. **Trend-Ending Drawdown Calculation**: Properly calculates drawdowns that occur when upward trends end
6. **Statistical Analysis**: Generates min/mean/median/max statistics by regime
7. **Console Output**: Prints regime with most trends and highest mean drawdown
8. **Visualization**: Creates comprehensive plots showing trend-ending drawdown analysis
9. **Independence**: Only depends on `/common` folder components
10. **Value Objects**: Uses immutable DTOs and VOs throughout
11. **Error Handling**: Graceful handling of edge cases and data issues

## Expected Output

### Console Output
```
BULLISH MARKET REGIME DRAWDOWN ANALYSIS
================================================================================
Analysis Period: 2023-01-01 to 2024-01-01
Total Trading Days: 252
Total Upward Trends Analyzed: 45

OVERALL SUMMARY
----------------------------------------
Average Drawdown Across All Trends: 1.23%
Regime with Most Trends: momentum_uptrend
Regime with Highest Mean Drawdown: high_volatility_rally

REGIME-SPECIFIC ANALYSIS
----------------------------------------

LOW VOLATILITY UPTREND
  Total Trends: 18
  Min Drawdown: 0.15%
  Mean Drawdown: 0.89%
  Median Drawdown: 0.67%
  Max Drawdown: 2.45%
  Std Dev Drawdown: 0.78%

MOMENTUM UPTREND
  Total Trends: 25
  Min Drawdown: 0.22%
  Mean Drawdown: 1.12%
  Median Drawdown: 0.95%
  Max Drawdown: 3.21%
  Std Dev Drawdown: 1.02%

HIGH VOLATILITY RALLY
  Total Trends: 12
  Min Drawdown: 0.45%
  Mean Drawdown: 1.89%
  Median Drawdown: 1.45%
  Max Drawdown: 4.67%
  Std Dev Drawdown: 1.34%
```

### Visualization
- **Plot 1**: SPY price chart with upward trend periods highlighted and drawdowns marked
- **Plot 2**: Box plot showing drawdown distribution by market regime
- **Plot 3**: Bar chart showing drawdown frequency by regime
- **Plot 4**: Heatmap showing average drawdown by trend duration and regime

## File Structure

```
src/analysis/
├── bullish_regime_drawdown_analysis.py    # Main analysis implementation
└── run_bullish_regime_analysis.py         # Entry point script

features/
└── bullish_market_regime_drawdown_analysis.md  # This documentation file
```

## Dependencies

- **DataRetriever**: For fetching SPY data and calculating features
- **MarketStateClassifier**: For HMM-based market regime classification
- **MarketStateType**: Enum for market regime types
- **Standard libraries**: pandas, numpy, matplotlib, dataclasses

## Testing Strategy

1. **Unit Tests**: Test individual methods with mock data
2. **Integration Tests**: Test complete workflow with real data
3. **Edge Case Tests**: Test with minimal data, no trends, etc.
4. **Value Object Tests**: Verify immutability and validation
5. **Visualization Tests**: Verify plot generation and saving

## Future Enhancements

1. **Confidence Metrics**: Include HMM confidence scores in analysis
2. **Statistical Significance**: Add statistical tests for regime differences
3. **Interactive Plots**: Create interactive visualizations
4. **Export Functionality**: Export results to CSV/JSON
5. **Comparative Analysis**: Compare with bearish regime behavior
