# Moving Average Velocity/Elasticity Analysis Guide

## Overview

The Moving Average Velocity Analysis module analyzes the elasticity/velocity (short term MA / longer term MA) that best signals upward and downward trends in SPY over the last 6 months.

**Key Features:**
- Focuses purely on moving average analysis from SPY price changes
- No trading strategy, position management, or actual trades
- Uses only daily close prices for SPY
- Analyzes the last 6 months of data
- Identifies optimal MA combinations for upward and downward trend signals
- Generates comprehensive reports and visualizations

## What is MA Velocity/Elasticity?

MA Velocity is defined as the ratio of a short-term moving average to a longer-term moving average:

```
MA Velocity = Short MA / Long MA
```

For example:
- SMA 10 / SMA 50 = 0.2 (velocity ratio)
- SMA 20 / SMA 100 = 0.2 (velocity ratio)

The goal is to find MA combinations where changes in this velocity ratio best signal the start of upward or downward trends that last at least 3 days.

## Analysis Methodology

### Signal Identification
1. **Velocity Changes**: Monitor changes in MA velocity (increases for upward signals, decreases for downward signals)
2. **Trend Validation**: Check if the signal leads to a sustained trend of at least 3 days
3. **Success Criteria**: A signal is successful if it produces a trend in the expected direction without significant reversals

### Success Rate Calculation
- **Success Rate**: Percentage of signals that lead to successful trends
- **Trend Duration**: Average length of successful trends
- **Trend Return**: Average return of successful trends

## Installation and Setup

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -e .
```

### Data Requirements

The analysis uses the existing `DataRetriever` class to fetch SPY data from Yahoo Finance. No additional data sources are required.

## Usage

### Command Line Interface

The easiest way to run the analysis is using the provided CLI script:

```bash
# Basic analysis with default parameters (last 6 months)
python run_ma_analysis.py

# Different symbol
python run_ma_analysis.py --symbol QQQ

# Save plot to file
python run_ma_analysis.py --save-plot ma_analysis_results.png

# Custom MA periods
python run_ma_analysis.py --short-periods 5,10,15 --long-periods 30,50,100
```

### Programmatic Usage

You can also use the analysis module directly in your code:

```python
from src.analysis.ma_velocity_analysis import MAVelocityAnalyzer

# Initialize analyzer (will use last 6 months by default)
analyzer = MAVelocityAnalyzer(symbol='SPY')

# Define MA periods to test
short_periods = [5, 10, 15, 20, 25]
long_periods = [30, 50, 100, 150, 200]

# Find optimal combinations
optimal_combinations = analyzer.find_optimal_ma_combinations(short_periods, long_periods)

# Generate report
report = analyzer.generate_report(optimal_combinations)
print(report)

# Create visualizations
analyzer.plot_results(optimal_combinations, save_path='results.png')
```

## Configuration Options

### MA Periods

- **Short Periods**: Typically 5-25 days (default: 5,10,15,20,25)
- **Long Periods**: Typically 30-200 days (default: 30,50,100,150,200)

### Trend Parameters

- **Min Duration**: Minimum trend duration in days (default: 3)
- **Max Duration**: Maximum trend duration in days (default: 60)
- **Analysis Period**: Last 6 months from today

### Analysis Parameters

- **Symbol**: Stock symbol to analyze (default: SPY)

## Output

### Report Format

The analysis generates a comprehensive report including:

```
================================================================================
MOVING AVERAGE VELOCITY/ELASTICITY ANALYSIS REPORT
================================================================================
Symbol: SPY
Analysis Period: 2023-07-15 to 2024-01-15
Total Trading Days: 126
Total Trend Signals: 45

SIGNAL SUMMARY:
  Upward Trend Signals: 23
  Downward Trend Signals: 22
  Upward Signal Success Rate: 65.2%
  Downward Signal Success Rate: 59.1%

OPTIMAL MOVING AVERAGE COMBINATIONS:
--------------------------------------------------
UPWARD TREND SIGNALS:
  Short MA: 15 days
  Long MA: 100 days
  Velocity Ratio: 0.150
  Success Rate: 75.0%
  Total Signals: 8
  Successful Signals: 6
  Average Trend Duration: 7.2 days
  Average Trend Return: 3.45%

DOWNWARD TREND SIGNALS:
  Short MA: 10 days
  Long MA: 50 days
  Velocity Ratio: 0.200
  Success Rate: 66.7%
  Total Signals: 6
  Successful Signals: 4
  Average Trend Duration: 5.8 days
  Average Trend Return: -2.12%
```

### Visualizations

The analysis creates a 4-panel visualization:

1. **Price and Optimal MAs for Upward Trends**: Shows SPY price with the optimal MA combination for upward trend signals
2. **Price and Optimal MAs for Downward Trends**: Shows SPY price with the optimal MA combination for downward trend signals
3. **Successful Trend Signals**: Displays all successful signal points on the price chart
4. **MA Velocity Ratios**: Shows the velocity ratios over time for both optimal combinations

## Understanding Results

### Success Rate Interpretation

- **Higher Success Rate**: Better at identifying genuine trend signals
- **Lower Success Rate**: More false signals, may need adjustment
- **Balanced Rates**: Good for both upward and downward trends

### Optimal Combinations

The analysis identifies the MA combinations that show the highest success rates:

- **For Upward Trends**: MA combination that best signals upward trend starts
- **For Downward Trends**: MA combination that best signals downward trend starts

### Velocity Ratio

- **Velocity > 1.0**: Short MA is above long MA (bullish signal)
- **Velocity < 1.0**: Short MA is below long MA (bearish signal)
- **Velocity = 1.0**: Short MA equals long MA (neutral)

## Testing

Run the test suite to verify the implementation:

```bash
# Run all tests
pytest tests/test_ma_velocity_analysis.py -v

# Run specific test
pytest tests/test_ma_velocity_analysis.py::TestMAVelocityAnalysis::test_analyzer_initialization -v
```

## Limitations and Considerations

### Data Limitations

- Analysis is based on historical data and may not predict future performance
- Results are specific to the 6-month analysis period
- Market conditions change over time, affecting signal effectiveness

### Technical Limitations

- Signal identification is based on velocity changes and may miss some patterns
- Success rate does not guarantee future performance
- MA velocity is a lagging indicator

### Constraints

- Only uses daily close prices (no intraday data)
- Focuses on trends lasting 3-60 days
- No consideration of volume or other technical indicators
- Limited to 6-month analysis period

## Integration with Existing Codebase

The MA velocity analysis integrates seamlessly with the existing codebase:

- **Data Retrieval**: Uses existing `DataRetriever` class
- **Caching**: Leverages existing cache infrastructure
- **Dependencies**: Only depends on data retrieval classes as specified
- **Architecture**: Follows existing project patterns and conventions

## Future Enhancements

Potential improvements to consider:

1. **Volume Analysis**: Incorporate volume data for signal validation
2. **Multiple Timeframes**: Analyze different timeframes (weekly, monthly)
3. **Dynamic Parameters**: Adaptive MA periods based on market volatility
4. **Backtesting**: Historical performance validation of optimal combinations
5. **Real-time Analysis**: Live monitoring of MA velocity signals
6. **Extended Analysis Periods**: Allow analysis of longer historical periods

## Troubleshooting

### Common Issues

1. **No Data Retrieved**: Check internet connection and symbol validity
2. **Insufficient Signals**: Try reducing the minimum trend duration or adjusting MA periods
3. **Poor Success Rates**: Experiment with different MA period combinations
4. **Memory Issues**: Reduce the MA period ranges

### Error Messages

- `"No Close price data available"`: Data retrieval failed
- `"Insufficient data for feature calculation"`: Need more historical data
- `"No signals identified"`: Adjust signal identification parameters

## Support

For issues or questions:

1. Check the test suite for usage examples
2. Review the source code documentation
3. Verify your data and parameters
4. Run with verbose output for debugging

---

*This analysis tool is designed for research and educational purposes. Always conduct thorough testing before using any technical analysis in trading decisions.*
