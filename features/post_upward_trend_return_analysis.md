# Post-Upward Trend Return Analysis

## Overview

This document outlines the implementation of an analysis that answers the question: **"What is the average return 1-3 days after an upward trend ends?"**

The analysis will identify upward trends in SPY (continuous positive daily returns lasting 3-10 days), measure the returns that occur in the 1-3 days immediately following the end of each trend, and provide statistical insights with visualizations.

---

## Analysis Goals

1. **Identify Upward Trends**: Detect all instances where SPY experiences continuous positive returns for 3-10 consecutive days
2. **Measure Post-Trend Returns**: For each identified upward trend, measure the returns over the next 1, 2, and 3 days after the trend ends
3. **Calculate Statistics**: Compute the average returns across all trend durations (3-10 days) for each post-trend period (1-day, 2-day, 3-day)
4. **Visualize Results**: Create plots showing SPY prices with identified upward trends and subsequent returns highlighted

---

## Definitions

### Upward Trend
A sequence of consecutive trading days where SPY experiences positive returns (close-to-close price increases):
- **Minimum Duration**: 3 consecutive days of positive returns
- **Maximum Duration**: 10 consecutive days of positive returns
- **Calculation**: Daily return = (Close[t] - Close[t-1]) / Close[t-1]
- **Requirement**: Each daily return must be > 0 for the entire trend period

### Post-Trend Return Period
The period immediately following the end of an upward trend:
- **Day 1**: The first trading day after the upward trend ends (when the positive return streak breaks)
- **Day 2**: The second trading day after the upward trend ends
- **Day 3**: The third trading day after the upward trend ends
- **Measurement**: Cumulative return from the end of the trend to the specified day

### Return Calculation
Returns measured from the closing price on the last day of the upward trend:
- **1-Day Return**: (Close[trend_end+1] - Close[trend_end]) / Close[trend_end]
- **2-Day Return**: (Close[trend_end+2] - Close[trend_end]) / Close[trend_end]
- **3-Day Return**: (Close[trend_end+3] - Close[trend_end]) / Close[trend_end]
- **Note**: These can be positive (continued gains) or negative (reversals/drawdowns)

---

## Implementation Design

### Architecture

The implementation will follow the project's established patterns:

1. **Location**: `src/analysis/post_upward_trend_return_analysis.py`
2. **Entry Point**: `src/analysis/run_post_trend_analysis.py`
3. **Dependencies**: Only `src/common/` modules (primarily `data_retriever.py`)
4. **Data Models**: Use `@dataclass(frozen=True)` for Value Objects (VOs) following .cursorrules guidelines

### Core Components

#### 1. Value Objects (VOs)

```python
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
```

```python
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
```

```python
@dataclass(frozen=True)
class PostTrendAnalysisResult:
    """Complete results of the post-trend return analysis."""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    upward_trends: List[UpwardTrend]
    statistics: PostTrendStatistics
```

#### 2. Main Analyzer Class

```python
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
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical price data for the specified period.
        
        Returns:
            DataFrame with OHLC data and calculated returns
        """
        
    def identify_upward_trends(self, data: pd.DataFrame) -> List[UpwardTrend]:
        """
        Identify all upward trends (3-10 consecutive days of positive returns)
        and calculate post-trend returns.
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of UpwardTrend objects with post-trend return data
        """
        
    def calculate_post_trend_returns(self, data: pd.DataFrame, 
                                     trend_end_idx: int) -> Tuple[Optional[float], ...]:
        """
        Calculate returns for 1, 2, and 3 days after trend ends.
        
        Args:
            data: DataFrame with price data
            trend_end_idx: Index of the last day of the upward trend
            
        Returns:
            Tuple of (return_1d, return_2d, return_3d, price_1d, price_2d, price_3d)
            Values are None if insufficient data exists
        """
        
    def calculate_statistics(self, trends: List[UpwardTrend]) -> PostTrendStatistics:
        """
        Calculate statistical measures of post-trend returns.
        
        Args:
            trends: List of identified upward trends
            
        Returns:
            PostTrendStatistics object with summary metrics
        """
        
    def run_analysis(self) -> PostTrendAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            PostTrendAnalysisResult with all findings
        """
        
    def generate_report(self, result: PostTrendAnalysisResult) -> str:
        """
        Generate a text report of the analysis results.
        
        Args:
            result: Complete analysis results
            
        Returns:
            Formatted report string
        """
        
    def plot_results(self, result: PostTrendAnalysisResult, save_path: str = None):
        """
        Create visualizations of the analysis results.
        
        Args:
            result: Complete analysis results
            save_path: Optional path to save the plot
        """
```

---

## Algorithm Details

### Step 1: Data Loading and Preparation

1. Load historical SPY data using `DataRetriever` from `src/common/data_retriever.py`
2. Calculate daily returns: `return[t] = (close[t] - close[t-1]) / close[t-1]`
3. Ensure data is sorted by date (ascending)
4. Ensure sufficient data exists for post-trend analysis (need at least 3 days after last trend)

### Step 2: Identify Upward Trends and Calculate Post-Trend Returns

```python
def identify_upward_trends(data: pd.DataFrame) -> List[UpwardTrend]:
    """
    Algorithm to identify upward trends and calculate post-trend returns:
    
    1. Iterate through each day in the dataset
    2. For each day, check if it starts a sequence of positive returns
    3. Count consecutive positive returns (stopping at first negative/zero return)
    4. If sequence length is between 3-10 days, record as an upward trend
    5. Calculate returns for 1, 2, and 3 days after the trend ends
    6. Return list of UpwardTrend objects
    """
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
                 price_1d, price_2d, price_3d) = calculate_post_trend_returns(
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
    
    return trends
```

### Step 3: Calculate Post-Trend Returns

```python
def calculate_post_trend_returns(data: pd.DataFrame, 
                                 trend_end_idx: int) -> Tuple[Optional[float], ...]:
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
```

### Step 4: Calculate Statistics

```python
def calculate_statistics(trends: List[UpwardTrend]) -> PostTrendStatistics:
    """
    Calculate comprehensive statistics for post-trend returns:
    
    1. Count trends with sufficient data for each period (1d, 2d, 3d)
    2. Calculate average, median, std, min, max for each period
    3. Calculate percentage of negative returns (reversals) for each period
    4. Group average returns by trend duration (3-10 days) for each period
    """
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
```

---

## Output Specifications

### 1. Console Output

The analysis will print a comprehensive report to the console:

```
================================================================================
POST-UPWARD TREND RETURN ANALYSIS REPORT
================================================================================
Symbol: SPY
Analysis Period: YYYY-MM-DD to YYYY-MM-DD
Total Trading Days: XXX

TREND IDENTIFICATION SUMMARY:
  Total Upward Trends (3-10 days): XXX
  Trends by Duration:
    3 days: XX trends
    4 days: XX trends
    5 days: XX trends
    6 days: XX trends
    7 days: XX trends
    8 days: XX trends
    9 days: XX trends
    10 days: XX trends

POST-TREND RETURN ANALYSIS:
  Trends with 1-day data: XX (XX.X%)
  Trends with 2-day data: XX (XX.X%)
  Trends with 3-day data: XX (XX.X%)

AVERAGE RETURNS AFTER TREND ENDS:
  1-Day Average Return: X.XX%
  2-Day Average Return: X.XX%
  3-Day Average Return: X.XX%

RETURN STATISTICS - 1 DAY AFTER:
  Mean: X.XX%
  Median: X.XX%
  Std Dev: X.XX%
  Max: X.XX%
  Min: X.XX%
  Negative Returns (Reversals): XX.X%

RETURN STATISTICS - 2 DAYS AFTER:
  Mean: X.XX%
  Median: X.XX%
  Std Dev: X.XX%
  Max: X.XX%
  Min: X.XX%
  Negative Returns (Reversals): XX.X%

RETURN STATISTICS - 3 DAYS AFTER:
  Mean: X.XX%
  Median: X.XX%
  Std Dev: X.XX%
  Max: X.XX%
  Min: X.XX%
  Negative Returns (Reversals): XX.X%

AVERAGE 1-DAY RETURN BY TREND DURATION:
  3-day trends: X.XX%
  4-day trends: X.XX%
  5-day trends: X.XX%
  6-day trends: X.XX%
  7-day trends: X.XX%
  8-day trends: X.XX%
  9-day trends: X.XX%
  10-day trends: X.XX%

AVERAGE 2-DAY RETURN BY TREND DURATION:
  3-day trends: X.XX%
  4-day trends: X.XX%
  5-day trends: X.XX%
  6-day trends: X.XX%
  7-day trends: X.XX%
  8-day trends: X.XX%
  9-day trends: X.XX%
  10-day trends: X.XX%

AVERAGE 3-DAY RETURN BY TREND DURATION:
  3-day trends: X.XX%
  4-day trends: X.XX%
  5-day trends: X.XX%
  6-day trends: X.XX%
  7-day trends: X.XX%
  8-day trends: X.XX%
  9-day trends: X.XX%
  10-day trends: X.XX%

TOP 5 BEST POST-TREND RETURNS (3-day):
  1. YYYY-MM-DD: +XX.XX% (X-day trend)
  2. YYYY-MM-DD: +XX.XX% (X-day trend)
  3. YYYY-MM-DD: +XX.XX% (X-day trend)
  4. YYYY-MM-DD: +XX.XX% (X-day trend)
  5. YYYY-MM-DD: +XX.XX% (X-day trend)

TOP 5 WORST POST-TREND RETURNS (3-day):
  1. YYYY-MM-DD: -XX.XX% (X-day trend)
  2. YYYY-MM-DD: -XX.XX% (X-day trend)
  3. YYYY-MM-DD: -XX.XX% (X-day trend)
  4. YYYY-MM-DD: -XX.XX% (X-day trend)
  5. YYYY-MM-DD: -XX.XX% (X-day trend)

================================================================================
```

### 2. Visualization

A matplotlib figure with multiple subplots:

#### Plot 1: SPY Price with Upward Trends and Post-Trend Markers
- **X-axis**: Date
- **Y-axis**: SPY Price
- **Elements**:
  - Line plot of SPY closing prices
  - Green shaded regions for identified upward trends
  - Red markers for negative post-trend returns (reversals/drawdowns)
  - Blue markers for positive post-trend returns (continuations)
  - Annotations for significant returns (|return| > 2%)

#### Plot 2: Return Distribution Comparison
- **Type**: Three overlapping histograms
- **X-axis**: Return percentage
- **Y-axis**: Frequency (normalized)
- **Elements**:
  - Distribution of 1-day returns (blue)
  - Distribution of 2-day returns (orange)
  - Distribution of 3-day returns (green)
  - Vertical lines for mean of each distribution

#### Plot 3: Average Return by Trend Duration (Grouped Bar Chart)
- **Type**: Grouped bar chart
- **X-axis**: Trend duration (3-10 days)
- **Y-axis**: Average return percentage
- **Elements**:
  - Three bars per duration (1-day, 2-day, 3-day returns)
  - Error bars showing standard deviation
  - Different colors for each time period

#### Plot 4: Cumulative Returns Analysis
- **Type**: Line plot
- **X-axis**: Days after trend ends (0, 1, 2, 3)
- **Y-axis**: Cumulative average return
- **Elements**:
  - Line showing progression of returns over 3 days
  - Separate lines for different trend durations
  - Shaded area showing standard deviation

#### Plot 5: Reversal Probability by Trend Duration
- **Type**: Bar chart
- **X-axis**: Trend duration (3-10 days)
- **Y-axis**: Percentage of negative returns
- **Elements**:
  - Three bars per duration (1-day, 2-day, 3-day reversal rates)
  - Horizontal line showing overall reversal rate

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `src/analysis/post_upward_trend_return_analysis.py`
2. Implement Value Objects (VOs) using `@dataclass(frozen=True)`
3. Implement `PostUpwardTrendReturnAnalyzer` class skeleton
4. Set up data loading using `DataRetriever`

### Phase 2: Trend Identification
1. Implement `identify_upward_trends()` method
2. Add logic to detect consecutive positive returns
3. Add validation for 3-10 day duration constraint
4. Test with sample data

### Phase 3: Post-Trend Return Calculation
1. Implement `calculate_post_trend_returns()` method
2. Add logic to calculate 1, 2, and 3-day returns after trend ends
3. Handle edge cases (insufficient data at end of dataset)
4. Test return calculation accuracy

### Phase 4: Statistical Analysis
1. Implement `calculate_statistics()` method
2. Calculate all summary statistics for each time period
3. Group statistics by trend duration
4. Calculate reversal probabilities
5. Validate statistical calculations

### Phase 5: Reporting
1. Implement `generate_report()` method
2. Format console output with all required sections
3. Add formatting for readability
4. Include top/bottom performing periods

### Phase 6: Visualization
1. Implement `plot_results()` method
2. Create all five subplots
3. Add labels, legends, and annotations
4. Test plot rendering and saving

### Phase 7: Entry Point
1. Create `src/analysis/run_post_trend_analysis.py`
2. Implement command-line interface
3. Add error handling
4. Test end-to-end execution

### Phase 8: Testing
1. Create `tests/test_post_upward_trend_return_analysis.py`
2. Write unit tests for all major methods
3. Write integration tests for complete workflow
4. Validate results with known data

---

## Dependencies

### Internal Dependencies (from `src/common/`)
- `data_retriever.py`: For fetching historical SPY data
- `functions.py`: For any utility functions (if needed)

### External Dependencies (already in requirements.txt)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical calculations
- `matplotlib`: Plotting and visualization
- `dataclasses`: For Value Objects
- `typing`: Type hints and Optional types

---

## Success Criteria

1. ✅ **Correct Trend Identification**: All upward trends (3-10 consecutive positive return days) are identified
2. ✅ **Accurate Return Calculation**: Post-trend returns are correctly calculated for 1, 2, and 3-day periods
3. ✅ **Comprehensive Statistics**: All required statistics are calculated and displayed for each time period
4. ✅ **Clear Console Output**: Results are printed in a readable, well-formatted manner
5. ✅ **Informative Visualization**: Plots clearly show SPY prices with trends and post-trend returns highlighted
6. ✅ **Independent Implementation**: Only depends on `src/common/` modules
7. ✅ **Follows .cursorrules**: Uses VOs with frozen dataclasses, follows project structure guidelines
8. ✅ **Well-Tested**: Has comprehensive unit and integration tests
9. ✅ **Executable**: Can be run via `python -m src.analysis.run_post_trend_analysis`
10. ✅ **Edge Case Handling**: Properly handles insufficient data at end of dataset

---

## Usage Example

```bash
# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Run the analysis
python -m src.analysis.run_post_trend_analysis

# Or run with custom parameters
python -m src.analysis.run_post_trend_analysis --symbol SPY --months 12
```

---

## Key Insights Expected

This analysis will help answer important questions:

1. **Reversal Tendency**: Do upward trends tend to reverse (negative returns) in the following days?
2. **Duration Impact**: Does the length of an upward trend (3 vs 10 days) affect post-trend behavior?
3. **Time Decay**: How do returns evolve over the 1-3 days following a trend? Do they improve or worsen?
4. **Risk Assessment**: What is the probability of a reversal after different trend durations?
5. **Strategy Implications**: Should traders take profits after certain trend durations, or let winners run?

---

## Future Enhancements (Out of Scope)

1. **Multiple Symbols**: Extend to analyze multiple tickers beyond SPY
2. **Configurable Periods**: Allow custom post-trend analysis periods (e.g., 5 days, 10 days)
3. **Intraday Analysis**: Analyze intraday price movements after trend ends
4. **Volume Integration**: Correlate post-trend returns with volume patterns
5. **Market Regime Analysis**: Compare post-trend behavior in different market conditions
6. **Machine Learning**: Predict post-trend returns based on trend characteristics

---

## Notes

- **Data Quality**: The accuracy of the analysis depends on the quality of historical data from the data retriever
- **Market Hours**: Only considers daily close-to-close returns, not intraday movements
- **Holidays**: Market holidays are automatically excluded by the data retriever
- **Lookback Period**: Default is 12 months but can be configured
- **Edge Cases**: Trends at the very end of the dataset may not have complete 3-day post-trend data
- **Statistical Significance**: Ensure sufficient sample size for each trend duration category
- **Interpretation**: Negative average returns indicate typical reversals/drawdowns after trends end

