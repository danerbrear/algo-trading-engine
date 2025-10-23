# Upward Trend Drawdown Analysis

## Overview

This document outlines the implementation of an analysis that answers the question: **"What is the average drawdown following a 3-10 day upward trend?"**

The analysis will identify upward trends in SPY (continuous positive daily returns lasting 3-10 days), measure the drawdowns that occur during these trends, and provide statistical insights with visualizations.

---

## Analysis Goals

1. **Identify Upward Trends**: Detect all instances where SPY experiences continuous positive returns for 3-10 consecutive days
2. **Measure Drawdowns**: For each identified upward trend, measure any price declines (drawdowns) that occur during the trend period
3. **Calculate Statistics**: Compute the average drawdown across all trend durations (3-10 days)
4. **Visualize Results**: Create a plot showing SPY prices with identified drawdowns highlighted

---

## Definitions

### Upward Trend
A sequence of consecutive trading days where SPY experiences positive returns (close-to-close price increases):
- **Minimum Duration**: 3 consecutive days of positive returns
- **Maximum Duration**: 10 consecutive days of positive returns
- **Calculation**: Daily return = (Close[t] - Close[t-1]) / Close[t-1]
- **Requirement**: Each daily return must be > 0 for the entire trend period

### Drawdown
A decline in SPY price that occurs during an upward trend:
- **Definition**: The percentage decline from a peak price to a subsequent trough price within the trend period
- **Calculation**: Drawdown = (Trough Price - Peak Price) / Peak Price
- **Scope**: Measured only during identified upward trend periods
- **Note**: Even during an overall upward trend, intraday or daily drawdowns can occur

---

## Implementation Design

### Architecture

The implementation will follow the project's established patterns:

1. **Location**: `src/analysis/upward_trend_drawdown_analysis.py`
2. **Entry Point**: `src/analysis/run_drawdown_analysis.py`
3. **Dependencies**: Only `src/common/` modules (primarily `data_retriever.py`)
4. **Data Models**: Use `@dataclass` for Value Objects (VOs) following .cursorrules guidelines

### Core Components

#### 1. Value Objects (VOs)

```python
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
```

```python
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
```

```python
@dataclass(frozen=True)
class DrawdownAnalysisResult:
    """Complete results of the drawdown analysis."""
    analysis_period_start: str
    analysis_period_end: str
    total_trading_days: int
    upward_trends: List[UpwardTrend]
    statistics: DrawdownStatistics
```

#### 2. Main Analyzer Class

```python
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
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical price data for the specified period.
        
        Returns:
            DataFrame with OHLC data and calculated returns
        """
        
    def identify_upward_trends(self, data: pd.DataFrame) -> List[UpwardTrend]:
        """
        Identify all upward trends (3-10 consecutive days of positive returns).
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of UpwardTrend objects
        """
        
    def calculate_drawdown_for_trend(self, data: pd.DataFrame, 
                                     start_idx: int, 
                                     end_idx: int) -> Tuple[float, float, float]:
        """
        Calculate the maximum drawdown during a specific trend period.
        
        Args:
            data: DataFrame with price data
            start_idx: Starting index of the trend
            end_idx: Ending index of the trend
            
        Returns:
            Tuple of (peak_price, trough_price, drawdown_pct)
        """
        
    def calculate_statistics(self, trends: List[UpwardTrend]) -> DrawdownStatistics:
        """
        Calculate statistical measures of drawdowns across all trends.
        
        Args:
            trends: List of identified upward trends
            
        Returns:
            DrawdownStatistics object with summary metrics
        """
        
    def run_analysis(self) -> DrawdownAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            DrawdownAnalysisResult with all findings
        """
        
    def generate_report(self, result: DrawdownAnalysisResult) -> str:
        """
        Generate a text report of the analysis results.
        
        Args:
            result: Complete analysis results
            
        Returns:
            Formatted report string
        """
        
    def plot_results(self, result: DrawdownAnalysisResult, save_path: str = None):
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

### Step 2: Identify Upward Trends

```python
def identify_upward_trends(data: pd.DataFrame) -> List[UpwardTrend]:
    """
    Algorithm to identify upward trends:
    
    1. Iterate through each day in the dataset
    2. For each day, check if it starts a sequence of positive returns
    3. Count consecutive positive returns (stopping at first negative/zero return)
    4. If sequence length is between 3-10 days, record as an upward trend
    5. Calculate drawdown for each identified trend
    6. Return list of UpwardTrend objects
    """
    trends = []
    i = 0
    
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
                
                # Calculate drawdown for this trend
                peak_price, trough_price, drawdown_pct = calculate_drawdown_for_trend(
                    data, trend_start_idx, trend_end_idx
                )
                
                # Create UpwardTrend object
                trend = UpwardTrend(
                    start_date=data.index[trend_start_idx],
                    end_date=data.index[trend_end_idx],
                    duration=consecutive_positive_days,
                    start_price=data['Close'].iloc[trend_start_idx],
                    end_price=data['Close'].iloc[trend_end_idx],
                    total_return=(...),
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
    
    return trends
```

### Step 3: Calculate Drawdowns

```python
def calculate_drawdown_for_trend(data: pd.DataFrame, 
                                 start_idx: int, 
                                 end_idx: int) -> Tuple[float, float, float]:
    """
    For a given trend period, calculate the maximum drawdown:
    
    1. Extract price data for the trend period
    2. Find the running maximum price (peak)
    3. For each subsequent price, calculate drawdown from peak
    4. Return the maximum drawdown observed
    """
    # Get high prices during the trend period (using High instead of Close for accuracy)
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
        
        # Update maximum drawdown if this is worse
        if drawdown < max_drawdown_pct:
            max_drawdown_pct = drawdown
            trough_price = current_low
    
    return peak_price, trough_price, max_drawdown_pct
```

### Step 4: Calculate Statistics

```python
def calculate_statistics(trends: List[UpwardTrend]) -> DrawdownStatistics:
    """
    Calculate comprehensive statistics:
    
    1. Total number of trends analyzed
    2. Number and percentage of trends with drawdowns
    3. Average drawdown across ALL trends (including 0% for trends with no drawdown)
    4. Average drawdown for ONLY trends that had drawdowns
    5. Min, max, median, standard deviation of drawdowns
    6. Average drawdown grouped by trend duration (3-10 days)
    """
    # Filter trends with actual drawdowns
    trends_with_drawdowns = [t for t in trends if t.has_drawdown]
    
    # Overall statistics
    total_trends = len(trends)
    num_with_drawdowns = len(trends_with_drawdowns)
    drawdown_percentage = (num_with_drawdowns / total_trends * 100) if total_trends > 0 else 0
    
    # Calculate average across ALL trends
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
```

---

## Output Specifications

### 1. Console Output

The analysis will print a comprehensive report to the console with the following sections:

```
================================================================================
UPWARD TREND DRAWDOWN ANALYSIS REPORT
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

DRAWDOWN ANALYSIS:
  Trends with Drawdowns: XX (XX.X%)
  Trends without Drawdowns: XX (XX.X%)

DRAWDOWN STATISTICS (across all trends):
  Average Drawdown: -X.XX%
  Median Drawdown: -X.XX%
  Standard Deviation: X.XX%
  Maximum Drawdown: -X.XX%
  Minimum Drawdown: -X.XX%

DRAWDOWN STATISTICS (only trends with drawdowns):
  Average Drawdown: -X.XX%
  Median Drawdown: -X.XX%

AVERAGE DRAWDOWN BY TREND DURATION:
  3-day trends: -X.XX%
  4-day trends: -X.XX%
  5-day trends: -X.XX%
  6-day trends: -X.XX%
  7-day trends: -X.XX%
  8-day trends: -X.XX%
  9-day trends: -X.XX%
  10-day trends: -X.XX%

TOP 5 LARGEST DRAWDOWNS:
  1. YYYY-MM-DD: -XX.XX% (X-day trend)
  2. YYYY-MM-DD: -XX.XX% (X-day trend)
  3. YYYY-MM-DD: -XX.XX% (X-day trend)
  4. YYYY-MM-DD: -XX.XX% (X-day trend)
  5. YYYY-MM-DD: -XX.XX% (X-day trend)

================================================================================
```

### 2. Visualization

A matplotlib figure with multiple subplots:

#### Plot 1: SPY Price with Upward Trends and Drawdowns
- **X-axis**: Date
- **Y-axis**: SPY Price
- **Elements**:
  - Line plot of SPY closing prices
  - Green shaded regions for identified upward trends
  - Red markers indicating drawdown trough points
  - Annotations for significant drawdowns (> 2%)

#### Plot 2: Drawdown Distribution
- **Type**: Histogram
- **X-axis**: Drawdown percentage bins
- **Y-axis**: Frequency (number of occurrences)
- **Elements**:
  - Bars showing distribution of drawdown magnitudes
  - Vertical line indicating average drawdown
  - Vertical line indicating median drawdown

#### Plot 3: Average Drawdown by Trend Duration
- **Type**: Bar chart
- **X-axis**: Trend duration (3-10 days)
- **Y-axis**: Average drawdown percentage
- **Elements**:
  - Bars for each trend duration
  - Error bars showing standard deviation
  - Horizontal line indicating overall average

#### Plot 4: Cumulative Trend Analysis
- **Type**: Line plot
- **X-axis**: Date
- **Y-axis**: Cumulative number of upward trends
- **Elements**:
  - Line showing cumulative count of upward trends over time
  - Different colors for different trend durations (stacked)

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `src/analysis/upward_trend_drawdown_analysis.py`
2. Implement Value Objects (VOs) using `@dataclass(frozen=True)`
3. Implement `UpwardTrendDrawdownAnalyzer` class skeleton
4. Set up data loading using `DataRetriever`

### Phase 2: Trend Identification
1. Implement `identify_upward_trends()` method
2. Add logic to detect consecutive positive returns
3. Add validation for 3-10 day duration constraint
4. Test with sample data

### Phase 3: Drawdown Calculation
1. Implement `calculate_drawdown_for_trend()` method
2. Add logic to track peak prices during trends
3. Calculate maximum drawdown from peak to trough
4. Test drawdown calculation accuracy

### Phase 4: Statistical Analysis
1. Implement `calculate_statistics()` method
2. Calculate all summary statistics
3. Group statistics by trend duration
4. Validate statistical calculations

### Phase 5: Reporting
1. Implement `generate_report()` method
2. Format console output with all required sections
3. Add formatting for readability

### Phase 6: Visualization
1. Implement `plot_results()` method
2. Create all four subplots
3. Add labels, legends, and annotations
4. Test plot rendering and saving

### Phase 7: Entry Point
1. Create `src/analysis/run_drawdown_analysis.py`
2. Implement command-line interface
3. Add error handling
4. Test end-to-end execution

### Phase 8: Testing
1. Create `tests/test_upward_trend_drawdown_analysis.py`
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

---

## Success Criteria

1. ✅ **Correct Trend Identification**: All upward trends (3-10 consecutive positive return days) are identified
2. ✅ **Accurate Drawdown Calculation**: Drawdowns are correctly calculated as peak-to-trough declines
3. ✅ **Comprehensive Statistics**: All required statistics are calculated and displayed
4. ✅ **Clear Console Output**: Results are printed in a readable, well-formatted manner
5. ✅ **Informative Visualization**: Plot clearly shows SPY prices with drawdowns highlighted
6. ✅ **Independent Implementation**: Only depends on `src/common/` modules
7. ✅ **Follows .cursorrules**: Uses VOs, DTOs, follows project structure guidelines
8. ✅ **Well-Tested**: Has comprehensive unit and integration tests
9. ✅ **Executable**: Can be run via `python -m src.analysis.run_drawdown_analysis`

---

## Usage Example

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the analysis
python -m src.analysis.run_drawdown_analysis

# Or run with custom parameters
python -m src.analysis.run_drawdown_analysis --symbol SPY --months 12
```

---

## Future Enhancements (Out of Scope)

1. **Multiple Symbols**: Extend to analyze multiple tickers beyond SPY
2. **Configurable Trend Duration**: Allow custom min/max trend durations
3. **Drawdown Severity Threshold**: Filter trends by minimum drawdown magnitude
4. **Export Results**: Save results to CSV or JSON format
5. **Interactive Plots**: Use plotly for interactive visualizations
6. **Real-time Analysis**: Integrate with live market data

---

## Notes

- **Data Quality**: The accuracy of the analysis depends on the quality of historical data from the data retriever
- **Market Hours**: Only considers daily close-to-close returns, not intraday movements
- **Holidays**: Market holidays are automatically excluded by the data retriever
- **Lookback Period**: Default is 12 months but can be configured
- **Drawdown Definition**: Uses High-Low data for intraday drawdown accuracy where available

