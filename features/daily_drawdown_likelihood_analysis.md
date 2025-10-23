# Daily Drawdown Likelihood Analysis

## Overview

This document outlines the implementation of an analysis that answers the question: **"What is the average daily likelihood of a drawdown for each day of a 3-10 day upward trend?"**

The analysis will identify upward trends in SPY (continuous positive daily returns lasting 3-10 days), and for each day position within those trends (day 1, day 2, ..., day 10), calculate the probability of a drawdown occurring on that specific day.

---

## Analysis Goals

1. **Identify Upward Trends**: Detect all instances where SPY experiences continuous positive returns for 3-10 consecutive days
2. **Measure Daily Drawdowns**: For each day within each trend, determine if a drawdown occurred (intraday decline from open/previous close)
3. **Calculate Likelihood by Day Position**: Compute the probability of a drawdown on day 1, day 2, ..., day 10 of a trend
4. **Visualize Patterns**: Create plots showing SPY prices with drawdowns and likelihood distribution by day position

---

## Definitions

### Upward Trend
A sequence of consecutive trading days where SPY experiences positive returns (close-to-close price increases):
- **Minimum Duration**: 3 consecutive days of positive returns
- **Maximum Duration**: 10 consecutive days of positive returns
- **Calculation**: Daily return = (Close[t] - Close[t-1]) / Close[t-1]
- **Requirement**: Each daily return must be > 0 for the entire trend period

### Drawdown (Daily)
An intraday decline in SPY price that occurs **on a specific day** during an upward trend:
- **Definition**: For a given day within a trend, a drawdown occurs if Low[day] < max(Open[day], Close[day-1])
- **Reference Point**: The higher of the day's opening price or the previous day's close price
- **Intraday Focus**: Measures whether the price dropped below the reference point at any point during the day
- **Two Types**:
  - **Intraday drawdown**: Low < Previous Close (but Open >= Previous Close)
  - **Gap down drawdown**: Low < Previous Close (and Open < Previous Close)
- **Binary Measure**: Each day either has a drawdown (True) or doesn't (False)

### Day Position
The position of a day within an upward trend:
- **Day 1**: First day of the trend (first positive return)
- **Day 2**: Second day of the trend
- **Day N**: Nth day of the trend (where N can be 3-10)

### Likelihood
The probability that a drawdown occurs on a specific day position:
- **Calculation**: (Number of trends where drawdown occurred on day N) / (Total number of trends that reached day N)
- **Example**: If 80 out of 100 trends had a drawdown on day 3, the likelihood for day 3 is 80%

---

## Implementation Design

### Architecture

The implementation will follow the project's established patterns:

1. **Location**: `src/analysis/daily_drawdown_likelihood_analysis.py`
2. **Entry Point**: `src/analysis/run_daily_likelihood_analysis.py`
3. **Dependencies**: Only `src/common/` modules (primarily `data_retriever.py`)
4. **Data Models**: Use `@dataclass` for Value Objects (VOs) following .cursorrules guidelines

### Core Components

#### 1. Value Objects (VOs)

```python
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
    daily_drawdowns: List[DailyDrawdown]  # One entry per day in the trend
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
    trends_with_daily_data: List[TrendWithDailyDrawdowns]
    likelihood_by_day_position: Dict[int, DailyLikelihoodStatistics]  # Key: day position (1-10)
    overall_daily_likelihood: float  # Average likelihood across all days
```

#### 2. Main Analyzer Class

```python
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
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical price data for the specified period.
        
        Returns:
            DataFrame with OHLCV data and calculated returns
        """
        
    def identify_upward_trends(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identify all upward trends (3-10 consecutive days of positive returns).
        
        Args:
            data: DataFrame with price data and returns
            
        Returns:
            List of tuples (start_index, end_index) for each trend
        """
        
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
        
    def calculate_likelihood_by_day_position(
        self, 
        trends: List[TrendWithDailyDrawdowns]
    ) -> Dict[int, DailyLikelihoodStatistics]:
        """
        Calculate drawdown likelihood for each day position (1-10).
        
        Args:
            trends: List of trends with daily drawdown data
            
        Returns:
            Dictionary mapping day position to likelihood statistics
        """
        
    def run_analysis(self) -> DailyDrawdownAnalysisResult:
        """
        Execute the complete analysis workflow.
        
        Returns:
            DailyDrawdownAnalysisResult with all findings
        """
        
    def generate_report(self, result: DailyDrawdownAnalysisResult) -> str:
        """
        Generate a text report of the analysis results.
        
        Args:
            result: Complete analysis results
            
        Returns:
            Formatted report string
        """
        
    def plot_results(self, result: DailyDrawdownAnalysisResult, save_path: str = None):
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
3. Ensure data has OHLC (Open, High, Low, Close) for intraday analysis

### Step 2: Identify Upward Trends

```python
def identify_upward_trends(data: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Same algorithm as the previous analysis:
    
    1. Iterate through each day in the dataset
    2. For each day, check if it starts a sequence of positive returns
    3. Count consecutive positive returns (stopping at first negative/zero return)
    4. If sequence length is between 3-10 days, record as an upward trend
    5. Return list of (start_index, end_index) tuples
    """
    trends = []
    i = 1  # Start at 1 because returns start at index 1
    
    while i < len(data) - 2:
        if data['Daily_Return'].iloc[i] > 0:
            trend_start_idx = i
            consecutive_positive_days = 0
            
            j = i
            while j < len(data) and data['Daily_Return'].iloc[j] > 0:
                consecutive_positive_days += 1
                j += 1
                if consecutive_positive_days >= 10:
                    break
            
            if 3 <= consecutive_positive_days <= 10:
                trend_end_idx = trend_start_idx + consecutive_positive_days - 1
                trends.append((trend_start_idx, trend_end_idx))
            
            i = j
        else:
            i += 1
    
    return trends
```

### Step 3: Check Daily Drawdown

```python
def check_daily_drawdown(data: pd.DataFrame, day_idx: int) -> Tuple[bool, float, str]:
    """
    For a given day, determine if a drawdown occurred:
    
    1. Get OHLC data for the day
    2. Get previous day's close
    3. Calculate reference point (max of open and previous close)
    4. Check if low is below reference point
    5. If yes, calculate magnitude and classify type
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
            drawdown_type = 'intraday'
        
        return True, drawdown_magnitude, drawdown_type
    else:
        return False, 0.0, 'none'
```

### Step 4: Analyze Trend Daily Drawdowns

```python
def analyze_trend_daily_drawdowns(
    data: pd.DataFrame, 
    start_idx: int, 
    end_idx: int
) -> TrendWithDailyDrawdowns:
    """
    For each day in the trend:
    1. Check if a drawdown occurred
    2. Record the details in a DailyDrawdown object
    3. Track overall statistics for the trend
    """
    duration = end_idx - start_idx + 1
    daily_drawdowns = []
    days_with_drawdowns = 0
    
    for day_offset in range(duration):
        day_idx = start_idx + day_offset
        day_position = day_offset + 1  # 1-based position
        
        # Check for drawdown on this day
        had_drawdown, magnitude, dd_type = check_daily_drawdown(data, day_idx)
        
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
        daily_drawdowns=daily_drawdowns,
        total_days_with_drawdowns=days_with_drawdowns,
        drawdown_frequency=days_with_drawdowns / duration if duration > 0 else 0.0
    )
    
    return trend
```

### Step 5: Calculate Likelihood by Day Position

```python
def calculate_likelihood_by_day_position(
    trends: List[TrendWithDailyDrawdowns]
) -> Dict[int, DailyLikelihoodStatistics]:
    """
    For each day position (1-10):
    1. Count how many trends reached this day
    2. Count how many had a drawdown on this day
    3. Calculate likelihood percentage
    4. Calculate average drawdown magnitude for days with drawdowns
    """
    likelihood_stats = {}
    
    for day_position in range(1, 11):  # Days 1-10
        # Collect all daily drawdowns for this position
        daily_data_for_position = []
        
        for trend in trends:
            # Check if this trend reached this day position
            if trend.duration >= day_position:
                # Get the daily drawdown for this position (0-indexed)
                daily_dd = trend.daily_drawdowns[day_position - 1]
                daily_data_for_position.append(daily_dd)
        
        if not daily_data_for_position:
            continue
        
        # Calculate statistics
        total_reaching = len(daily_data_for_position)
        with_drawdown = [dd for dd in daily_data_for_position if dd.had_drawdown]
        num_with_drawdown = len(with_drawdown)
        
        likelihood_pct = (num_with_drawdown / total_reaching * 100) if total_reaching > 0 else 0.0
        
        # Calculate drawdown magnitudes
        if with_drawdown:
            magnitudes = [dd.drawdown_magnitude for dd in with_drawdown]
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
            trends_with_drawdown_on_day=num_with_drawdown,
            likelihood_percentage=likelihood_pct,
            average_drawdown_magnitude=avg_magnitude,
            max_drawdown_magnitude=max_magnitude,
            min_drawdown_magnitude=min_magnitude
        )
        
        likelihood_stats[day_position] = stats
    
    return likelihood_stats
```

---

## Output Specifications

### 1. Console Output

The analysis will print a comprehensive report to the console with the following sections:

```
================================================================================
DAILY DRAWDOWN LIKELIHOOD ANALYSIS REPORT
================================================================================
Symbol: SPY
Analysis Period: YYYY-MM-DD to YYYY-MM-DD
Total Trading Days: XXX
Total Upward Trends Analyzed: XX

OVERALL STATISTICS:
  Average Daily Drawdown Likelihood: XX.X%
  Total Trend-Days Analyzed: XXX
  Trend-Days with Drawdowns: XXX (XX.X%)

LIKELIHOOD BY DAY POSITION:
--------------------------------------------------------------------------------
Day 1 (First day of trend):
  Trends Reaching Day 1: XX
  Drawdowns on Day 1: XX (XX.X%)
  Average Drawdown Magnitude: X.XX%
  Max Drawdown on Day 1: X.XX%

Day 2:
  Trends Reaching Day 2: XX
  Drawdowns on Day 2: XX (XX.X%)
  Average Drawdown Magnitude: X.XX%
  Max Drawdown on Day 2: X.XX%

Day 3:
  Trends Reaching Day 3: XX
  Drawdowns on Day 3: XX (XX.X%)
  Average Drawdown Magnitude: X.XX%
  Max Drawdown on Day 3: X.XX%

[... continues through Day 10 ...]

INSIGHTS:
  Most Likely Day for Drawdown: Day X (XX.X%)
  Least Likely Day for Drawdown: Day X (XX.X%)
  Highest Average Drawdown Magnitude: Day X (X.XX%)

TREND DURATION BREAKDOWN:
  3-day trends: XX trends
  4-day trends: XX trends
  5-day trends: XX trends
  6-day trends: XX trends
  7-day trends: XX trends
  8-day trends: XX trends
  9-day trends: XX trends
  10-day trends: XX trends

================================================================================
```

### 2. Visualization

A matplotlib figure with multiple subplots:

#### Plot 1: SPY Price with Drawdown Days Highlighted
- **X-axis**: Date
- **Y-axis**: SPY Price
- **Elements**:
  - Line plot of SPY closing prices
  - Green shaded regions for upward trends
  - Red markers on days with drawdowns
  - Size/intensity of markers proportional to drawdown magnitude

#### Plot 2: Likelihood by Day Position (Bar Chart)
- **X-axis**: Day position (1-10)
- **Y-axis**: Likelihood percentage (0-100%)
- **Elements**:
  - Bars showing drawdown likelihood for each day position
  - Error bars or confidence intervals
  - Horizontal line showing average likelihood
  - Color gradient (green to red) based on likelihood

#### Plot 3: Average Drawdown Magnitude by Day Position
- **X-axis**: Day position (1-10)
- **Y-axis**: Average drawdown magnitude (%)
- **Elements**:
  - Line plot with markers
  - Shows average size of drawdowns when they occur
  - Helps identify if certain days have larger drawdowns

#### Plot 4: Heatmap - Likelihood by Day Position and Trend Duration
- **X-axis**: Day position (1-10)
- **Y-axis**: Trend duration (3-10 days)
- **Color**: Likelihood percentage
- **Elements**:
  - Heatmap showing how likelihood varies by both position and total trend length
  - Helps identify if patterns differ for shorter vs longer trends

#### Plot 5: Cumulative Likelihood Curve
- **X-axis**: Day position (1-10)
- **Y-axis**: Cumulative probability (0-100%)
- **Elements**:
  - Shows the probability of experiencing at least one drawdown by day N
  - Helps answer "By day 5, what % of trends have had a drawdown?"

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `src/analysis/daily_drawdown_likelihood_analysis.py`
2. Implement Value Objects (VOs) using `@dataclass(frozen=True)`
3. Implement `DailyDrawdownLikelihoodAnalyzer` class skeleton
4. Set up data loading using `DataRetriever`

### Phase 2: Trend Identification
1. Implement `identify_upward_trends()` method (reuse from previous analysis)
2. Test with sample data

### Phase 3: Daily Drawdown Detection
1. Implement `check_daily_drawdown()` method
2. Add logic to detect intraday drawdowns
3. Classify drawdown types (intraday vs gap down)
4. Test drawdown detection accuracy

### Phase 4: Trend Analysis
1. Implement `analyze_trend_daily_drawdowns()` method
2. Create daily drawdown records for each trend
3. Calculate trend-level statistics
4. Test with various trend patterns

### Phase 5: Likelihood Calculation
1. Implement `calculate_likelihood_by_day_position()` method
2. Aggregate data across all trends
3. Calculate likelihood percentages
4. Validate statistical calculations

### Phase 6: Reporting
1. Implement `generate_report()` method
2. Format console output with all required sections
3. Add insights and interpretation

### Phase 7: Visualization
1. Implement `plot_results()` method
2. Create all five subplots
3. Add labels, legends, and formatting
4. Test plot rendering and saving

### Phase 8: Entry Point
1. Create `src/analysis/run_daily_likelihood_analysis.py`
2. Implement command-line interface
3. Add error handling
4. Test end-to-end execution

### Phase 9: Testing
1. Create `tests/test_daily_drawdown_likelihood_analysis.py`
2. Write unit tests for all major methods
3. Write integration tests for complete workflow
4. Validate results with known data

---

## Dependencies

### Internal Dependencies (from `src/common/`)
- `data_retriever.py`: For fetching historical SPY data with OHLC

### External Dependencies (already in requirements.txt)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical calculations
- `matplotlib`: Plotting and visualization
- `seaborn`: Enhanced heatmap visualization
- `dataclasses`: For Value Objects

---

## Success Criteria

1. ✅ **Correct Trend Identification**: All upward trends (3-10 consecutive positive return days) are identified
2. ✅ **Accurate Daily Drawdown Detection**: Each day's drawdown status is correctly determined
3. ✅ **Precise Likelihood Calculation**: Probabilities are correctly calculated for each day position
4. ✅ **Comprehensive Statistics**: All required statistics are calculated and displayed
5. ✅ **Clear Console Output**: Results are printed in a readable, well-formatted manner
6. ✅ **Informative Visualization**: Plots clearly show likelihood patterns by day position
7. ✅ **Independent Implementation**: Only depends on `src/common/` modules
8. ✅ **Follows .cursorrules**: Uses VOs, DTOs, follows project structure guidelines
9. ✅ **Well-Tested**: Has comprehensive unit and integration tests
10. ✅ **Executable**: Can be run via `python -m src.analysis.run_daily_likelihood_analysis`

---

## Usage Example

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the analysis (default: SPY, 12 months)
python -m src.analysis.run_daily_likelihood_analysis

# Custom analysis period
python -m src.analysis.run_daily_likelihood_analysis --symbol SPY --months 6

# Custom output file
python -m src.analysis.run_daily_likelihood_analysis --output daily_likelihood.png

# Skip plotting (console output only)
python -m src.analysis.run_daily_likelihood_analysis --no-plot
```

---

## Expected Insights

This analysis will answer questions such as:

1. **Are drawdowns more likely early or late in a trend?**
   - If likelihood increases with day position, later days are riskier
   - If likelihood decreases, early days are more volatile

2. **Is there a "safe" day position?**
   - Identifying day positions with lower drawdown likelihood
   - Could inform entry/exit timing strategies

3. **Do drawdown patterns differ by trend duration?**
   - The heatmap will reveal if 3-day trends behave differently than 10-day trends
   - Helps understand if longer trends show different risk profiles

4. **What's the cumulative risk?**
   - The cumulative curve shows the probability of experiencing at least one drawdown by day N
   - Helps set realistic expectations for trend-following strategies

---

## Future Enhancements (Out of Scope)

1. **Multiple Symbols**: Extend to analyze multiple tickers beyond SPY
2. **Conditional Analysis**: Analyze likelihood conditioned on market regime or volatility
3. **Time-of-Day Analysis**: Incorporate intraday timing (morning vs afternoon drawdowns)
4. **Severity Analysis**: Group by drawdown magnitude (small vs large)
5. **Recovery Analysis**: Track how quickly prices recover from drawdowns
6. **Comparative Analysis**: Compare drawdown patterns across different market periods

---

## Key Differences from Previous Analysis

### Previous Analysis (Upward Trend Drawdown)
- Focused on **single-day trend-ending** drawdowns
- Close-to-close measurement on the day that ends the trend
- Single statistic: average drawdown when trends end

### This Analysis (Daily Drawdown Likelihood)
- Focused on **intraday** drawdown occurrence during trends
- Day-by-day binary measurement (did intraday drawdown occur?)
- Uses High/Low prices to detect intraday declines
- Multiple statistics: likelihood for each day position (1-10)
- Reveals **temporal patterns** within trends
- Measures drawdowns **during** trends, not when trends end

---

## Notes

- **Data Quality**: Requires OHLC (Open, High, Low, Close) data for accurate intraday drawdown detection
- **Market Hours**: Only considers daily data, not minute-by-minute intraday movements
- **Holidays**: Market holidays are automatically excluded by the data retriever
- **Lookback Period**: Default is 12 months but can be configured
- **Statistical Significance**: Longer analysis periods provide more reliable likelihood estimates
- **Trend Overlap**: Trends are non-overlapping by design (each trend is analyzed independently)

