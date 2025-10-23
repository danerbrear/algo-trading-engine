# Trend Detector Utility

## Overview

The `TrendDetector` utility class provides reusable trend detection functionality for trading strategies. It abstracts away common logic for detecting upward price trends in market data, making it easier to maintain consistent trend detection across different strategies.

## Location

- **Module**: `src/common/trend_detector.py`
- **Tests**: `tests/test_trend_detector.py`

## Features

### 1. Forward Trend Detection

Scans through historical data to find completed upward trends. This is useful for:
- **Reversal strategies**: Finding trends that have ended with a negative day
- **Historical analysis**: Identifying past trend patterns
- **Backtest preparation**: Pre-computing trend information

**Method**: `TrendDetector.detect_forward_trends()`

**Parameters**:
- `data`: DataFrame with 'Close' column
- `min_duration`: Minimum consecutive positive days (default: 3)
- `max_duration`: Maximum trend duration to consider (default: 10)
- `reversal_threshold`: Maximum allowed drawdown within trend (default: 2%)
- `require_reversal`: Whether trend must end with negative day (default: True)

**Returns**: List of `TrendInfo` objects

**Example**:
```python
from src.common.trend_detector import TrendDetector

trends = TrendDetector.detect_forward_trends(
    data=price_data,
    min_duration=3,
    max_duration=10,
    require_reversal=True
)

for trend in trends:
    print(f"Trend from {trend.start_date} to {trend.end_date}")
    print(f"Duration: {trend.duration} days")
    print(f"Return: {trend.net_return:.2%}")
    if trend.reversal_date:
        print(f"Reversal on {trend.reversal_date}: {trend.reversal_drawdown:.2%}")
```

### 2. Backward Trend Detection

Validates whether a signal point (e.g., velocity increase) occurred during a sustained upward trend. This is useful for:
- **Momentum strategies**: Confirming signals are part of real trends
- **Signal validation**: Filtering out false positives
- **Real-time trading**: Checking current market conditions

**Method**: `TrendDetector.check_backward_trend()`

**Parameters**:
- `data`: DataFrame with 'Close' column
- `signal_index`: Index in DataFrame where signal occurred
- `min_duration`: Minimum trend duration to validate (default: 3)
- `max_duration`: Maximum lookback period (default: 60)
- `reversal_threshold`: Maximum allowed drawdown (default: 2%)

**Returns**: Tuple of `(success: bool, duration: int, return: float)`

**Example**:
```python
from src.common.trend_detector import TrendDetector

# Check if velocity signal at current index is part of upward trend
success, duration, return_val = TrendDetector.check_backward_trend(
    data=price_data,
    signal_index=current_idx,
    min_duration=3,
    max_duration=60
)

if success:
    print(f"Valid trend detected: {duration} days, {return_val:.2%} return")
```

## Data Classes

### TrendInfo

Represents information about a detected trend.

**Fields**:
- `start_date`: datetime - When the trend started
- `end_date`: datetime - Last day of positive returns
- `duration`: int - Number of consecutive positive days
- `start_price`: float - Price at trend start
- `end_price`: float - Price at trend end
- `net_return`: float - Total return over the trend period
- `reversal_drawdown`: float - Drawdown on reversal day (optional)
- `reversal_date`: datetime - Date of first negative day (optional)

## Usage in Strategies

### Upward Trend Reversal Strategy

Uses forward trend detection to find completed trends for reversal trading:

```python
from src.common.trend_detector import TrendDetector

def _detect_upward_trends(self, data: pd.DataFrame) -> List[TrendInfo]:
    return TrendDetector.detect_forward_trends(
        data=data,
        min_duration=self.min_trend_duration,
        max_duration=self.max_trend_duration,
        reversal_threshold=0.02,
        require_reversal=True
    )
```

### Velocity Signal Momentum Strategy

Uses backward trend detection to validate momentum signals:

```python
from src.common.trend_detector import TrendDetector

def _check_backward_trend_success(self, data, signal_index, trend_type, 
                                  min_duration=3, max_duration=60):
    if trend_type != 'up':
        return False, 0, 0.0
    
    return TrendDetector.check_backward_trend(
        data=data,
        signal_index=signal_index,
        min_duration=min_duration,
        max_duration=max_duration,
        reversal_threshold=0.02
    )
```

## Key Design Decisions

### 1. Separate Forward and Backward Methods

**Why**: Different use cases require different approaches:
- **Forward**: Needs complete trend information including reversal details
- **Backward**: Needs quick validation from a specific point

### 2. Reversal Threshold

**Default**: 2% drawdown allowed within trend
**Rationale**: Filters out unstable trends with significant reversals while allowing normal market fluctuations

### 3. Immutable TrendInfo

**Why**: Ensures trend data can't be accidentally modified, making it safer to use across multiple parts of the codebase

### 4. Static Methods

**Why**: No state is needed - all inputs are provided as parameters, making the utility easier to test and use

## Benefits

### 1. Code Reusability
- Single implementation of trend detection logic
- Reduces duplication between strategies
- Easier to maintain and update

### 2. Consistency
- All strategies use the same trend detection algorithm
- Easier to compare strategy performance
- More predictable behavior

### 3. Testability
- Comprehensive test suite (18 tests)
- Tests cover forward detection, backward validation, edge cases
- Easier to test strategies in isolation

### 4. Flexibility
- Configurable parameters for different strategy needs
- Can be used for various timeframes and thresholds
- Supports both reversal and momentum strategies

## Testing

The utility includes comprehensive tests covering:

- **Forward trend detection**: Various trend lengths and patterns
- **Backward trend validation**: Signal confirmation scenarios
- **Trend sustainability**: Reversal detection within trends
- **Edge cases**: Empty data, single rows, boundary conditions
- **Data class**: TrendInfo creation and attributes

Run tests with:
```bash
pytest tests/test_trend_detector.py -v
```

## Migration Notes

### From Strategy-Specific Methods

**Before** (in strategy):
```python
def _detect_upward_trends(self, data):
    trends = []
    returns = data['Close'].pct_change()
    # ... 60+ lines of trend detection logic
    return trends
```

**After** (in strategy):
```python
from src.common.trend_detector import TrendDetector

def _detect_upward_trends(self, data):
    return TrendDetector.detect_forward_trends(
        data=data,
        min_duration=self.min_trend_duration,
        max_duration=self.max_trend_duration,
        require_reversal=True
    )
```

### Compatibility

The abstraction maintains backward compatibility:
- Same `TrendInfo` structure
- Same algorithm parameters
- Same return types
- All existing tests still pass

## Future Enhancements

Potential additions to consider:

1. **Downward trend detection**: Mirror functionality for bearish trends
2. **Trend quality metrics**: Additional scoring for trend strength
3. **Multi-timeframe support**: Detect trends across different periods
4. **Volume confirmation**: Incorporate volume data in trend validation
5. **Customizable reversal logic**: Different reversal detection methods

