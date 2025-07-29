# Volume Trading Backtest Feature Implementation Plan

## Feature Description
Prevent trades when volume is less than 10 for an option contract when backtesting any strategy.

## Overview
This feature will add volume validation to the backtesting system to ensure that only options with sufficient liquidity (volume >= 10) are considered for trading. This will improve the realism of backtests by avoiding illiquid options that would be difficult to trade in real market conditions.

**Key Design Principle**: Volume validation will be implemented at the highest level of abstraction possible - in the `BacktestEngine._add_position()` method. This approach minimizes changes to individual strategies and keeps the validation logic centralized.

## Implementation Plan

### Phase 1: Core Volume Validation Infrastructure

#### Task 1.1: Create Volume Validation Module
**File**: `src/backtest/volume_validator.py`
**Description**: Create a dedicated module for volume validation logic
**Components**:
- `VolumeValidator` class with static methods
- `validate_option_volume(option: Option, min_volume: int = 10) -> bool`
- `validate_spread_volume(spread_options: list[Option], min_volume: int = 10) -> bool`
- `get_volume_status(option: Option) -> dict` for detailed volume information

#### Task 1.2: Create Volume Configuration System
**File**: `src/backtest/config.py`
**Description**: Create configuration system for volume thresholds
**Components**:
- `VolumeConfig` DTO with configurable minimum volume
- `VolumeStats` DTO for tracking volume validation statistics
- Default settings and validation
- Integration with existing backtest configuration

### Phase 2: Backtest Engine Integration (Central Implementation)

#### Task 2.1: Update BacktestEngine Constructor
**File**: `src/backtest/main.py`
**Description**: Add volume configuration to BacktestEngine constructor
**Components**:
- Add `min_volume: int = 10` parameter to constructor
- Add `enable_volume_validation: bool = True` parameter
- Add volume statistics tracking attributes

#### Task 2.2: Implement Volume Validation in _add_position()
**File**: `src/backtest/main.py`
**Description**: Add volume validation to the _add_position method
**Components**:
- Add volume validation before position is added
- Validate all options in spread_options list
- Log volume validation decisions
- Track volume rejection statistics
- Return early if volume validation fails

#### Task 2.3: Add Volume Statistics and Reporting
**File**: `src/backtest/main.py`
**Description**: Add volume-related statistics to backtest results
**Components**:
- Track rejected trades due to volume
- Track API fetch attempts and failures
- Track cache updates with fresh volume data
- Add comprehensive volume statistics to final report
- Add volume validation summary in `_end()` method

### Phase 3: Data Integration and Validation

#### Task 3.1: Verify Volume Data Availability
**File**: `src/common/models.py`
**Description**: Ensure volume data is properly handled in Option model
**Components**:
- Verify volume field is properly handled in Option class
- Add volume validation helper methods if needed
- Ensure volume data is preserved in serialization/deserialization

#### Task 3.2: Update Data Retriever with Volume Data Fallback
**File**: `src/common/data_retriever.py`
**Description**: Ensure volume data is properly loaded from cache with API fallback
**Components**:
- Verify volume data is preserved in data loading
- Add volume data validation in data loading process
- Implement API fallback when volume data is missing from cache
- Add error handling and logging for API failures
- Update cache with fresh volume data when available

### Phase 4: Testing and Validation

#### Task 4.1: Create Volume Validation Tests
**File**: `tests/test_volume_validator.py`
**Description**: Create comprehensive tests for volume validation
**Components**:
- Test volume validation logic
- Test edge cases (missing volume data, zero volume, etc.)
- Test integration with BacktestEngine._add_position()

#### Task 4.2: Create Integration Tests
**File**: `tests/test_volume_backtest_integration.py`
**Description**: Test volume validation in full backtest scenarios
**Components**:
- Test volume validation in credit spread strategies
- Test volume statistics reporting
- Test volume configuration options

### Phase 5: Documentation and Configuration

#### Task 5.1: Update Documentation
**File**: `README.md`
**Description**: Document the new volume validation feature
**Components**:
- Add volume validation section to README
- Document configuration options
- Add usage examples

#### Task 5.2: Create Volume Configuration Guide
**File**: `docs/volume_validation_guide.md`
**Description**: Create detailed guide for volume validation configuration
**Components**:
- Configuration options and defaults
- Best practices for volume thresholds
- Troubleshooting guide

## Implementation Details

### Volume Validation in BacktestEngine._add_position()
```python
def _add_position(self, position: Position, current_date: datetime = None):
    """
    Add a position to the positions list with volume validation.
    """
    
    # Volume validation - check all options in the spread
    if self.volume_config.enable_volume_validation and position.spread_options:
        for option in position.spread_options:
            if not self._validate_option_volume(option, current_date or position.entry_date):
                print(f"⚠️  Volume validation failed: {option.symbol} has insufficient volume")
                self.volume_stats = self.volume_stats.increment_rejected_positions()
                return  # Reject the position
    
    # Existing position addition logic...
    position_size = self._get_position_size(position)
    if position_size == 0:
        print(f"⚠️  Warning: Not enough capital to add position. Position size is 0.")
        return
    
    position.set_quantity(position_size)
    # ... rest of existing logic
```

### Volume Statistics Tracking
```python
@dataclass(frozen=True)
class VolumeStats:
    """DTO for tracking volume validation statistics"""
    positions_rejected_volume: int = 0
    options_checked: int = 0
    api_fetch_failures: int = 0
    api_errors: int = 0
    cache_updates: int = 0
    
    def increment_rejected_positions(self) -> 'VolumeStats':
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume + 1,
            options_checked=self.options_checked,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates
        )
    
    def increment_options_checked(self) -> 'VolumeStats':
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked + 1,
            api_fetch_failures=self.api_fetch_failures,
            api_errors=self.api_errors,
            cache_updates=self.cache_updates
        )

@dataclass(frozen=True)
class VolumeConfig:
    """DTO for volume validation configuration"""
    min_volume: int = 10
    enable_volume_validation: bool = True
    
    def __post_init__(self):
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000,
                 start_date: datetime = datetime.now(),
                 end_date: datetime = datetime.now(),
                 max_position_size: float = None,
                 volume_config: VolumeConfig = None):
        # ... existing initialization
        self.volume_config = volume_config or VolumeConfig()
        self.volume_stats = VolumeStats()
```

### Volume Validation Helper Method
```python
def _validate_option_volume(self, option: Option, date: datetime) -> bool:
    """
    Validate if an option has sufficient volume for trading with API fallback.
    """
    self.volume_stats = self.volume_stats.increment_options_checked()
    
    # Get volume data with fallback to API if needed
    volume = self._get_volume_data_with_fallback(option, date)
    
    if volume is None:
        return False
    return volume >= self.volume_config.min_volume
```

### Volume Data Fallback Implementation
```python
def _get_volume_data_with_fallback(self, option: Option, date: datetime) -> Optional[int]:
    """
    Get volume data with fallback to API if not in cache.
    
    Args:
        option: The option to get volume data for
        date: The date for the data
        
    Returns:
        Optional[int]: Volume data if available, None otherwise
    """
    # First try to get from cache
    if option.volume is not None:
        return option.volume
    
    # If volume is missing, try to fetch from API once
    try:
        print(f"📡 Fetching fresh volume data for {option.symbol} on {date.strftime('%Y-%m-%d')}")
        
        # Fetch fresh data from API (get_specific_option_contract automatically updates cache)
        fresh_option_data = self.options_handler.get_specific_option_contract(
            option.strike, 
            option.expiration, 
            option.option_type.value, 
            date
        )
        
        if fresh_option_data and fresh_option_data.volume is not None:
            print(f"✅ Successfully fetched volume data: {fresh_option_data.volume}")
            self.volume_stats = VolumeStats(
                positions_rejected_volume=self.volume_stats.positions_rejected_volume,
                options_checked=self.volume_stats.options_checked,
                api_fetch_failures=self.volume_stats.api_fetch_failures,
                api_errors=self.volume_stats.api_errors,
                cache_updates=self.volume_stats.cache_updates + 1
            )
            return fresh_option_data.volume
        else:
            print(f"⚠️  No volume data available from API for {option.symbol}")
            self.volume_stats = VolumeStats(
                positions_rejected_volume=self.volume_stats.positions_rejected_volume,
                options_checked=self.volume_stats.options_checked,
                api_fetch_failures=self.volume_stats.api_fetch_failures + 1,
                api_errors=self.volume_stats.api_errors,
                cache_updates=self.volume_stats.cache_updates
            )
            return None
            
    except Exception as e:
        print(f"❌ Error fetching volume data from API for {option.symbol}: {e}")
        self.volume_stats = VolumeStats(
            positions_rejected_volume=self.volume_stats.positions_rejected_volume,
            options_checked=self.volume_stats.options_checked,
            api_fetch_failures=self.volume_stats.api_fetch_failures,
            api_errors=self.volume_stats.api_errors + 1,
            cache_updates=self.volume_stats.cache_updates
        )
        return None
```

## Success Criteria

1. **Centralized Validation**: Volume validation occurs only in `BacktestEngine._add_position()`
2. **Strategy Independence**: No changes required to individual strategies
3. **Volume Validation**: All option trades are validated for minimum volume (>= 10)
4. **Statistics Tracking**: Volume rejection statistics are tracked and reported
5. **Configuration**: Volume thresholds are configurable via BacktestEngine constructor
6. **Backward Compatibility**: Existing backtests continue to work with volume validation disabled
7. **Performance**: Volume validation adds minimal overhead to backtest execution
8. **Logging**: Clear logging of volume validation decisions for debugging

## Risk Mitigation

1. **Data Quality**: Handle missing or invalid volume data gracefully
2. **Performance**: Optimize volume validation to minimize backtest execution time
3. **Configuration**: Provide sensible defaults and clear configuration options
4. **Testing**: Comprehensive testing to ensure volume validation works correctly
5. **Documentation**: Clear documentation for users to understand and configure volume validation

## Timeline Estimate

- **Phase 1**: 1-2 days (Core infrastructure)
- **Phase 2**: 2-3 days (Backtest engine integration)
- **Phase 3**: 1 day (Data integration)
- **Phase 4**: 2-3 days (Testing)
- **Phase 5**: 1-2 days (Documentation)

**Total Estimated Time**: 7-11 days (reduced from 9-15 days due to simplified approach)

## Compliance with .cursor/rules

### ✅ **Fully Compliant Areas:**

1. **Agentic Standards**:
   - ✅ No deprecated functions in the plan
   - ✅ All new functions will have unit tests (Task 4.1 and 4.2)
   - ✅ Uses existing infrastructure (leverages `get_specific_option_contract`)

2. **DTO Rules**:
   - ✅ Uses existing `Option` DTO properly
   - ✅ Creates proper DTOs (`VolumeConfig`, `VolumeStats`) instead of Dicts
   - ✅ Maintains separation of concerns (validation logic separate from data transfer)
   - ✅ Uses proper naming conventions and validation
   - ✅ No mixing of business logic with data transfer

3. **VO Rules**:
   - ✅ Uses existing `Option` Value Object correctly
   - ✅ Maintains immutability principles with frozen dataclasses
   - ✅ Uses domain-specific validation in `__post_init__`
   - ✅ Proper error handling with meaningful messages
   - ✅ Value-based equality and immutability

4. **Project Structure**:
   - ✅ Follows existing package structure (`src/backtest/`, `src/common/`)
   - ✅ Uses proper file organization and naming
   - ✅ Maintains single responsibility per directory
   - ✅ Follows existing import patterns and dependencies

### **Key Compliance Features:**

1. **Immutable DTOs**: `VolumeConfig` and `VolumeStats` use `@dataclass(frozen=True)`
2. **Proper Validation**: Configuration validation in `__post_init__` methods
3. **Domain-Specific Names**: Clear, descriptive names following domain language
4. **Separation of Concerns**: Volume validation logic is centralized and separate from data transfer
5. **Comprehensive Testing**: Dedicated test tasks for all new functionality
6. **Existing Infrastructure**: Leverages proven cache update mechanisms
