# Volume Trading Backtest Feature Implementation Plan

## Feature Description
Prevent trades when volume is less than 10 for an option contract when backtesting any strategy.

## Overview
This feature will add volume validation to the backtesting system to ensure that only options with sufficient liquidity (volume >= 10) are considered for trading. This will improve the realism of backtests by avoiding illiquid options that would be difficult to trade in real market conditions.

**Key Design Principle**: 
1. **Strategy Responsibility**: Strategies handle data fetching and ensure volume data is available in options
2. **BacktestEngine Validation**: BacktestEngine validates volume requirements before adding positions
3. **Separation of Concerns**: Data fetching is handled by Strategy, validation by BacktestEngine

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
- Validate all options in spread_options list (assumes data is already fetched by Strategy)
- Log volume validation decisions
- Track volume rejection statistics
- Return early if volume validation fails

#### Task 2.3: Add Volume Statistics and Reporting
**File**: `src/backtest/main.py`
**Description**: Add volume-related statistics to backtest results
**Components**:
- Track rejected trades due to volume
- Track options checked for volume validation
- Add volume validation summary in `_end()` method
- Simplified statistics (no API tracking since data fetching is handled by Strategy)

### Phase 3: Data Integration and Validation

#### Task 3.1: Verify Volume Data Availability
**File**: `src/common/models.py`
**Description**: Ensure volume data is properly handled in Option model
**Components**:
- Verify volume field is properly handled in Option class
- Add volume validation helper methods if needed
- Ensure volume data is preserved in serialization/deserialization

#### Task 3.2: Update Strategy Data Fetching
**File**: `src/strategies/credit_spread_minimal.py`
**Description**: Ensure strategies fetch volume data when creating positions
**Components**:
- Update strategy to fetch fresh volume data when creating options
- Ensure all options have volume data before creating positions
- Add volume data validation in position creation methods
- Handle API failures gracefully in strategy layer

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
def _add_position(self, position: Position):
    """
    Add a position to the positions list with volume validation.
    """
    
    # Volume validation - check all options in the spread
    if self.volume_config.enable_volume_validation and position.spread_options:
        for option in position.spread_options:
            if not self._validate_option_volume(option):
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
    
    def increment_rejected_positions(self) -> 'VolumeStats':
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume + 1,
            options_checked=self.options_checked
        )
    
    def increment_options_checked(self) -> 'VolumeStats':
        return VolumeStats(
            positions_rejected_volume=self.positions_rejected_volume,
            options_checked=self.options_checked + 1
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
def _validate_option_volume(self, option: Option) -> bool:
    """
    Validate if an option has sufficient volume for trading.
    
    Note: Data fetching is handled by the Strategy class. This method
    only validates the volume data that is already present in the option.
    """
    self.volume_stats = self.volume_stats.increment_options_checked()
    
    if option.volume is None:
        return False
    return option.volume >= self.volume_config.min_volume
```

### Strategy Data Fetching Example
```python
def _create_call_credit_spread_from_chain(self, date: datetime, prediction: dict) -> Position:
    """
    Create a call credit spread position with volume data validation.
    """
    # ... existing logic to find options ...
    
    # Ensure volume data is available for both options
    atm_option = self._ensure_volume_data(atm_option, date)
    otm_option = self._ensure_volume_data(otm_option, date)
    
    if atm_option is None or otm_option is None:
        print(f"⚠️  Could not fetch volume data for options")
        return None
    
    # Create position with validated options
    position = Position(
        symbol=self.symbol,
        expiration_date=expiration_date,
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        strike_price=atm_strike,
        entry_date=date,
        entry_price=entry_price,
        spread_options=[atm_option, otm_option]
    )
    
    return position

def _ensure_volume_data(self, option: Option, date: datetime) -> Option:
    """
    Ensure option has volume data, fetch if missing.
    
    Args:
        option: The option to check/fetch volume data for
        date: The date for the data
        
    Returns:
        Option: The option with volume data, or None if unable to fetch
    """
    if option.volume is not None:
        return option
    
    # Fetch fresh data from API
    fresh_option = self.options_handler.get_specific_option_contract(
        option.strike, 
        option.expiration, 
        option.option_type.value, 
        date
    )
    
    if fresh_option and fresh_option.volume is not None:
        print(f"📡 Fetched volume data for {option.symbol}: {fresh_option.volume}")
        return fresh_option
    else:
        print(f"⚠️  No volume data available for {option.symbol}")
        return None
```

## Success Criteria

1. **Separation of Concerns**: Data fetching handled by Strategy, validation by BacktestEngine
2. **Strategy Responsibility**: Strategies ensure volume data is available before creating positions
3. **BacktestEngine Validation**: BacktestEngine validates volume requirements before adding positions
4. **Volume Validation**: All option trades are validated for minimum volume (>= 10)
5. **Statistics Tracking**: Volume rejection statistics are tracked and reported
6. **Configuration**: Volume thresholds are configurable via BacktestEngine constructor
7. **Backward Compatibility**: Existing backtests continue to work with volume validation disabled
8. **Performance**: Volume validation adds minimal overhead to backtest execution
9. **Logging**: Clear logging of volume validation decisions for debugging

## Risk Mitigation

1. **Data Quality**: Handle missing or invalid volume data gracefully
2. **Performance**: Optimize volume validation to minimize backtest execution time
3. **Configuration**: Provide sensible defaults and clear configuration options
4. **Testing**: Comprehensive testing to ensure volume validation works correctly
5. **Documentation**: Clear documentation for users to understand and configure volume validation

## Timeline Estimate

- **Phase 1**: ✅ **COMPLETED** (Core infrastructure)
- **Phase 2**: ✅ **COMPLETED** (Backtest engine integration)
- **Phase 3**: ✅ **COMPLETED** (Data integration)
- **Phase 4**: ✅ **COMPLETED** (Testing)
- **Phase 5**: ✅ **COMPLETED** (Documentation)

**Total Estimated Time**: 7-11 days (reduced from 9-15 days due to simplified approach)
**Current Status**: ✅ **ALL TASKS COMPLETED** - Feature fully implemented and documented

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
7. **Simplified Statistics**: Removed API tracking since data fetching is handled by Strategy
