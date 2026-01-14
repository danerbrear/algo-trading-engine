# Public API Refactoring: Engine-Centric Data Fetching

## Overview

This feature refactors the package to expose only a minimal, clean public API consisting of:
- `BacktestEngine` - For backtesting strategies
- `PaperTradingEngine` - For paper trading strategies  
- `models` - Configuration and metrics DTOs/VOs

All data fetching logic will be internalized within the engine classes, eliminating the need for child projects to handle data retrieval, caching, or API interactions.

## Goals

1. **Minimal Public API**: Only expose `BacktestEngine`, `PaperTradingEngine`, and `models` modules
2. **Self-Contained Engines**: Engines handle all data fetching internally
3. **Simplified Child Projects**: Child projects only need to provide strategy implementations
4. **Better Encapsulation**: Hide internal implementation details (DataRetriever, OptionsHandler, etc.)
5. **Backward Compatibility**: Maintain existing functionality while simplifying the interface

## Current State

### Public API (Too Exposed)
Currently, child projects need to:
- Import and use `DataRetriever` directly
- Manage `OptionsHandler` instances
- Handle data fetching, caching, and API interactions
- Understand internal package structure (`common`, `backtest`, etc.)

### Example Current Usage
```python
from algo_trading_engine.core import BacktestEngine
from algo_trading_engine.common.data_retriever import DataRetriever
from algo_trading_engine.common.options_handler import OptionsHandler
from algo_trading_engine.backtest.strategy_builder import create_strategy_from_args
from algo_trading_engine.models import BacktestConfig

# Child project must handle data fetching
retriever = DataRetriever(symbol="SPY", lstm_start_date="2024-01-01")
data = retriever.fetch_data_for_period(start_date, 'backtest')

# Child project must handle options
options_handler = OptionsHandler("SPY")
strategy = create_strategy_from_args(
    strategy_name="credit_spread",
    symbol="SPY",
    options_handler=options_handler
)
strategy.set_data(data, retriever.treasury_rates)

# Finally create engine
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=100000,
    start_date=start_date,
    end_date=end_date
)
```

## Target State

### Public API (Minimal)
Child projects only need:
- `BacktestEngine` or `PaperTradingEngine`
- `models` (for configuration and metrics)
- Strategy implementation (from child project)

### Example Target Usage
```python
from algo_trading_engine.core import BacktestEngine
from algo_trading_engine.models import BacktestConfig
from my_strategies import MyCustomStrategy

# Simple configuration - engine handles everything
config = BacktestConfig(
    symbol="SPY",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=100000,
    strategy_type="credit_spread",  # or strategy instance
    max_position_size=0.40
)

# Engine handles all data fetching internally
engine = BacktestEngine.from_config(config)

# Run backtest
success = engine.run()

# Get results
metrics = engine.get_performance_metrics()
```

## Implementation Plan

### Phase 1: Refactor BacktestEngine Constructor

#### 1.1 Create Factory Method
- Add `BacktestEngine.from_config()` factory method
- Accepts `BacktestConfig` DTO with all necessary parameters
- Handles data fetching internally

**New BacktestConfig Structure:**
```python
@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    strategy_type: str  # or Strategy instance
    max_position_size: Optional[float] = None
    volume_config: Optional[VolumeConfig] = None
    enable_progress_tracking: bool = True
    quiet_mode: bool = True
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False  # For API rate limiting
    lstm_start_date_offset: int = 120  # Days before start_date for LSTM
```

#### 1.2 Internalize Data Fetching
- Move `DataRetriever` instantiation inside `BacktestEngine.from_config()`
- Move `OptionsHandler` instantiation inside engine
- Move `strategy.set_data()` call inside engine initialization
- Handle all caching, API calls, and data preparation internally

**Implementation:**
```python
@classmethod
def from_config(cls, config: BacktestConfig) -> 'BacktestEngine':
    """
    Create BacktestEngine from configuration.
    
    Handles all data fetching, strategy creation, and setup internally.
    Child projects only need to provide configuration.
    """
    # Internal: Create data retriever
    lstm_start_date = (config.start_date - timedelta(days=config.lstm_start_date_offset))
    retriever = DataRetriever(
        symbol=config.symbol,
        lstm_start_date=lstm_start_date.strftime("%Y-%m-%d"),
        quiet_mode=config.quiet_mode,
        use_free_tier=config.use_free_tier
    )
    
    # Internal: Fetch data
    data = retriever.fetch_data_for_period(
        config.start_date.strftime("%Y-%m-%d"),
        'backtest'
    )
    
    # Internal: Create options handler
    options_handler = OptionsHandler(
        symbol=config.symbol,
        api_key=config.api_key,
        use_free_tier=config.use_free_tier
    )
    
    # Internal: Create strategy
    if isinstance(config.strategy_type, str):
        strategy = create_strategy_from_args(
            strategy_name=config.strategy_type,
            symbol=config.symbol,
            options_handler=options_handler
        )
    else:
        # Strategy instance provided
        strategy = config.strategy_type
        strategy.options_handler = options_handler
    
    # Internal: Set data on strategy
    strategy.set_data(data, retriever.treasury_rates)
    
    # Create and return engine
    return cls(
        data=data,
        strategy=strategy,
        initial_capital=config.initial_capital,
        start_date=config.start_date,
        end_date=config.end_date,
        max_position_size=config.max_position_size,
        volume_config=config.volume_config,
        enable_progress_tracking=config.enable_progress_tracking,
        quiet_mode=config.quiet_mode
    )
```

#### 1.3 Maintain Backward Compatibility
- Keep existing constructor for internal use
- Mark as `@internal` or move to private module
- Existing code continues to work

### Phase 2: Refactor PaperTradingEngine

#### 2.1 Similar Factory Method
- Add `PaperTradingEngine.from_config()` factory method
- Accepts `PaperTradingConfig` DTO
- Handles live data fetching internally

**PaperTradingConfig Structure:**
```python
@dataclass(frozen=True)
class PaperTradingConfig:
    symbol: str
    initial_capital: float
    strategy_type: str  # or Strategy instance
    max_position_size: Optional[float] = None
    volume_config: Optional[VolumeConfig] = None
    execution_delay_seconds: float = 0.0
    api_key: Optional[str] = None  # Polygon.io API key (falls back to POLYGON_API_KEY env var)
    use_free_tier: bool = False
```

#### 2.2 Internalize Live Data Fetching
- Create `DataRetriever` internally for live data
- Create `OptionsHandler` internally
- Handle real-time data updates internally

### Phase 3: Restrict Public API

#### 3.1 Update `__init__.py` Files
- `src/algo_trading_engine/__init__.py`: Only export engines and models
- `src/algo_trading_engine/core/__init__.py`: Keep current exports
- Hide internal modules: `common`, `backtest`, `prediction`, etc.

**New Root `__init__.py`:**
```python
"""
Algo Trading Engine - Public API

This package provides engines for backtesting and paper trading.
"""

# Public API: Engines
from algo_trading_engine.core import BacktestEngine
from algo_trading_engine.core import PaperTradingEngine

# Public API: Models (configs and metrics)
from algo_trading_engine.models import (
    BacktestConfig,
    PaperTradingConfig,
    VolumeConfig,
    PerformanceMetrics,
    PositionStats,
    StrategyPerformanceStats,
    OverallPerformanceStats
)

# Public API: Strategy base class (for child projects to implement)
from algo_trading_engine.core import Strategy

__all__ = [
    # Engines
    'BacktestEngine',
    'PaperTradingEngine',
    # Models
    'BacktestConfig',
    'PaperTradingConfig',
    'VolumeConfig',
    'PerformanceMetrics',
    'PositionStats',
    'StrategyPerformanceStats',
    'OverallPerformanceStats',
    # Strategy base
    'Strategy',
]
```

#### 3.2 Mark Internal Modules
- Add `__all__` restrictions to internal modules
- Document which modules are internal-only
- Consider using `_internal` prefix or separate package

### Phase 4: Update Documentation

#### 4.1 Update Usage Examples
- Update `PACKAGE_INTERFACE_USAGE.md` with new simplified examples
- Remove examples showing internal API usage
- Focus on engine-centric usage patterns

#### 4.2 Migration Guide
- Document how to migrate from old API to new API
- Provide before/after examples
- List deprecated patterns

### Phase 5: Testing

#### 5.1 Engine Factory Tests
- Test `BacktestEngine.from_config()` with various configs
- Test `PaperTradingEngine.from_config()`
- Test error handling for invalid configs

#### 5.2 Integration Tests
- Test full backtest flow using only public API
- Test that internal modules are not accessible
- Test backward compatibility with old constructor

#### 5.3 Child Project Simulation
- Create example child project using only public API
- Verify no internal imports are needed
- Test strategy implementation patterns

## Design Decisions

### 1. Factory Methods vs Constructor
**Decision**: Use factory methods (`from_config()`) as primary API, keep constructor for internal use.

**Rationale**:
- Factory methods provide clear, single-purpose initialization
- Constructor can remain for backward compatibility
- Factory methods can handle complex setup logic

### 2. Strategy Creation
**Decision**: Support both string-based strategy names and Strategy instances.

**Rationale**:
- String names for built-in strategies (simpler)
- Strategy instances for custom strategies from child projects (flexible)

### 3. Data Fetching Location
**Decision**: All data fetching happens in engine initialization.

**Rationale**:
- Simplifies child project code
- Centralizes data management
- Easier to optimize and cache

### 4. Configuration Objects
**Decision**: Use frozen dataclasses for all configuration.

**Rationale**:
- Immutable configuration prevents bugs
- Clear, type-hinted API
- Easy to validate

## Migration Path

### For Existing Code (Internal)
- Continue using existing constructors
- Gradually migrate to factory methods
- No breaking changes initially

### For Child Projects
1. Update imports to use only public API
2. Replace data fetching code with `BacktestEngine.from_config()`
3. Remove `DataRetriever` and `OptionsHandler` usage
4. Simplify strategy creation

## Success Criteria

1. ✅ Child projects can use package with only 3 imports:
   - `from algo_trading_engine.core import BacktestEngine`
   - `from algo_trading_engine.models import BacktestConfig, PerformanceMetrics`
   - Strategy implementation from child project

2. ✅ No data fetching code in child projects
   - All `DataRetriever` usage removed
   - All `OptionsHandler` usage removed
   - All data preparation handled by engine

3. ✅ Internal modules are not accessible
   - Import errors for `algo_trading_engine.common.*`
   - Import errors for `algo_trading_engine.backtest.*` (except via engine)
   - Clear public API boundaries

4. ✅ Backward compatibility maintained
   - Existing internal code continues to work
   - Old constructor still available (marked internal)
   - Gradual migration path available

5. ✅ Documentation updated
   - Usage examples show only public API
   - Migration guide available
   - Clear API boundaries documented

## Implementation Checklist

### Phase 1: BacktestEngine Refactoring
- [ ] Extend `BacktestConfig` with all necessary fields
- [ ] Implement `BacktestEngine.from_config()` factory method
- [ ] Move `DataRetriever` creation inside factory
- [ ] Move `OptionsHandler` creation inside factory
- [ ] Move strategy creation inside factory
- [ ] Move `strategy.set_data()` call inside factory
- [ ] Add unit tests for factory method
- [ ] Add integration tests for factory method

### Phase 2: PaperTradingEngine Refactoring
- [ ] Extend `PaperTradingConfig` with all necessary fields
- [ ] Implement `PaperTradingEngine.from_config()` factory method
- [ ] Move live data fetching inside factory
- [ ] Add unit tests for factory method

### Phase 3: Public API Restriction
- [ ] Update root `__init__.py` to only export public API
- [ ] Add `__all__` to internal modules
- [ ] Test that internal modules are not importable
- [ ] Update type stubs if needed

### Phase 4: Documentation
- [ ] Update `PACKAGE_INTERFACE_USAGE.md`
- [ ] Create migration guide
- [ ] Update README with new usage examples
- [ ] Document public API boundaries

### Phase 5: Testing
- [ ] Test factory methods with various configs
- [ ] Test error handling
- [ ] Test backward compatibility
- [ ] Create example child project
- [ ] Integration tests for full workflow

## Future Enhancements

1. **Strategy Registry**: Built-in strategy registry for string-based strategy names
2. **Data Caching**: Automatic caching of fetched data within engine
3. **Async Support**: Async data fetching for paper trading
4. **Event System**: Engine events for monitoring and logging
5. **Configuration Validation**: Pydantic-based validation for configs

## Risks and Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Maintain backward compatibility, gradual migration path

### Risk 2: Performance Impact
**Mitigation**: Profile data fetching, optimize caching, lazy loading where possible

### Risk 3: Flexibility Loss
**Mitigation**: Allow Strategy instances for custom implementations, keep constructor available internally

### Risk 4: Testing Complexity
**Mitigation**: Comprehensive test coverage, example child project for validation

## Timeline Estimate

- **Phase 1**: 2-3 days (BacktestEngine refactoring)
- **Phase 2**: 1-2 days (PaperTradingEngine refactoring)
- **Phase 3**: 1 day (Public API restriction)
- **Phase 4**: 1 day (Documentation)
- **Phase 5**: 2 days (Testing and validation)

**Total**: ~7-9 days

## Notes

- This refactoring significantly simplifies the API for child projects
- Internal code can continue using existing patterns
- Migration is optional but recommended for cleaner code
- All existing functionality is preserved

