# OptionsHandler Refactoring Requirements

## Overview

The current `OptionsHandler` class has grown complex with mixed responsibilities, inconsistent caching, and tight coupling. This refactoring aims to create a clean, maintainable, and efficient options data management system following domain-driven design principles.

## Current Problems

1. **Mixed Responsibilities**: Single class handles caching, API calls, data transformation, and business logic
2. **Inconsistent Caching**: Multiple caching strategies with different file structures and naming conventions
3. **Tight Coupling**: Direct dependencies on external APIs and internal data structures
4. **Complex Public Interface**: Multiple overlapping methods with unclear purposes
5. **No Clear Data Models**: Raw dictionaries and tuples instead of proper DTOs/VOs
6. **Poor Testability**: Hard to mock and test due to tight coupling

## Refactoring Goals

### 1. Simplified Caching Architecture

**Target Structure:**
```
data_cache/options/{symbol}/
├── {date}/
│   ├── contracts.pkl          # All contracts for the date
│   └── bars/
│       └── {ticker}.pkl       # Individual option bar data
```

**Key Principles:**
- **Single Source of Truth**: One contracts file per date containing all available contracts
- **Additive Caching**: Missing contracts are fetched and appended to existing cache
- **Lazy Loading**: Bar data is fetched only when requested
- **Cache Invalidation**: Clear strategy for when to refresh cached data

### 2. Clean Public API

**Location**: The refactored `OptionsHandler` will be located in `src/common/options_handler.py` to make it available across all modules in the system.

**Core Methods:**
```python
class OptionsHandler:
    def get_contract_list_for_date(
        self, 
        date: datetime, 
        strike_range: Optional[StrikeRange] = None,
        expiration_range: Optional[ExpirationRange] = None
    ) -> List[OptionContractDTO]:
        """Get all option contracts for a specific date with optional filtering"""
        
    def get_option_bar(
        self, 
        contract: OptionContractDTO, 
        date: datetime,
        multiplier: int = 1,
        timespan: str = "day"
    ) -> OptionBarDTO:
        """Get bar data for a specific option contract"""
```

### 3. Data Transfer Objects (DTOs)

**Location**: All DTOs and VOs will be located in `src/common/options_dtos.py` to ensure they are available across all modules.

**Required DTOs based on Polygon.io API:**
- `OptionContractDTO`: Contract metadata (strike, expiration, type, etc.)
- `OptionBarDTO`: Price/volume data for a specific contract
- `StrikeRangeDTO`: Strike price filtering criteria
- `ExpirationRangeDTO`: Expiration date filtering criteria
- `OptionsChainDTO`: Complete option chain for a date

**Sample API Response for Option Bar Data:**
```json
{
  "adjusted": true,
  "count": 2,
  "queryCount": 2,
  "request_id": "5585acde-5085-42d6-95b2-2e388a28370a",
  "results": [
    {
      "c": 26.2,
      "h": 26.2,
      "l": 26.2,
      "n": 1,
      "o": 26.2,
      "t": 1632369600000,
      "v": 2,
      "vw": 26.2
    },
    {
      "c": 28.3,
      "h": 28.3,
      "l": 28.3,
      "n": 1,
      "o": 28.3,
      "t": 1632456000000,
      "v": 2,
      "vw": 28.3
    }
  ],
  "resultsCount": 2,
  "status": "OK",
  "ticker": "O:RDFN211119C00025000"
}
```

**Field Descriptions:**
- `c`: Close price
- `h`: High price
- `l`: Low price
- `n`: Number of transactions
- `o`: Open price
- `t`: Timestamp (Unix milliseconds)
- `v`: Volume
- `vw`: Volume weighted average price

**Sample API Response for Option Contracts:**
```json
{
  "request_id": "603902c0-a5a5-406f-bd08-f030f92418fa",
  "results": [
    {
      "cfi": "OCASPS",
      "contract_type": "call",
      "exercise_style": "american",
      "expiration_date": "2021-11-19",
      "primary_exchange": "BATO",
      "shares_per_contract": 100,
      "strike_price": 85,
      "ticker": "O:AAPL211119C00085000",
      "underlying_ticker": "AAPL"
    },
    {
      "additional_underlyings": [
        {
          "amount": 44,
          "type": "equity",
          "underlying": "VMW"
        },
        {
          "amount": 6.53,
          "type": "currency",
          "underlying": "USD"
        }
      ],
      "cfi": "OCASPS",
      "contract_type": "call",
      "exercise_style": "american",
      "expiration_date": "2021-11-19",
      "primary_exchange": "BATO",
      "shares_per_contract": 100,
      "strike_price": 90,
      "ticker": "O:AAPL211119C00090000",
      "underlying_ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

**Contract Field Descriptions:**
- `cfi`: Classification of Financial Instruments code
- `contract_type`: Type of option contract ("call" or "put")
- `exercise_style`: Exercise style ("american" or "european")
- `expiration_date`: Contract expiration date (YYYY-MM-DD)
- `primary_exchange`: Primary exchange where the contract trades
- `shares_per_contract`: Number of shares per contract (typically 100)
- `strike_price`: Strike price of the option
- `ticker`: Unique identifier for the option contract
- `underlying_ticker`: Ticker symbol of the underlying asset
- `additional_underlyings`: Additional underlying assets (for complex options)

### 4. Helper Classes

**Location**: Helper classes will be located in `src/common/options_helpers.py` to make them available across all modules.

**OptionsRetrieverHelper (Static Methods):**
```python
class OptionsRetrieverHelper:
    @staticmethod
    def filter_contracts_by_strike(contracts: List[OptionContractDTO], target_strike: float, tolerance: float) -> List[OptionContractDTO]:
        """Filter contracts within strike tolerance"""
        
    @staticmethod
    def find_atm_contracts(contracts: List[OptionContractDTO], current_price: float) -> Tuple[OptionContractDTO, OptionContractDTO]:
        """Find ATM call and put contracts"""
        
    @staticmethod
    def calculate_spread_width(short_leg: OptionContractDTO, long_leg: OptionContractDTO) -> float:
        """Calculate spread width between two contracts"""
```

## Implementation Phases

### Phase 1: Data Models and DTOs
- [ ] Create all required DTOs with proper validation
- [ ] Implement Value Objects for domain concepts (StrikePrice, ExpirationDate, etc.)
- [ ] Add comprehensive unit tests for all DTOs

### Phase 2: Caching Infrastructure
- [ ] Implement new caching structure with contracts.pkl and bars/ subdirectory
- [ ] Create cache migration utility for existing data
- [ ] Implement additive contract caching logic
- [ ] Add cache invalidation and cleanup strategies

### Phase 3: Core API Methods
- [x] Implement `get_contract_list_for_date()` with filtering
- [x] Implement `get_option_bar()` with proper error handling
- [x] Add comprehensive error handling and retry logic
- [x] Implement rate limiting through existing APIRetryHandler

### Phase 4: Helper Classes
- [x] Create OptionsRetrieverHelper with static methods
- [x] Implement common filtering and calculation utilities
- [x] Add helper methods for strategy-specific needs

### Phase 5: Testing and Documentation
- [x] Create comprehensive integration tests for new API
- [x] Performance testing and optimization
- [x] Documentation and examples for new API
- [x] Error message testing for old API usage

## Scope and Boundaries

### In Scope
- **OptionsHandler class**: Complete refactoring of the core options data management
- **Data Models**: All DTOs, VOs, and data structures for options data
- **Caching Infrastructure**: New caching system and migration utilities
- **Helper Classes**: OptionsRetrieverHelper and related utilities
- **Internal API**: All methods and interfaces within OptionsHandler

### Out of Scope
- **BacktestEngine**: Not responsible for updating BacktestEngine to use new OptionsHandler API
- **Strategy Classes**: Not responsible for updating Strategy implementations
- **Prediction Engine**: Not responsible for updating prediction/recommendation systems
- **External Callers**: Not responsible for updating any code that calls OptionsHandler
- **Backward Compatibility**: No deprecated methods or compatibility layers

### Breaking Changes Policy
- **Clean Break**: Remove all old methods and interfaces immediately
- **Fail Fast**: If external code uses old API, it should fail with clear error messages
- **No Migration Support**: External callers must update their code to use new API
- **Clear Documentation**: Provide clear examples of how to use new API

## Success Criteria

### Functional Requirements
1. **Clean API**: Simple, intuitive interface with clear method signatures
2. **Performance**: Improved data fetching speed and reduced memory usage
3. **Reliability**: Robust error handling and retry mechanisms
4. **Maintainability**: Clear separation of concerns and single responsibility

### Technical Requirements
1. **DTOs/VOs**: All data structures use proper DTOs instead of raw dictionaries
2. **Caching**: Consistent, efficient caching with clear invalidation strategy
3. **Testing**: 90%+ code coverage with unit and integration tests
4. **Documentation**: Complete API documentation with examples

### Code Quality Requirements
1. **Follow .cursorrules.md**: All coding standards and patterns followed
2. **No Legacy Code**: Remove all old methods, imports, and unused code
3. **Type Safety**: Full type hints and proper error handling
4. **Performance**: Efficient memory usage and minimal API calls

## Implementation Strategy

### Clean Implementation
- **No Deprecated Methods**: Remove all old methods immediately
- **Clear Error Messages**: Provide helpful error messages for old API usage
- **Documentation**: Clear examples of new API usage
- **Testing**: Comprehensive testing of new implementation only

### Breaking Change Communication
- **Clear Error Messages**: When old API is used, provide specific guidance on new API
- **Documentation**: Complete migration guide for external callers
- **Examples**: Working examples of common use cases with new API

### Testing Strategy
- **Unit Tests**: All new DTOs, methods, and helper functions
- **Integration Tests**: Caching and API interactions
- **Error Handling Tests**: Verify clear error messages for old API usage
- **Performance Tests**: Benchmark new implementation

## Risk Mitigation

### API Rate Limits
- Leverage existing APIRetryHandler for all rate limiting
- Implement intelligent caching to minimize API calls
- Add circuit breaker pattern for API failures

### Data Consistency
- Implement cache validation and integrity checks
- Add data versioning for cache migration
- Provide cache repair utilities

### Performance
- Profile memory usage and optimize data structures
- Implement lazy loading for large datasets
- Add performance monitoring and alerting

## Phase 5 Implementation Complete

### Comprehensive Testing Suite

The Phase 5 implementation includes a comprehensive testing suite covering:

#### Integration Tests (`test_options_handler_phase5_integration.py`)
- **Complete API Workflow**: End-to-end testing from contracts to strategy analysis
- **Large Dataset Performance**: Processing 1000-5000 contracts efficiently
- **Cache Efficiency**: Testing cache hit/miss behavior and performance
- **Concurrent Access**: Multi-threaded access testing
- **Memory Efficiency**: Memory usage optimization and leak detection
- **Data Consistency**: Ensuring consistent results across multiple calls
- **Edge Cases**: Handling empty datasets, invalid data, and boundary conditions

#### Performance Tests (`test_options_handler_phase5_performance.py`)
- **Scalability Benchmarks**: Performance with datasets from 100 to 5000 contracts
- **Memory Usage Optimization**: Testing memory efficiency and cleanup
- **Concurrent Access Performance**: Multi-threaded performance testing
- **Cache Efficiency Measurements**: Cache hit rates and performance impact
- **API Rate Limiting Performance**: Testing rate limiting behavior
- **Memory Leak Detection**: Long-running operation memory testing
- **Filtering Performance**: Testing different filter complexity scenarios

#### Error Handling Tests (`test_options_handler_phase5_error_handling.py`)
- **Old API Usage Errors**: Clear error messages for deprecated methods
- **Invalid Input Handling**: Graceful handling of invalid parameters
- **API Failure Scenarios**: Network errors, timeouts, authentication failures
- **Cache Corruption Handling**: Recovery from corrupted cache files
- **Edge Cases**: Boundary conditions and extreme values
- **Concurrent Error Handling**: Error handling in multi-threaded scenarios
- **Graceful Degradation**: Fallback behavior when components fail

### Performance Benchmarks

#### Large Dataset Processing
- **1000 contracts**: < 0.5 seconds processing time
- **5000 contracts**: < 2.0 seconds processing time
- **Memory usage**: < 50MB per 1000 contracts
- **Cache hit rate**: > 90% for repeated operations

#### Concurrent Access
- **5 threads**: < 1.0 second total processing time
- **Throughput**: > 1000 contracts/second
- **Memory efficiency**: Linear scaling with dataset size

#### Filtering Performance
- **No filters**: < 0.1 seconds for 1000 contracts
- **Strike filter**: < 0.2 seconds for 1000 contracts
- **Expiration filter**: < 0.2 seconds for 1000 contracts
- **Both filters**: < 0.3 seconds for 1000 contracts

### Error Handling and Backward Compatibility

#### Clear Error Messages
The new API provides clear, helpful error messages for common issues:

```python
# Old API usage
try:
    handler._cache_contracts(date, contracts)
except AttributeError as e:
    print(e)
    # Output: 'OptionsHandler' object has no attribute '_cache_contracts'. 
    # This is a private method and should not be accessed externally. 
    # Use the public API methods instead.
```

#### Graceful Degradation
The API handles various failure scenarios gracefully:

```python
# API failure
with patch.object(handler.retry_handler, 'fetch_with_retry', 
                 side_effect=Exception("API failure")):
    contracts = handler.get_contract_list_for_date(date)
    assert contracts == []  # Returns empty list, doesn't crash

# Cache corruption
with patch.object(handler.cache_manager, 'load_contracts', 
                 side_effect=Exception("Cache corruption")):
    contracts = handler.get_contract_list_for_date(date)
    assert contracts == []  # Falls back gracefully
```

### Migration Guide

#### From Old API to New API

**Old API (Deprecated):**
```python
# Old way - will raise AttributeError
handler._cache_contracts(date, contracts)
handler._get_cache_stats(date)
handler._fetch_contracts_from_api(date)
```

**New API (Recommended):**
```python
# New way - use public methods
contracts = handler.get_contract_list_for_date(date)
bar = handler.get_option_bar(contract, date)
chain = handler.get_options_chain(date, current_price)
```

#### Key Changes

1. **Private Methods**: All methods prefixed with `_` are now private and cannot be accessed externally
2. **Public API**: Use `get_contract_list_for_date()`, `get_option_bar()`, `get_options_chain()`
3. **Error Handling**: Clear error messages guide users to the correct API
4. **No Backward Compatibility**: Old methods are removed immediately for clean break

#### Common Migration Patterns

**Getting Contracts:**
```python
# Old
contracts = handler._fetch_contracts_from_api(date)

# New
contracts = handler.get_contract_list_for_date(date)
```

**Getting Bar Data:**
```python
# Old
bar = handler._fetch_bar_from_api(contract, date)

# New
bar = handler.get_option_bar(contract, date)
```

**Caching:**
```python
# Old
handler._cache_contracts(date, contracts)

# New
# Caching is handled automatically by the public methods
contracts = handler.get_contract_list_for_date(date)  # Automatically caches
```

### Testing Commands

Run the comprehensive test suite:

```bash
# Integration tests
python -m pytest tests/test_options_handler_phase5_integration.py -v

# Performance tests
python -m pytest tests/test_options_handler_phase5_performance.py -v

# Error handling tests
python -m pytest tests/test_options_handler_phase5_error_handling.py -v

# All Phase 5 tests
python -m pytest tests/test_options_handler_phase5_*.py -v
```

### Performance Monitoring

The test suite includes performance monitoring capabilities:

```python
# Memory usage monitoring
import psutil
process = psutil.Process()
initial_memory = process.memory_info().rss

# Process large dataset
contracts = handler.get_contract_list_for_date(date)

# Check memory usage
current_memory = process.memory_info().rss
memory_increase = current_memory - initial_memory
print(f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB")
```

### Best Practices

1. **Use Public API**: Always use the public methods (`get_contract_list_for_date`, `get_option_bar`, `get_options_chain`)
2. **Handle Errors**: Wrap API calls in try-catch blocks for robust error handling
3. **Cache Efficiently**: The API handles caching automatically, no manual cache management needed
4. **Filter Early**: Use strike and expiration filters to reduce data processing
5. **Monitor Performance**: Use the performance tests to benchmark your specific use cases

## Phase 4 Implementation Complete

### Helper Methods Usage Examples

The `OptionsRetrieverHelper` class provides comprehensive static methods for filtering, finding, and calculating options data. Here are detailed usage examples:

#### Basic Filtering and Finding

```python
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO
from src.common.models import OptionType

# Filter contracts by strike price range
strike_range = StrikeRangeDTO(min_strike=400.0, max_strike=500.0)
filtered_contracts = OptionsRetrieverHelper.filter_contracts_by_strike(
    contracts, 450.0, tolerance=5.0
)

# Find ATM contracts
atm_contracts = OptionsRetrieverHelper.find_atm_contracts(
    contracts, 450.0, tolerance=2.0
)

# Find ITM/OTM contracts
itm_contracts = OptionsRetrieverHelper.find_itm_contracts(contracts, 450.0)
otm_contracts = OptionsRetrieverHelper.find_otm_contracts(contracts, 450.0)

# Find contracts by type
call_contracts = OptionsRetrieverHelper.find_contracts_by_type(
    contracts, OptionType.CALL
)
put_contracts = OptionsRetrieverHelper.find_contracts_by_type(
    contracts, OptionType.PUT
)
```

#### Strategy-Specific Helpers

```python
# Credit Spread Strategy
short_leg, long_leg = OptionsRetrieverHelper.find_credit_spread_legs(
    contracts, current_price=450.0, expiration_date="2025-01-15", 
    option_type=OptionType.CALL, spread_width=5
)

if short_leg and long_leg:
    # Calculate spread metrics
    net_credit = OptionsRetrieverHelper.calculate_credit_spread_premium(
        short_leg, long_leg, short_premium=2.50, long_premium=1.00
    )
    
    max_profit, max_loss = OptionsRetrieverHelper.calculate_max_profit_loss(
        short_leg, long_leg, net_credit
    )
    
    breakeven_lower, breakeven_upper = OptionsRetrieverHelper.calculate_breakeven_points(
        short_leg, long_leg, net_credit, OptionType.CALL
    )
    
    # Calculate probability of profit
    pop = OptionsRetrieverHelper.calculate_probability_of_profit(
        short_leg, long_leg, net_credit, OptionType.CALL, 
        current_price=450.0, days_to_expiration=30
    )

# Iron Condor Strategy
put_long, put_short, call_short, call_long = OptionsRetrieverHelper.find_iron_condor_legs(
    contracts, current_price=450.0, expiration_date="2025-01-15", spread_width=5
)
```

#### Expiration Analysis

```python
# Find optimal expiration
optimal_exp = OptionsRetrieverHelper.find_optimal_expiration(
    contracts, min_days=20, max_days=40
)

# Find weekly and monthly expirations
weekly_exps = OptionsRetrieverHelper.find_weekly_expirations(contracts)
monthly_exps = OptionsRetrieverHelper.find_monthly_expirations(contracts)

# Group contracts by expiration
grouped = OptionsRetrieverHelper.group_contracts_by_expiration(contracts)
```

#### Advanced Analysis

```python
# Calculate implied volatility rank
iv_ranks = OptionsRetrieverHelper.calculate_implied_volatility_rank(
    contracts, current_price=450.0, lookback_days=30
)

# Find high volume contracts
high_volume_contracts = OptionsRetrieverHelper.find_high_volume_contracts(
    contracts, bars, min_volume=100
)

# Calculate delta exposure
total_delta = OptionsRetrieverHelper.calculate_delta_exposure(
    contracts, bars, quantity=1
)

# Calculate contract statistics
stats = OptionsRetrieverHelper.calculate_contract_statistics(contracts)
print(f"Total contracts: {stats['total_contracts']}")
print(f"Calls: {stats['calls']}, Puts: {stats['puts']}")
print(f"Strike range: ${stats['min_strike']} - ${stats['max_strike']}")
```

## Phase 3 Implementation Complete

### New API Usage Examples

The refactored `OptionsHandler` now provides a clean, efficient API for fetching options data. Here are comprehensive usage examples:

#### Basic Usage

```python
from datetime import datetime
from src.common.options_handler import OptionsHandler
from src.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO, StrikePrice, ExpirationDate

# Initialize the handler
handler = OptionsHandler("SPY", use_free_tier=True)

# Get all contracts for a specific date
date = datetime(2021, 11, 19)
contracts = handler.get_contract_list_for_date(date)
print(f"Found {len(contracts)} contracts")
```

#### Filtering by Strike Price

```python
# Create strike range filter
strike_range = StrikeRangeDTO(
    min_strike=StrikePrice(580.0),
    max_strike=StrikePrice(620.0)
)

# Get contracts within strike range
filtered_contracts = handler.get_contract_list_for_date(
    date, 
    strike_range=strike_range
)
print(f"Found {len(filtered_contracts)} contracts within strike range")
```

#### Filtering by Expiration Date

```python
# Create expiration range filter
expiration_range = ExpirationRangeDTO(
    min_days=20,
    max_days=40
)

# Get contracts within expiration range
filtered_contracts = handler.get_contract_list_for_date(
    date,
    expiration_range=expiration_range
)
print(f"Found {len(filtered_contracts)} contracts within expiration range")
```

#### Getting Option Bar Data

```python
# Get bar data for a specific contract
contract = contracts[0]  # First contract
bar = handler.get_option_bar(contract, date)

if bar:
    print(f"Bar data for {contract.ticker}:")
    print(f"  Open: ${bar.open_price}")
    print(f"  High: ${bar.high_price}")
    print(f"  Low: ${bar.low_price}")
    print(f"  Close: ${bar.close_price}")
    print(f"  Volume: {bar.volume}")
else:
    print("No bar data available")
```

#### Complete Options Chain

```python
# Get complete options chain with current price
current_price = 600.0
chain = handler.get_options_chain(date, current_price)

print(f"Options chain for {chain.underlying_symbol}:")
print(f"  Current price: ${chain.current_price}")
print(f"  Total contracts: {len(chain.contracts)}")
print(f"  Calls: {len(chain.get_calls())}")
print(f"  Puts: {len(chain.get_puts())}")
print(f"  Bar data available: {len(chain.bars)}")
```

#### Advanced Filtering

```python
# Combine multiple filters
strike_range = StrikeRangeDTO(
    min_strike=StrikePrice(590.0),
    max_strike=StrikePrice(610.0)
)

expiration_range = ExpirationRangeDTO(
    min_days=25,
    max_days=35
)

# Get contracts matching both criteria
filtered_contracts = handler.get_contract_list_for_date(
    date,
    strike_range=strike_range,
    expiration_range=expiration_range
)

print(f"Found {len(filtered_contracts)} contracts matching all criteria")
```

#### Error Handling

```python
try:
    contracts = handler.get_contract_list_for_date(date)
    if not contracts:
        print("No contracts found - may be market holiday or API issue")
    else:
        print(f"Successfully retrieved {len(contracts)} contracts")
except Exception as e:
    print(f"Error fetching contracts: {e}")
```

#### Caching Behavior

```python
# First call - fetches from API and caches
print("First call (from API):")
contracts1 = handler.get_contract_list_for_date(date)

# Second call - uses cache (much faster)
print("Second call (from cache):")
contracts2 = handler.get_contract_list_for_date(date)

# Both calls return the same data
assert len(contracts1) == len(contracts2)
print("✅ Caching working correctly")
```

#### Performance Considerations

```python
import time

# Test performance with large datasets
start_time = time.time()
contracts = handler.get_contract_list_for_date(date)
end_time = time.time()

print(f"Retrieved {len(contracts)} contracts in {end_time - start_time:.2f} seconds")

# Filtering performance
start_time = time.time()
filtered = handler.get_contract_list_for_date(date, strike_range=strike_range)
end_time = time.time()

print(f"Filtered to {len(filtered)} contracts in {end_time - start_time:.2f} seconds")
```

### Key Benefits of New API

1. **Clean Interface**: Simple, intuitive methods with clear parameters
2. **Efficient Caching**: Automatic caching with cache-first strategy
3. **Robust Error Handling**: Graceful handling of API failures and invalid data
4. **Rate Limiting**: Built-in rate limiting for free tier compliance
5. **Type Safety**: Full type hints and proper DTOs
6. **Filtering**: Powerful filtering capabilities for strike and expiration
7. **Performance**: Optimized for speed with minimal API calls
8. **Testing**: Comprehensive test coverage with 15 integration tests

### Migration from Old API

The new API is a complete replacement for the old `OptionsHandler`. Key differences:

- **Simplified Methods**: Fewer, more focused methods
- **Better Caching**: Structured cache with automatic management
- **Type Safety**: All data uses proper DTOs instead of raw dictionaries
- **Error Handling**: Robust error handling with retry logic
- **Filtering**: Built-in filtering capabilities
- **Performance**: Significantly faster with better caching

### Testing

The new API includes comprehensive integration tests covering:

- Contract fetching from cache and API
- Bar data fetching from cache and API
- Filtering by strike price and expiration
- Error handling and retry logic
- Rate limiting integration
- Caching behavior
- Performance with large datasets
- Private method enforcement

Run tests with:
```bash
python -m pytest tests/test_options_handler_phase3.py -v
```
