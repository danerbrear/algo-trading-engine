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
- [ ] Implement `get_contract_list_for_date()` with filtering
- [ ] Implement `get_option_bar()` with proper error handling
- [ ] Add comprehensive error handling and retry logic
- [ ] Implement rate limiting through existing APIRetryHandler

### Phase 4: Helper Classes
- [ ] Create OptionsRetrieverHelper with static methods
- [ ] Implement common filtering and calculation utilities
- [ ] Add helper methods for strategy-specific needs

### Phase 5: Testing and Documentation
- [ ] Create comprehensive integration tests for new API
- [ ] Performance testing and optimization
- [ ] Documentation and examples for new API
- [ ] Error message testing for old API usage

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
