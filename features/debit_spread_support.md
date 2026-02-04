# Debit Spread Support Implementation Plan

**Status:** Phase 0 Complete ✅ | In Progress 🚧  
**Created:** 2026-02-03  
**Last Updated:** 2026-02-03  
**Goal:** Add full debit spread support across the trading engine codebase

---

## Implementation Checklist

### Phase 0: Architectural Refactoring 🏗️ (Foundation)
- [x] **0.1** Move Position class from `backtest/models.py` to `vo/position.py`
- [x] **0.2** Refactor Position to abstract base class with strategy subclasses
  - [x] Create abstract Position base class
  - [x] Implement CreditSpreadPosition
  - [x] Implement DebitSpreadPosition (ready for Phase 1)
  - [x] Implement LongCallPosition
  - [x] Implement ShortCallPosition
  - [x] Implement LongPutPosition
  - [x] Implement ShortPutPosition
  - [x] Removed LongStockPosition (options-only repo)
- [x] **0.3** Create `create_position()` factory function
- [x] **0.4** Update all imports (17 files)
  - [x] Update 7 source files
  - [x] Update 10 test files
- [x] **0.5** Update all Position instantiations to use factory
- [x] **0.6** Run full test suite (586 tests passing ✅)

### Phase 1: Core Data Models & Enums
- [ ] **1.1** Add CALL_DEBIT_SPREAD to StrategyType enum (backtest/models.py)
- [ ] **1.2** Add PUT_DEBIT_SPREAD to StrategyType enum (backtest/models.py)
- [ ] **1.3** Add CALL_DEBIT_SPREAD to StrategyType enum (backtest/config.py)
- [ ] **1.4** Add PUT_DEBIT_SPREAD to StrategyType enum (backtest/config.py)
- [ ] **1.5** Create SpreadResultDTOs (optional but recommended)
  - [ ] CreditSpreadResultDTO
  - [ ] DebitSpreadResultDTO
- [ ] **1.6** Run tests to verify enum additions

### Phase 2: Position Implementation
- [x] **2.1** DebitSpreadPosition class created (done in Phase 0)
- [ ] **2.2** Update factory to handle debit spread types
- [ ] **2.3** Test debit spread P&L calculations
  - [ ] get_return_dollars()
  - [ ] _get_return()
  - [ ] calculate_exit_price()
  - [ ] calculate_exit_price_from_bars()
  - [ ] get_return_dollars_from_assignment()
  - [ ] max_profit()
  - [ ] max_loss()
- [ ] **2.4** Update capital manager for debit spreads
  - [ ] Verify capital allocation works
  - [ ] Test risk calculations

### Phase 3: Options Helper Integration
- [ ] **3.1** Verify find_debit_spread_max_reward_risk() works
- [ ] **3.2** Update return types to DTOs (if doing Phase 1.5)
- [ ] **3.3** Test debit spread selection logic

### Phase 4: Strategy Implementation
- [ ] **4.1** Create debit spread strategy methods
  - [ ] _create_call_debit_spread()
  - [ ] _create_put_debit_spread()
- [ ] **4.2** Add signal generation for debit spreads
- [ ] **4.3** Test strategy integration with backtest engine

### Phase 5: Recommendation Engine & UI
- [ ] **5.1** Update recommendation engine for debit spreads
  - [ ] _position_from_decision()
  - [ ] get_open_positions_status()
  - [ ] Position closing logic
- [ ] **5.2** Update CLI display for debit spreads
- [ ] **5.3** Update plotting for debit spread equity curves
- [ ] **5.4** Test paper trading with debit spreads

### Phase 6: Testing & Documentation
- [ ] **6.1** Add debit spread test cases
  - [ ] position_statistics_test.py
  - [ ] capital_manager_test.py
  - [ ] strategy_enhancements_test.py
- [ ] **6.2** Integration tests
  - [ ] Backtest with debit spreads
  - [ ] Paper trading with debit spreads
  - [ ] Mixed credit/debit spread portfolios
- [ ] **6.3** Update documentation
  - [ ] strategy_builder_guide.md
  - [ ] volume_validation_guide.md (if applicable)
  - [ ] README updates

### Success Criteria
- [x] **SC-1** Position class moved to vo/position.py
- [x] **SC-2** All Position subclasses implemented and tested
- [x] **SC-3** No conditional strategy_type logic remains in Position
- [x] **SC-4** Factory pattern working correctly
- [x] **SC-5** All 586+ tests passing
- [ ] **SC-6** Debit spread enum values added
- [ ] **SC-7** Debit spread P&L matches manual calculations
- [ ] **SC-8** Can execute debit spread trades in paper trading
- [ ] **SC-9** No regressions in existing credit spread functionality

**Current Progress:** Phase 0 Complete (100%) | Overall Progress: ~15%

---

## Implementation Log

### ✅ 2026-02-03: Phase 0 Complete
**Architectural Refactoring Successfully Completed**

**What Was Done:**
- Created new file `src/algo_trading_engine/vo/position.py` with:
  - Abstract `Position` base class
  - 7 concrete position subclasses (options only, including `DebitSpreadPosition`)
  - `create_position()` factory function
- Removed equity/stock strategy support (LONG_STOCK) - this is options-only
- Removed all Position code from `common/models.py`
- Updated `vo/__init__.py` to export all Position classes
- Updated 17 files with new imports:
  - Changed from `algo_trading_engine.common.models` → `algo_trading_engine.vo`
  - 7 source files updated
  - 10 test files updated
- All Position instantiations now use factory pattern
- **Result: All 586 tests passing ✅**

**Benefits Achieved:**
- Eliminated all conditional `strategy_type` logic from Position class
- Each strategy type now encapsulates its own P&L logic
- `DebitSpreadPosition` class ready for Phase 1 enum integration
- Better code organization (Position models in `vo/` directory)
- Follows SOLID principles (Open/Closed Principle)
- Easy to add new strategies without modifying existing code

**Files Changed:** 18 total (1 new, 17 modified)

**Update - Removed Equity Strategies:** Removed `LongStockPosition` and `LONG_STOCK` enum - this is an options-only repository. All 586 tests still passing ✅

**Quick Reference - New Import Pattern:**
```python
# Import Position and factory
from algo_trading_engine.vo import Position, create_position

# Import specific position types
from algo_trading_engine.vo import CreditSpreadPosition, DebitSpreadPosition

# Create positions using factory
position = create_position(
    symbol="SPY",
    expiration_date=datetime(2025, 9, 6),
    strategy_type=StrategyType.PUT_CREDIT_SPREAD,  # or PUT_DEBIT_SPREAD (Phase 1)
    strike_price=500.0,
    entry_date=date,
    entry_price=1.05,
    spread_options=[atm_option, otm_option]
)
```

---

## 1. Executive Summary

This document outlines the plan to add debit spread support to the algo trading engine. Currently, the codebase only supports credit spreads (PUT_CREDIT_SPREAD, CALL_CREDIT_SPREAD). We need to add debit spreads (PUT_DEBIT_SPREAD, CALL_DEBIT_SPREAD) with proper P&L calculations, position management, and strategy integration.

**Key Changes:**

1. **✅ Architectural Refactoring (Phase 0) - COMPLETED:**
   - ✅ Moved `Position` class from `backtest/models.py` to `vo/position.py`
   - ✅ Refactored `Position` to abstract base class with strategy-specific subclasses
   - ✅ Eliminated conditional logic based on `strategy_type`
   - ✅ Improved maintainability and follows Open/Closed Principle
   - ✅ All 586 tests passing

2. **🚧 Debit Spread Support (Phases 1-6) - IN PROGRESS:**
   - [ ] Add debit spread enum values
   - ✅ Implement `DebitSpreadPosition` class (ready for enum values)
   - [ ] Add debit spread strategy methods
   - [ ] Full integration with backtesting and paper trading

**Key Difference:**
- **Credit Spread:** Sell ATM/ITM, Buy OTM → Receive net credit upfront → Profit = Credit - Cost to Close
- **Debit Spread:** Buy ATM/ITM, Sell OTM → Pay net debit upfront → Profit = Exit Value - Debit Paid

**Benefits of New Architecture:**
- ✅ Eliminates conditional strategy_type checks
- ✅ Each strategy type encapsulates its own logic
- ✅ Easy to add new strategies without modifying existing code
- ✅ Better testability and maintainability
- ✅ Follows SOLID principles

---

## 2. Current State Analysis

### 2.1 Existing Credit Spread Infrastructure

**Files with Credit Spread Logic:**
```
src/algo_trading_engine/
├── backtest/models.py              # StrategyType enum
├── vo/position.py                  # ✅ Position classes (moved from backtest)
├── common/models.py                # SignalType enum, Option, OptionChain
├── common/options_helpers.py       # find_credit_spread_max_credit_width()
├── strategies/
│   ├── velocity_signal_momentum_strategy.py  # _create_put_credit_spread()
│   └── credit_spread_minimal.py    # Minimal credit spread strategy
├── prediction/
│   ├── capital_manager.py          # Capital allocation per strategy
│   └── recommendation_engine.py    # Position recommendation logic
└── core/engine.py                  # Trading engine base class
```

**Enums:**
- `StrategyType` (backtest/models.py): CALL_CREDIT_SPREAD, PUT_CREDIT_SPREAD

**Position P&L Calculations:**
- `get_return_dollars()`: Credit received - Cost to close
- `_get_return()`: Percentage return based on credit received
- `profit_target_hit()`, `stop_loss_hit()`: Based on credit spread logic

### 2.2 Recently Added Infrastructure

**New Helper Function:**
- `find_debit_spread_max_reward_risk()` in `options_helpers.py` (lines 937-1079)
  - Already implements debit spread selection logic
  - Returns: `itm_contract`, `otm_contract`, bars, `debit`, `width`, `max_profit`, `max_loss`, `reward_risk_ratio`

---

## 3. Implementation Plan

### Phase 0: Architectural Refactoring 🏗️ (Foundation - Do First)

**Critical:** This phase must be completed before adding debit spread support. It refactors the Position class to use proper OOP principles instead of conditional logic.

#### 0.1 Move Position Class to vo/position.py

**Current Location:** `src/algo_trading_engine/backtest/models.py`  
**New Location:** `src/algo_trading_engine/vo/position.py`

**Rationale:**
- Position is used by both `BacktestEngine` and `PaperTradingEngine`
- It's a core domain model / value object
- Should live in vo with other domain models, separated from common utilities

**Steps:**
1. Move `Position` class from `backtest/models.py` to `vo/position.py`
2. Keep `StrategyType` enum in `backtest/models.py` (for now)
3. Update all imports across the codebase
4. Run all tests to ensure nothing breaks

**✅ COMPLETED** - All Position classes now in `vo/position.py`

**Files to Update Imports:**
```bash
# Find all files importing Position
grep -r "from.*backtest.models import.*Position" src/
grep -r "from.*backtest import.*Position" src/
```

Expected files:
- `src/algo_trading_engine/backtest/main.py`
- `src/algo_trading_engine/core/engine.py`
- `src/algo_trading_engine/prediction/recommendation_engine.py`
- All test files that use Position

**New Import Statement:**
```python
from algo_trading_engine.vo import Position, create_position
# Or for specific types:
from algo_trading_engine.vo import CreditSpreadPosition, DebitSpreadPosition
```

---

#### 0.2 Refactor Position to Abstract Base Class with Strategy Subclasses ✅

**Current Design:** Single `Position` class with conditional logic based on `strategy_type`

**New Design:** Abstract `Position` base class with concrete subclasses for each strategy type

**✅ COMPLETED** - All Position subclasses implemented in `vo/position.py`

**Abstract Methods (methods with conditional strategy_type logic):**
- `get_return_dollars(exit_price: float) -> float`
- `_get_return(exit_price: float) -> float`
- `calculate_exit_price(current_option_chain: OptionChain) -> float`
- `calculate_exit_price_from_bars(atm_bar, otm_bar) -> float`

**Concrete Subclasses (Options Only):**

```python
from abc import ABC, abstractmethod

class Position(ABC):
    """
    Abstract base class for all position types.
    
    Common attributes and non-strategy-specific methods are defined here.
    Strategy-specific P&L and pricing logic is delegated to subclasses.
    """
    
    def __init__(self, symbol: str, expiration_date: datetime, strategy_type: StrategyType, 
                 strike_price: float, entry_date: datetime, entry_price: float, 
                 exit_price: float = None, spread_options: list[Option] = None):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.quantity = None
        self.strategy_type = strategy_type
        self.strike_price = strike_price
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.spread_options: list[Option] = spread_options if spread_options is not None else []
        # ... existing validation logic ...
    
    # Non-strategy-specific methods remain concrete
    def set_quantity(self, quantity: int): ...
    def profit_target_hit(self, profit_target: float, exit_price: float) -> bool: ...
    def stop_loss_hit(self, stop_loss: float, exit_price: float) -> bool: ...
    def get_days_to_expiration(self, current_date: datetime) -> int: ...
    def __str__(self) -> str: ...
    
    # Strategy-specific methods become abstract
    @abstractmethod
    def get_return_dollars(self, exit_price: float) -> float:
        """Calculate dollar return for this position."""
        pass
    
    @abstractmethod
    def _get_return(self, exit_price: float) -> float:
        """Calculate percentage return for this position."""
        pass
    
    @abstractmethod
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price from option chain."""
        pass
    
    @abstractmethod
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """Calculate current exit price from option bars."""
        pass


class CreditSpreadPosition(Position):
    """Position for credit spread strategies (CALL_CREDIT_SPREAD, PUT_CREDIT_SPREAD)."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # Credit received - Cost to close
        return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        # Existing credit spread logic
        ...
    
    def calculate_exit_price_from_bars(self, atm_bar, otm_bar) -> float:
        # Existing credit spread logic
        ...


class DebitSpreadPosition(Position):
    """Position for debit spread strategies (CALL_DEBIT_SPREAD, PUT_DEBIT_SPREAD)."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # Exit value - Debit paid
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        # Debit spread logic
        ...
    
    def calculate_exit_price_from_bars(self, atm_bar, otm_bar) -> float:
        # Debit spread logic
        ...


class LongCallPosition(Position):
    """Position for long call options."""
    ...

class ShortCallPosition(Position):
    """Position for short call options."""
    ...

class LongPutPosition(Position):
    """Position for long put options."""
    ...

class ShortPutPosition(Position):
    """Position for short put options."""
    ...
```

**Factory Pattern for Position Creation:**

```python
def create_position(symbol: str, expiration_date: datetime, strategy_type: StrategyType,
                   strike_price: float, entry_date: datetime, entry_price: float,
                   exit_price: float = None, spread_options: list[Option] = None) -> Position:
    """Factory function to create appropriate Position subclass based on strategy_type."""
    
    if strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
        return CreditSpreadPosition(symbol, expiration_date, strategy_type, strike_price,
                                   entry_date, entry_price, exit_price, spread_options)
    elif strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
        return DebitSpreadPosition(symbol, expiration_date, strategy_type, strike_price,
                                  entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.LONG_CALL:
        return LongCallPosition(symbol, expiration_date, strategy_type, strike_price,
                               entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.SHORT_CALL:
        return ShortCallPosition(symbol, expiration_date, strategy_type, strike_price,
                                entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.LONG_PUT:
        return LongPutPosition(symbol, expiration_date, strategy_type, strike_price,
                              entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.SHORT_PUT:
        return ShortPutPosition(symbol, expiration_date, strategy_type, strike_price,
                               entry_date, entry_price, exit_price, spread_options)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
```

**Benefits:**
- ✅ Eliminates conditional logic based on strategy_type
- ✅ Follows Open/Closed Principle (open for extension, closed for modification)
- ✅ Each strategy type encapsulates its own P&L logic
- ✅ Easy to add new strategy types without modifying existing code
- ✅ Better testability (test each strategy type in isolation)

**Testing Strategy:**
1. **Create comprehensive tests for each Position subclass**
2. **Update all existing tests to use factory function**
3. **Test all P&L calculations match previous behavior**

**Migration Path:**
1. Create abstract base class and subclasses in `common/models.py`
2. Add factory function
3. Update all Position instantiations to use factory
4. Update all imports from `backtest.models` to `common.models`
5. Run full test suite to verify no regressions

---

### Phase 1: Core Data Models & Enums ✅ (Foundational)

#### 3.1 Add Debit Spread Types to Enums

**File:** `src/algo_trading_engine/backtest/models.py`
```python
class StrategyType(Enum):
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_DEBIT_SPREAD = "call_debit_spread"      # NEW
    PUT_DEBIT_SPREAD = "put_debit_spread"        # NEW
    LONG_STOCK = "long_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
```

**File:** `src/algo_trading_engine/backtest/config.py`
```python
class StrategyType(Enum):
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_DEBIT_SPREAD = "call_debit_spread"      # NEW
    PUT_DEBIT_SPREAD = "put_debit_spread"        # NEW
    LONG_STOCK = "long_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
```

**Testing:**
- Update enum imports in all test files
- Verify enum values are accessible

---

#### 3.2 Create Spread Result DTOs (Optional but Recommended)

**File:** `src/algo_trading_engine/dto/options_dtos.py` (new DTOs)

```python
@dataclass(frozen=True)
class CreditSpreadResultDTO:
    """Result from finding an optimal credit spread."""
    atm_contract: OptionContractDTO
    otm_contract: OptionContractDTO
    atm_bar: OptionBarDTO
    otm_bar: OptionBarDTO
    credit: float
    width: float
    credit_width_ratio: float

@dataclass(frozen=True)
class DebitSpreadResultDTO:
    """Result from finding an optimal debit spread."""
    itm_contract: OptionContractDTO
    otm_contract: OptionContractDTO
    itm_bar: OptionBarDTO
    otm_bar: OptionBarDTO
    debit: float
    width: float
    max_profit: float
    max_loss: float
    reward_risk_ratio: float
```

**Rationale:**
- Type safety over `Dict[str, Any]`
- Better IDE autocomplete and type checking
- Clear documentation of expected fields
- Can defer this to Phase 4 if needed

**Testing:**
- Unit tests for DTO creation and validation
- Ensure immutability

---

### Phase 2: Add Debit Spread Enum Values 🔧 (Quick Win)

**Note:** With the new architecture from Phase 0, this phase is simplified. We just add enum values and create the `DebitSpreadPosition` class (already outlined in Phase 0).

#### 3.3 Update Enum Values

**Files:** 
- `src/algo_trading_engine/backtest/models.py`
- `src/algo_trading_engine/backtest/config.py`

Add to `StrategyType`:
```python
CALL_DEBIT_SPREAD = "call_debit_spread"
PUT_DEBIT_SPREAD = "put_debit_spread"
```

#### 3.4 Implement DebitSpreadPosition Class ✅

**File:** `src/algo_trading_engine/vo/position.py`

**✅ COMPLETED** - Already implemented in Phase 0.2 with proper debit spread P&L logic.

**Original Methods to Reference (from old Position class):**

##### `get_return_dollars()` (line ~201)
```python
def get_return_dollars(self, exit_price: float) -> float:
    """Get the return in dollars for a position."""
    if self.quantity is None:
        raise ValueError("Quantity is not set")
    
    # Credit spreads: Initial Credit - Cost to Close
    if self.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
        return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
    
    # Debit spreads: Exit Value - Initial Debit
    elif self.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    # Other position types: Standard calculation
    else:
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
```

##### `_get_return()` (line ~218)
```python
def _get_return(self, exit_price: float) -> float:
    """Get the percentage return for a position."""
    if self.quantity is None or exit_price is None:
        raise ValueError("Quantity is not set")
    
    # Credit spreads: Return based on credit received
    if self.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
        return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    # Debit spreads: Return based on debit paid
    elif self.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
        # For debit spreads, return is based on initial debit paid
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    # Other position types
    else:
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
```

##### `calculate_exit_price()` (line ~232)
Add debit spread cases:
```python
def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
    # ... existing code ...
    
    # Calculate current net credit/debit based on strategy type
    if self.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
        current_net_credit = current_atm_price - current_otm_price
        return current_net_credit
    elif self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
        current_net_credit = current_atm_price - current_otm_price
        return current_net_credit
    
    # NEW: Debit spread calculations
    elif self.strategy_type == StrategyType.CALL_DEBIT_SPREAD:
        # For call debit spread: buy ITM call, sell OTM call
        # Current value = ITM price - OTM price (what you'd get closing now)
        current_net_value = current_atm_price - current_otm_price
        return current_net_value
    elif self.strategy_type == StrategyType.PUT_DEBIT_SPREAD:
        # For put debit spread: buy ITM put, sell OTM put
        current_net_value = current_atm_price - current_otm_price
        return current_net_value
    
    else:
        raise ValueError(f"Invalid strategy type: {self.strategy_type}")
```

##### `calculate_exit_price_from_bars()` (line ~275)
Similar updates for bar-based calculations.

##### `profit_target_hit()` and `stop_loss_hit()` (lines ~108, ~114)
- Review if current logic works for debit spreads
- May need separate logic since profit direction is reversed

**Testing:**
- Unit tests for each P&L method with debit spread positions
- Edge cases: zero debit, max profit scenarios, max loss scenarios
- Integration tests with backtesting engine

---

#### 3.5 Update Capital Manager

**File:** `src/algo_trading_engine/prediction/capital_manager.py`

- Verify that capital manager correctly handles debit spread strategy names
- Ensure risk calculations work for debit spreads
  - Risk for debit spread = debit paid (known upfront)
  - Max loss = debit paid
  - Different from credit spreads where risk = width - credit

**Method to Review:**
- `calculate_position_quantity()` - Ensure debit spread risk is calculated correctly
- Position sizing should be based on debit paid, not spread width

**Testing:**
- Unit tests for debit spread capital allocation
- Verify max risk is enforced correctly

---

### Phase 3: Options Helper Refactoring 🔨

#### 3.6 Update Return Types

**File:** `src/algo_trading_engine/common/options_helpers.py`

If implementing DTOs (Phase 1), update:
- `find_credit_spread_max_credit_width()` → Return `Optional[CreditSpreadResultDTO]`
- `find_debit_spread_max_reward_risk()` → Return `Optional[DebitSpreadResultDTO]`

**Impact:**
- All callers need to be updated
- Better type safety
- Breaking change if already in use

---

### Phase 4: Recommendation Engine & UI 🖥️ (Medium Priority)

#### 3.7 Update Recommendation Engine

**File:** `src/algo_trading_engine/prediction/recommendation_engine.py`

- Ensure recommendation engine can suggest debit spread positions
- Update position status display to show debit spread info correctly
- Verify P&L calculations in recommendations

**Methods to Review:**
- `_position_from_decision()` - Needs to handle debit spread types
- `get_open_positions_status()` - Display logic for debit spreads
- Position closing logic - Ensure debit spreads close correctly

**Testing:**
- Integration tests with recommendation engine
- Test debit spread recommendations and closures

---

#### 3.8 Update CLI and Plotting

**Files:**
- `src/algo_trading_engine/prediction/recommend_cli.py`
- `src/algo_trading_engine/prediction/plot_equity_curve.py`

- Ensure CLI displays debit spreads correctly
- Update any hardcoded strategy type checks
- Verify equity curve plotting works with mixed position types

**Testing:**
- Manual testing of CLI output
- Verify plots render correctly with debit spreads

---

### Phase 7: Testing & Documentation 📝 (Ongoing)

#### 3.9 Test Coverage

**Update Existing Tests:**
- `position_statistics_test.py` - Add debit spread test cases
- `capital_manager_test.py` - Test debit spread capital allocation
- `strategy_enhancements_test.py` - Add debit spread scenarios

**Test Scenarios:**
- P&L calculations (profit, loss, breakeven)
- Position sizing and risk management
- Profit target and stop loss with debit spreads
- Integration with backtesting engine

---

#### 3.10 Documentation Updates

**Files to Update:**
```
docs/
├── strategy_builder_guide.md        # Add debit spread examples
├── volume_validation_guide.md       # Update if applicable
```

**README Updates:**
- `src/algo_trading_engine/strategies/README.md` - Document debit spread strategies

---

## 6. Success Criteria

1. **Architectural refactoring complete:** 
   - Position class moved to common/models.py
   - All Position subclasses implemented and tested
   - No conditional strategy_type logic remains
   - Factory pattern working correctly
2. **All tests pass:** 586+ tests passing (current) + new debit spread tests + refactoring tests
4. **P&L accuracy:** Debit spread P&L matches manual calculations
5. **Paper trading:** Can execute debit spread trades in paper trading mode
7. **No regressions:** Existing credit spread functionality unchanged

---

## 9. References

- Current Implementation: `find_debit_spread_max_reward_risk()` in `options_helpers.py:937-1079`
- Credit Spread Reference: `velocity_signal_momentum_strategy.py:342` (`_create_put_credit_spread()`)
- Position Model: `backtest/models.py:69-584`
- Capital Manager: `prediction/capital_manager.py`
