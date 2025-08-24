# Value Object Rules

## Core Principles

### 1. **Immutability**
- Value Objects should be immutable after creation
- Use `@dataclass(frozen=True)` or `@property` decorators
- Prevent modification of internal state
- Return new instances for any "changes"

### 2. **Value-Based Equality**
- Two Value Objects are equal if all their attributes are equal
- Implement `__eq__` method based on all attributes
- Use `__hash__` for hashable collections
- Consider using `@dataclass` with `eq=True`

### 3. **Self-Contained Behavior**
- Value Objects should contain their own validation logic
- Include business rules and constraints
- Provide methods for common operations
- Encapsulate related data and behavior

### 4. **Domain Representation**
- Represent real-world domain concepts
- Use descriptive, domain-specific names
- Avoid technical implementation details
- Focus on business meaning, not data storage

## Naming Conventions

### 5. **Descriptive Names**
- Use domain terminology, not technical terms
- Examples: `MarketState`, `TradingSignal`, `PriceRange`
- Avoid generic names like `Data`, `Info`, `Object`
- Use nouns that represent domain concepts

### 6. **Consistent Naming**
- Use PascalCase for class names
- Use snake_case for attributes and methods
- Use descriptive attribute names
- Follow domain language conventions

## Structure and Organization

### 7. **Location in Project**
- Place Value Objects in `src/common/models.py` for shared domain objects
- Create package-specific Value Objects in their respective packages
- Use `src/model/` for model-specific Value Objects
- Use `src/strategies/` for strategy-specific Value Objects

### 8. **Package Organization**
```python
# src/common/models.py - Shared Value Objects
class MarketState:
    """Represents a market state identified by HMM"""
    
class TradingSignal:
    """Represents a trading signal with confidence"""
    
class PriceRange:
    """Represents a price range with validation"""
```

## Implementation Guidelines

### 9. **Constructor Design**
- Validate all input parameters
- Use type hints for all parameters
- Provide clear error messages for invalid data
- Consider using factory methods for complex creation

### 10. **Validation Rules**
- Validate data at construction time
- Use domain-specific validation rules
- Provide meaningful error messages
- Consider using `pydantic` for complex validation

### 11. **Method Design**
- Keep methods focused and single-purpose
- Return new instances for "modifications"
- Use descriptive method names
- Include proper error handling

## Domain-Specific Examples

### 12. **Market State Value Objects**
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from decimal import Decimal

class MarketStateType(Enum):
    LOW_VOLATILITY_UPTREND = "low_volatility_uptrend"
    MOMENTUM_UPTREND = "momentum_uptrend"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY_DOWNTREND = "high_volatility_downtrend"
    HIGH_VOLATILITY_RALLY = "high_volatility_rally"

@dataclass(frozen=True)
class MarketState:
    """Represents a market state with characteristics"""
    state_type: MarketStateType
    volatility: Decimal
    average_return: Decimal
    confidence: Decimal
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
```

### 13. **Trading Signal Value Objects**
```python
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

class SignalType(Enum):
    HOLD = "hold"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"

@dataclass(frozen=True)
class TradingSignal:
    """Represents a trading signal with strategy and confidence"""
    signal_type: SignalType
    confidence: Decimal
    ticker: str
    expiration_date: Optional[str] = None
    strike_prices: Optional[tuple[Decimal, Decimal]] = None
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.signal_type != SignalType.HOLD and not self.expiration_date:
            raise ValueError("Expiration date required for option strategies")
    
    def is_option_strategy(self) -> bool:
        return self.signal_type in [SignalType.CALL_CREDIT_SPREAD, SignalType.PUT_CREDIT_SPREAD]
```

### 14. **Price and Financial Value Objects**
```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass(frozen=True)
class PriceRange:
    """Represents a price range with validation"""
    low: Decimal
    high: Decimal
    
    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("Low price cannot be greater than high price")
        if self.low < 0:
            raise ValueError("Price cannot be negative")
    
    def contains(self, price: Decimal) -> bool:
        return self.low <= price <= self.high
    
    def spread(self) -> Decimal:
        return self.high - self.low

@dataclass(frozen=True)
class Volatility:
    """Represents volatility with validation"""
    value: Decimal
    period: int  # days
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Volatility cannot be negative")
        if self.period <= 0:
            raise ValueError("Period must be positive")
    
    def is_high(self) -> bool:
        return self.value > Decimal('0.3')  # 30% threshold
    
    def is_low(self) -> bool:
        return self.value < Decimal('0.1')  # 10% threshold
```

## Usage Patterns

### 15. **Creation Patterns**
```python
# Factory method for complex creation
class MarketStateFactory:
    @staticmethod
    def create_from_hmm_output(state_id: int, confidence: float) -> MarketState:
        state_map = {
            0: MarketStateType.LOW_VOLATILITY_UPTREND,
            1: MarketStateType.MOMENTUM_UPTREND,
            # ... other mappings
        }
        return MarketState(
            state_type=state_map[state_id],
            confidence=Decimal(str(confidence)),
            # ... other parameters
        )
```

### 16. **Collection Patterns**
```python
from typing import List

@dataclass(frozen=True)
class PredictionResult:
    """Collection of predictions for multiple time periods"""
    predictions: List[TradingSignal]
    market_state: MarketState
    timestamp: str
    
    def __post_init__(self):
        if not self.predictions:
            raise ValueError("Must have at least one prediction")
    
    def high_confidence_signals(self, threshold: Decimal = Decimal('0.7')) -> List[TradingSignal]:
        return [signal for signal in self.predictions if signal.confidence >= threshold]
```

## Validation and Error Handling

### 17. **Validation Strategies**
- Use `__post_init__` for validation in dataclasses
- Provide clear, domain-specific error messages
- Validate business rules, not just data types
- Consider using custom exceptions for domain errors

### 18. **Error Messages**
```python
class DomainError(Exception):
    """Base exception for domain-specific errors"""
    pass

class InvalidMarketStateError(DomainError):
    """Raised when market state is invalid"""
    pass

class InvalidTradingSignalError(DomainError):
    """Raised when trading signal is invalid"""
    pass
```

## Performance Considerations

### 19. **Memory Efficiency**
- Use `@dataclass(frozen=True)` for immutability
- Consider using `__slots__` for memory optimization
- Use appropriate data types (Decimal for financial calculations)
- Avoid unnecessary object creation

### 20. **Caching and Hashing**
- Implement `__hash__` for hashable collections
- Use Value Objects as dictionary keys when appropriate
- Consider caching frequently used Value Objects
- Use `__eq__` for value-based equality

## Testing Guidelines

### 21. **Value Object Testing**
- Test all validation rules
- Test equality and hash methods
- Test business logic methods
- Test edge cases and boundary conditions
- Use factories for creating test data

### 22. **Test Examples**
```python
def test_market_state_validation():
    # Test valid creation
    state = MarketState(
        state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
        volatility=Decimal('0.15'),
        average_return=Decimal('0.02'),
        confidence=Decimal('0.8')
    )
    assert state.confidence == Decimal('0.8')
    
    # Test invalid confidence
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        MarketState(
            state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
            volatility=Decimal('0.15'),
            average_return=Decimal('0.02'),
            confidence=Decimal('1.5')  # Invalid
        )
```

## Integration with Other Patterns

### 23. **DTO Integration**
- Convert Value Objects to DTOs for external communication
- Convert DTOs to Value Objects for domain processing
- Use mapping functions for conversions
- Maintain domain integrity during conversions

### 24. **Repository Integration**
- Use Value Objects as parameters and return types
- Avoid exposing internal data structures
- Maintain domain boundaries
- Use Value Objects for query criteria

## Best Practices

### 25. **Domain Focus**
- Keep Value Objects focused on domain concepts
- Avoid technical implementation details
- Use domain language in names and methods
- Represent business rules, not technical constraints

### 26. **Composition Over Inheritance**
- Prefer composition for complex Value Objects
- Use simple inheritance for related concepts
- Keep inheritance hierarchies shallow
- Focus on behavior, not just data

### 27. **Documentation**
- Document business rules and constraints
- Include examples of valid and invalid usage
- Document domain-specific methods
- Use docstrings for complex Value Objects
