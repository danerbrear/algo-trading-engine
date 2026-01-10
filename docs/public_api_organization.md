# Public API Organization

## Overview

The `algo-trading-engine` package provides a clean, organized public API with three main layers:

### 1. Core API (Top-Level Imports)
Primary components for running backtests and paper trading:

```python
from algo_trading_engine import (
    # Engines
    BacktestEngine,
    PaperTradingEngine,
    
    # Configuration
    BacktestConfig,
    PaperTradingConfig,
    VolumeConfig,
    VolumeStats,
    
    # Strategy Base
    Strategy,
    
    # Metrics
    PerformanceMetrics,
    PositionStats,
)
```

### 2. Data Transfer Objects (DTOs)
For API communication and options data:

```python
from algo_trading_engine.dto import (
    OptionContractDTO,
    OptionBarDTO,
    StrikeRangeDTO,
    ExpirationRangeDTO,
    OptionsChainDTO,
)
```

### 3. Runtime Types and Value Objects
For strategy development and domain models:

```python
from algo_trading_engine.types import (
    # Enums
    StrategyType,
    
    # Runtime Objects
    Position,
    Option,
    
    # Value Objects
    TreasuryRates,
)
```

## Usage Examples

### Basic Backtest

```python
from datetime import datetime
from algo_trading_engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2025, 1, 1),
    symbol="SPY",
    strategy_type="velocity_momentum"
)

engine = BacktestEngine.from_config(config)
engine.run()
```

### Working with Position Results

```python
from algo_trading_engine import BacktestEngine
from algo_trading_engine.types import StrategyType, Position

# After running a backtest
engine = BacktestEngine.from_config(config)
engine.run()

# Access positions (internal engine state)
for position in engine.positions:
    if position.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
        print(f"Credit spread position: {position.symbol} @ {position.strike_price}")
```

### Custom Strategy with Full Type Support

```python
from algo_trading_engine import Strategy
from algo_trading_engine.types import Position, StrategyType, Option, TreasuryRates
from algo_trading_engine.dto import OptionsChainDTO
from typing import Optional

class MyCustomStrategy(Strategy):
    def should_open_position(
        self,
        current_date,
        current_price: float,
        options_chain: OptionsChainDTO,
        treasury_rates: TreasuryRates
    ) -> Optional[Position]:
        # Access treasury rates
        risk_free_rate = treasury_rates.get_1y_rate(current_date)
        
        # Create position with proper types
        position = Position(
            symbol="SPY",
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            ...
        )
        return position
```

### Working with Options

```python
from algo_trading_engine.types import Option
from algo_trading_engine.dto import OptionContractDTO

# Use DTOs for fetching data from APIs
option_dto = OptionContractDTO(
    symbol="SPY250117C00500000",
    strike=500.0,
    expiration="2025-01-17",
    ...
)

# Convert to Value Object for domain logic
option = Option(
    strike=option_dto.strike,
    expiration=option_dto.expiration,
    ...
)
```

## Design Principles

### Separation of Concerns
- **Top-level API**: Execution engines and configurations
- **dto package**: External data representation (API responses, serialization)
- **types package**: Internal domain models (business logic, calculations)

### Import Patterns
```python
# ✅ Recommended: Explicit imports
from algo_trading_engine import BacktestEngine, BacktestConfig
from algo_trading_engine.types import StrategyType, Position
from algo_trading_engine.dto import OptionContractDTO

# ✅ Also valid: Sub-package imports
from algo_trading_engine import types, dto
position = types.Position(...)
option_dto = dto.OptionContractDTO(...)

# ❌ Avoid: Internal imports
from algo_trading_engine.common.models import Option  # Internal detail
from algo_trading_engine.backtest.models import Position  # Internal detail
```

### Backward Compatibility
All existing code continues to work. The new `types` sub-package is an addition, not a breaking change.

## Package Structure

```
algo_trading_engine/
├── __init__.py              # Core API exports
├── dto/
│   └── __init__.py          # DTOs for API communication
├── types/
│   └── __init__.py          # Runtime types and value objects
├── models/
│   ├── config.py            # Configuration DTOs
│   └── metrics.py           # Metrics DTOs
├── backtest/                # Internal implementation
├── core/                    # Internal implementation
├── common/                  # Internal utilities
└── ...                      # Other internal packages
```

## Adding New Public Types

When adding new types to the public API:

1. **For DTOs** (external data): Add to `dto/__init__.py`
2. **For runtime types/VOs**: Add to `types/__init__.py`
3. **For configs/metrics**: Add to `models/` package
4. **Update tests**: Add to `test_types_subpackage.py` or `test_public_api.py`

## Testing

All public API functionality is tested:

```bash
# Test types sub-package
pytest tests/test_types_subpackage.py -v

# Test overall public API
pytest tests/test_public_api.py -v

# Test CLI integration
pytest tests/test_cli_integration.py -v
```
