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

### 3. Enums
For strategy development and configuration:

```python
from algo_trading_engine.enums import (
    StrategyType,
    OptionType,
    MarketStateType,
    SignalType,
    BarTimeInterval,  # For configuring data granularity
)
```

### 4. Value Objects and Runtime Types
For strategy development and domain models:

```python
from algo_trading_engine.vo import (
    # Runtime Objects
    Position,
    Option,
    
    # Value Objects
    TreasuryRates,
    StrikePrice,
    ExpirationDate,
    MarketState,
    TradingSignal,
    PriceRange,
    Volatility,
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

### Configuring Bar Intervals

```python
from datetime import datetime
from algo_trading_engine import BacktestEngine, BacktestConfig
from algo_trading_engine.enums import BarTimeInterval

# Daily bars (default) - for position/swing trading
config_daily = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2026, 1, 1),
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.DAY  # Default
)

# Hourly bars - for intraday trading (max 729 days)
config_hourly = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.HOUR
)

# Minute bars - for day trading (max 59 days)
config_minute = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 11, 1),
    end_date=datetime(2024, 12, 20),
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.MINUTE
)
```

**Bar Interval Reference:**
- `BarTimeInterval.DAY`: Daily bars, no date limit
- `BarTimeInterval.HOUR`: Hourly bars, max 729 days (yfinance limit)
- `BarTimeInterval.MINUTE`: Minute bars, max 59 days (yfinance limit)

### Working with Position Results

```python
from algo_trading_engine import BacktestEngine
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Position

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
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Position, Option, TreasuryRates
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
from algo_trading_engine.vo import Option
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
- **enums package**: Public enums for strategy development
- **vo package**: Value objects and runtime types (business logic, calculations)

### Import Patterns
```python
# ✅ Recommended: Explicit imports
from algo_trading_engine import BacktestEngine, BacktestConfig
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Position
from algo_trading_engine.dto import OptionContractDTO

# ✅ Also valid: Sub-package imports
from algo_trading_engine import vo, enums, dto
position = types.Position(...)
option_dto = dto.OptionContractDTO(...)

# ❌ Avoid: Internal imports
from algo_trading_engine.common.models import Option  # Internal detail
from algo_trading_engine.backtest.models import Position  # Internal detail
```

### Backward Compatibility
The `vo` and `enums` sub-packages replace the previous `types` sub-package. Update imports from `algo_trading_engine.types` to use `algo_trading_engine.vo` and `algo_trading_engine.enums` instead.

## Package Structure

```
algo_trading_engine/
├── __init__.py              # Core API exports
├── dto/
│   └── __init__.py          # DTOs for API communication
├── enums/
│   └── __init__.py          # Public enums
├── vo/
│   ├── __init__.py          # Value objects and runtime types
│   └── value_objects.py     # Public value objects
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
2. **For enums**: Add to `enums/__init__.py`
3. **For runtime types/VOs**: Add to `vo/__init__.py` and `vo/value_objects.py`
4. **For configs/metrics**: Add to `models/` package
5. **Update tests**: Add to `test_types_subpackage.py` or `test_public_api.py`

## Testing

All public API functionality is tested:

```bash
# Test vo and enums sub-packages
pytest tests/test_types_subpackage.py -v

# Test overall public API
pytest tests/test_public_api.py -v

# Test CLI integration
pytest tests/test_cli_integration.py -v
```
