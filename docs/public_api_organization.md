# Public API Organization

## Overview

The `algo-trading-engine` package exposes engines and helpers from the package root, plus these subpackages: `backtest`, `models`, `dto`, `vo`, `enums`, `indicators`, `plotting`, `database`, `logging`, and `gui`. Implementation that is not part of that surface lives in `_internal`.

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

    # Data access
    DataRetriever,
    FedEventsProvider,
    OptionsHandler,
    OptionsRetrieverHelper,
)
```

`BacktestEngine` is also available as `algo_trading_engine.backtest.BacktestEngine`. Config and metrics classes are defined in `algo_trading_engine.models`.

### 2. Data Transfer Objects (DTOs)
For API communication, options data, and persisted decisions:

```python
from algo_trading_engine.dto import (
    OptionContractDTO,
    OptionBarDTO,
    StrikeRangeDTO,
    ExpirationRangeDTO,
    OptionsChainDTO,
    ProposedPositionRequestDTO,
    DecisionResponseDTO,
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
    UniversalCloseCondition,
)
```

### 4. Value Objects and Runtime Types
For strategy development and domain models:

```python
from algo_trading_engine.vo import (
    # Runtime objects
    Position,
    SpreadPosition,
    create_position,
    CreditSpreadPosition,
    DebitSpreadPosition,
    LongCallPosition,
    ShortCallPosition,
    LongPutPosition,
    ShortPutPosition,
    Option,

    # Value objects
    TreasuryRates,
    StrikePrice,
    ExpirationDate,
    MarketState,
    TradingSignal,
    PriceRange,
    Volatility,
)
```

### 5. Indicators, plotting, decisions, logging, and GUI

```python
from algo_trading_engine.indicators import Indicator, SMAIndicator, ATRIndicator
from algo_trading_engine.plotting import PlotConfig, PlotSpec, HeatmapSpec, build_plot_spec, show_plot
from algo_trading_engine.database import DecisionStore, JsonDecisionStore
from algo_trading_engine.logging import get_logger, configure_logger, remove_logger_sink
from algo_trading_engine.gui import RunObserver, JsonLinesRunObserver, observer_from_env
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
- **Top-level API**: Engines, `Strategy`, configuration, metrics, and data-access helpers
- **backtest** and **models**: Backtest engine, configs, and performance metrics
- **dto**: Records passed across layers (options payloads, proposed positions, decisions)
- **enums**: Fixed sets used by strategies and configuration
- **vo**: Positions, option contracts, and other domain values
- **indicators**: Technical indicators for custom strategies
- **plotting**: Plot specs and `show_plot`
- **database**: Decision persistence
- **logging** and **gui**: Run logs and an observer stream for an external host
- **_internal**: Package-private implementation. It ships with the wheel and is not a supported import path.

A public package's `__init__.py` re-exports only names defined in modules inside that same package.

### Import Patterns
```python
# Recommended: explicit imports
from algo_trading_engine import BacktestEngine, BacktestConfig, DataRetriever
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Position
from algo_trading_engine.dto import OptionContractDTO, DecisionResponseDTO

# Also valid: sub-package imports
from algo_trading_engine import vo, enums, dto
position_type = vo.Position
option_dto = dto.OptionContractDTO(...)

# Avoid: package-private modules
from algo_trading_engine._internal.strategies.credit_spread_minimal import CreditSpreadStrategy
from algo_trading_engine.backtest._volume_validator import VolumeValidator
```

## Package Structure

```
algo_trading_engine/
├── __init__.py              # Root exports
├── backtest/                # BacktestEngine
├── models/                  # BacktestConfig, PaperTradingConfig, metrics
├── dto/                     # Public DTOs
├── vo/                      # Positions, options, and value objects
├── enums/                   # Public enums
├── indicators/              # Indicator, SMAIndicator, ATRIndicator
├── plotting/                # PlotConfig, specs, show_plot
├── database/                # JsonDecisionStore
├── logging/                 # get_logger, configure_logger
├── gui/                     # RunObserver, JsonLinesRunObserver
├── strategy.py              # Strategy
├── trade.py                 # PaperTradingEngine
├── data_retriever.py        # DataRetriever
├── fed_events_provider.py   # FedEventsProvider
├── options_handler.py       # OptionsHandler
├── options_helpers.py       # OptionsRetrieverHelper
└── _internal/               # Package-private implementation
```

## Adding New Public Types

When adding a name to the public API:

1. **DTOs**: Define the type in `dto/` and export it from `dto/__init__.py`.
2. **Enums**: Define the type in `enums/` and export it from `enums/__init__.py`.
3. **Value objects and positions**: Define the type in `vo/` and export it from `vo/__init__.py`.
4. **Configs and metrics**: Add them to `models/` and, when they belong on the package root, register them in `algo_trading_engine/__init__.py`.
5. **Indicators, plotting, database, logging, and gui**: Define the name in a module inside that package and export it from that package's `__init__.py`.
6. **Root exports**: Add the name to `_LAZY_EXPORTS` or the subpackage to `_LAZY_SUBMODULES`. Those maps must not reference `_internal`.
7. **Tests**: Cover the export in `tests/test_public_api.py` or `tests/test_lazy_public_api.py`. `tests/test_public_surface_boundary.py` checks that public package inits stay inside their own package.

## Testing

```bash
# Value objects and enums
pytest tests/test_types_subpackage.py -v

# Root public API
pytest tests/test_public_api.py -v

# Lazy subpackages and import boundaries
pytest tests/test_lazy_public_api.py tests/test_public_surface_boundary.py -v

# CLI integration
pytest tests/test_cli_integration.py -v
```
