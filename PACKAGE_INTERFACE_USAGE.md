# Package Interface Usage Guide

This document explains how to use the new package interface for backtesting and paper trading.

## Overview

The package now provides a clean, reusable interface with:
- **Core Interfaces**: Strategy ABC, TradingEngine ABC, DataProvider protocol
- **Configuration DTOs**: Immutable configuration objects
- **Performance Metrics**: Value objects for tracking performance
- **Engine Implementations**: BacktestEngine and PaperTradingEngine

## Basic Usage

### Creating a Strategy

```python
from algo_trading_engine.core import Strategy
from algo_trading_engine.backtest.models import Position
from datetime import datetime
from typing import Callable
import pandas as pd

class MyStrategy(Strategy):
    def on_new_date(
        self,
        date: datetime,
        positions: tuple[Position, ...],
        add_position: Callable[[Position], None],
        remove_position: Callable[[datetime, Position, float, ...], None]
    ) -> None:
        # Your strategy logic here
        pass
    
    def on_end(
        self,
        positions: tuple[Position, ...],
        remove_position: Callable[[datetime, Position, float, ...], None],
        date: datetime
    ) -> None:
        # Close remaining positions
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        # Validate data requirements
        return True
```

### Running a Backtest

#### Using the New Interface (Recommended)

```python
from algo_trading_engine.core import BacktestEngine
from algo_trading_engine.models import BacktestConfig, VolumeConfig
from datetime import datetime
import pandas as pd

# Create configuration
config = BacktestConfig(
    initial_capital=100000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    symbol="SPY",
    max_position_size=0.40,
    volume_config=VolumeConfig(min_volume=10)
)

# Create strategy (using existing strategy)
strategy = create_strategy_from_args(
    strategy_name="credit_spread",
    symbol="SPY",
    options_handler=options_handler
)

# Get data
data = data_retriever.fetch_data_for_period(config.start_date, 'backtest')
strategy.set_data(data, data_retriever.treasury_rates)

# Create and run engine
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=config.initial_capital,
    start_date=config.start_date,
    end_date=config.end_date,
    max_position_size=config.max_position_size,
    volume_config=config.volume_config
)

success = engine.run()

# Get performance metrics
metrics = engine.get_performance_metrics()
print(f"Total Return: {metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Win Rate: {metrics.win_rate:.1f}%")
```

#### Using Legacy Interface (Still Supported)

```python
# The old constructor still works for backward compatibility
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=100000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    max_position_size=0.40
)
```

### Paper Trading (Stub Implementation)

```python
from algo_trading_engine.core import PaperTradingEngine
from algo_trading_engine.models import PaperTradingConfig

config = PaperTradingConfig(
    initial_capital=100000,
    symbol="SPY",
    max_position_size=0.40
)

engine = PaperTradingEngine(
    strategy=strategy,
    data_provider=data_retriever,  # DataRetriever implements DataProvider protocol
    config=config
)

# Note: PaperTradingEngine.run() is not yet implemented
```

## Performance Metrics

The `get_performance_metrics()` method returns a `PerformanceMetrics` object:

```python
metrics = engine.get_performance_metrics()

# Overall metrics
print(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
print(f"Win Rate: {metrics.win_rate:.1f}%")

# Strategy-specific metrics
for strategy_stat in metrics.strategy_stats:
    print(f"{strategy_stat.strategy_type.value}: {strategy_stat.win_rate:.1f}% win rate")

# Individual position stats
for pos_stat in metrics.closed_positions:
    print(f"Position: {pos_stat.return_dollars:+.2f} ({pos_stat.return_percentage:+.2f}%)")
```

## Configuration Objects

### BacktestConfig

```python
from algo_trading_engine.models import BacktestConfig, VolumeConfig

config = BacktestConfig(
    initial_capital=100000,          # Starting capital
    start_date=datetime(2024, 1, 1), # Backtest start
    end_date=datetime(2024, 12, 31), # Backtest end
    symbol="SPY",                     # Symbol to trade
    max_position_size=0.40,          # Max 40% of capital per position
    volume_config=VolumeConfig(      # Volume validation
        min_volume=10,
        enable_volume_validation=True
    ),
    enable_progress_tracking=True,    # Show progress bar
    quiet_mode=False                  # Verbose output
)
```

### PaperTradingConfig

```python
from algo_trading_engine.models import PaperTradingConfig

config = PaperTradingConfig(
    initial_capital=100000,
    symbol="SPY",
    max_position_size=0.40,
    execution_delay_seconds=0,  # Simulate execution delay
    volume_config=VolumeConfig(min_volume=10)
)
```

## Data Provider Protocol

The `DataProvider` protocol allows for flexible data sources:

```python
from algo_trading_engine.core.data_provider import DataProvider
from typing import Protocol
import pandas as pd
from datetime import datetime

class MyDataProvider:
    """Custom data provider implementation."""
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        # Your data fetching logic
        pass
    
    def get_current_price(self, symbol: str) -> float:
        # Get live price
        pass
    
    def get_option_chain(self, symbol: str, date: datetime):
        # Get options data
        pass
    
    def load_treasury_rates(self, start_date: datetime, end_date=None):
        # Load treasury rates
        pass

# MyDataProvider automatically implements DataProvider protocol
```

## Migration Guide

### For Existing Code

Existing code continues to work without changes. The new interfaces are additive:

- ✅ `BacktestEngine` still works with old constructor
- ✅ `Strategy` from `backtest.models` still works (now inherits from `core.strategy`)
- ✅ All existing imports continue to work

### For New Code

Use the new interfaces for better structure:

```python
# Old way (still works)
from algo_trading_engine.backtest.models import Strategy
from algo_trading_engine.backtest.main import BacktestEngine

# New way (recommended)
from algo_trading_engine.core import Strategy, BacktestEngine
from algo_trading_engine.models import BacktestConfig, PerformanceMetrics
```

## Best Practices

1. **Use Configuration DTOs**: Prefer `BacktestConfig` over individual parameters
2. **Get Performance Metrics**: Always call `get_performance_metrics()` after running
3. **Implement DataProvider**: For custom data sources, implement the `DataProvider` protocol
4. **Type Hints**: Use the provided type hints for better IDE support
5. **Immutable Configs**: Configuration objects are immutable - create new ones for changes

## Future Enhancements

- Full PaperTradingEngine implementation
- Event system for monitoring
- Strategy registry pattern
- Additional performance metrics
- Multi-symbol portfolio support

