# Strategy Builder Pattern Guide

This guide explains how to use the Strategy Builder Pattern implemented in the backtesting system to create and configure trading strategies.

## Overview

The Strategy Builder Pattern provides a flexible way to create trading strategies with different configurations. It separates the construction of complex strategy objects from their representation, allowing the same construction process to create different configurations.

## Key Components

### 1. StrategyBuilder (Abstract Base Class)
- Defines the interface for all strategy builders
- Ensures consistent method signatures across different strategy types
- Provides abstract methods for setting common parameters

### 2. Concrete Strategy Builders
- `CreditSpreadStrategyBuilder`: Creates credit spread strategies
- `VelocitySignalMomentumStrategyBuilder`: Creates momentum strategies
- Each builder implements the specific logic for its strategy type

### 3. StrategyFactory
- Central registry for all available strategies
- Provides factory methods for creating strategies
- Handles strategy registration and discovery

## Usage Examples

### Basic Usage with Command Line

```bash
# Run credit spread strategy with default settings
python run_backtest.py --strategy credit_spread

# Run velocity momentum strategy with custom parameters
python run_backtest.py --strategy velocity_momentum --start-date-offset 30

# Run with custom configuration
python run_backtest.py --strategy credit_spread \
    --initial-capital 10000 \
    --stop-loss 0.5 \
    --profit-target 0.3 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### Programmatic Usage

```python
from src.backtest.strategy_builder import StrategyFactory, create_strategy_from_args

# Create strategy using factory
strategy = StrategyFactory.create_strategy(
    strategy_name='credit_spread',
    lstm_model=lstm_model,
    lstm_scaler=scaler,
    options_handler=options_handler,
    start_date_offset=60,
    stop_loss=0.6,
    profit_target=0.4
)

# Or use the convenience function
strategy = create_strategy_from_args(
    strategy_name='credit_spread',
    lstm_model=lstm_model,
    lstm_scaler=scaler,
    options_handler=options_handler,
    start_date_offset=60
)
```

### Manual Builder Usage

```python
from src.backtest.strategy_builder import CreditSpreadStrategyBuilder

# Create builder and configure step by step
builder = CreditSpreadStrategyBuilder()
strategy = (builder
           .set_lstm_model(lstm_model)
           .set_lstm_scaler(scaler)
           .set_options_handler(options_handler)
           .set_start_date_offset(60)
           .set_stop_loss(0.6)
           .set_profit_target(0.4)
           .build())
```

## Available Strategies

### Credit Spread Strategy (`credit_spread`)
- **Description**: Options credit spread strategy using LSTM predictions
- **Required Parameters**: `lstm_model`, `lstm_scaler`, `options_handler`
- **Optional Parameters**: `start_date_offset`, `stop_loss`, `profit_target`
- **Default Stop Loss**: 60%

### Velocity Signal Momentum Strategy (`velocity_momentum`)
- **Description**: Momentum-based strategy using velocity signals
- **Required Parameters**: None (uses default symbol 'SPY')
- **Optional Parameters**: `symbol`, `start_date_offset`
- **Default Start Date Offset**: 60 days

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--strategy` | string | `credit_spread` | Strategy to use for backtesting |
| `--start-date-offset` | int | 60 | Start date offset for strategy |
| `--stop-loss` | float | 0.6 | Stop loss percentage |
| `--profit-target` | float | None | Profit target percentage |
| `--initial-capital` | float | 5000 | Initial capital for backtesting |
| `--max-position-size` | float | 0.20 | Maximum position size as fraction of capital |
| `--start-date` | string | 2025-01-01 | Start date for backtest (YYYY-MM-DD) |
| `--end-date` | string | 2025-08-01 | End date for backtest (YYYY-MM-DD) |
| `--symbol` | string | SPY | Symbol to trade |
| `--verbose` | flag | False | Run in quiet mode |

## Extending the System

### Adding a New Strategy

1. **Create the Strategy Class**
```python
from src.backtest.models import Strategy

class MyNewStrategy(Strategy):
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def on_new_date(self, date, positions, add_position, remove_position):
        # Implement strategy logic
        pass
```

2. **Create the Builder Class**
```python
from src.backtest.strategy_builder import StrategyBuilder

class MyNewStrategyBuilder(StrategyBuilder):
    def reset(self):
        self._custom_param = None
        self._lstm_model = None
        self._lstm_scaler = None
        self._options_handler = None
        self._start_date_offset = 0
        self._stop_loss = None
        self._profit_target = None
    
    def set_custom_param(self, value):
        self._custom_param = value
        return self
    
    def set_lstm_model(self, model):
        self._lstm_model = model
        return self
    
    def set_lstm_scaler(self, scaler):
        self._lstm_scaler = scaler
        return self
    
    def set_options_handler(self, handler):
        self._options_handler = handler
        return self
    
    def set_start_date_offset(self, offset):
        self._start_date_offset = offset
        return self
    
    def set_stop_loss(self, stop_loss):
        self._stop_loss = stop_loss
        return self
    
    def set_profit_target(self, profit_target):
        self._profit_target = profit_target
        return self
    
    def build(self):
        return MyNewStrategy(
            custom_param=self._custom_param,
            start_date_offset=self._start_date_offset
        )
```

## Configuration Profiles

You can create predefined configuration profiles for different risk levels:

```python
# Conservative profile
conservative_config = {
    'start_date_offset': 90,
    'stop_loss': 0.4,
    'profit_target': 0.2
}

# Aggressive profile
aggressive_config = {
    'start_date_offset': 30,
    'stop_loss': 0.8,
    'profit_target': 0.6
}

# Apply configuration
strategy = StrategyFactory.create_strategy('credit_spread', **conservative_config)
```

## Error Handling

The builder pattern includes comprehensive error handling:

- **Missing Required Parameters**: Raises `ValueError` with descriptive message
- **Invalid Strategy Name**: Shows available strategies
- **Configuration Validation**: Validates parameters before strategy creation

## Benefits

1. **Flexibility**: Easy to switch between strategies and configurations
2. **Validation**: Built-in parameter validation
3. **Extensibility**: Simple to add new strategies
4. **Type Safety**: Strong typing throughout the builder chain
5. **Method Chaining**: Fluent interface for configuration
6. **Separation of Concerns**: Strategy creation logic separated from usage
