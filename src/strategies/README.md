# Trading Strategies

This directory contains implementations of various options trading strategies for the LSTM POC system.

## Overview

The strategies in this directory implement the `Strategy` base class and provide specific trading logic for different market conditions and risk profiles.

## Available Strategies

### 1. Velocity Signal Momentum Strategy (`velocity_signal_momentum_strategy.py`)

A momentum-based strategy that trades credit spreads to capitalize on upward or downward trends.

#### Key Features
- **Momentum Detection**: Identifies trends using price action and technical indicators
- **Credit Spread Focus**: Primarily trades put and call credit spreads
- **Sharpe Ratio Optimization**: Uses risk-adjusted returns for position sizing
- **Treasury Rate Integration**: Incorporates real risk-free rates for accurate calculations

#### Strategy Logic
1. **Signal Generation**: 
   - Price must increase over the trend period
   - No significant reversals (>2% drop) during the trend
   - Trend must last at least 3 days
   - Trend must not exceed 60 days

2. **Position Management**:
   - Opens positions when no existing positions are held
   - Uses Sharpe ratio to determine optimal expiration dates
   - Implements stop-loss and profit target logic

3. **Risk Management**:
   - Calculates position-specific Sharpe ratios
   - Uses treasury rates for risk-free rate calculations
   - Implements volume validation for liquidity

#### Usage
```python
from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy

# Create strategy instance
strategy = VelocitySignalMomentumStrategy()

# Set data (including treasury rates)
strategy.set_data(market_data, options_data, treasury_rates)

# Strategy is used by the backtesting engine
```

#### Configuration
- `start_date_offset`: Number of days to skip at the beginning (default: 60)
- `profit_target`: Optional profit target percentage
- `stop_loss`: Optional stop loss percentage

### 2. Credit Spread Minimal Strategy (`credit_spread_minimal.py`)

A minimal implementation of credit spread trading for testing and development.

#### Key Features
- **Simple Logic**: Basic credit spread implementation
- **Development Focus**: Used for testing and validation
- **Volume Validation**: Integrates with the volume validation system

## Strategy Base Class

All strategies inherit from the `Strategy` base class in `src/backtest/models.py`:

### Required Methods
- `on_new_date()`: Called for each trading day
- `on_end()`: Called at the end of the backtest

### Optional Methods
- `_has_buy_signal()`: Determine if a buy signal exists
- `_determine_expiration_date()`: Find optimal expiration dates
- `_calculate_sharpe_ratio()`: Calculate risk-adjusted returns

### Data Access
Strategies have access to:
- `self.data`: Market data (OHLCV)
- `self.options_data`: Options chain data
- `self.treasury_data`: Treasury rates for risk calculations

## Treasury Rate Integration

Strategies can access treasury rates through the `TreasuryRates` Value Object:

```python
def _get_risk_free_rate(self, date: datetime) -> float:
    """Get risk-free rate for Sharpe ratio calculations"""
    if self.treasury_data is None:
        return 0.0
    return float(self.treasury_data.get_risk_free_rate(date))
```

## Volume Validation

All strategies integrate with the volume validation system:

```python
# Strategies should validate volume before creating positions
if self._validate_option_volume(option):
    # Create position
    add_position(position)
```

## Testing

Each strategy has corresponding unit tests in the `tests/` directory:

```bash
# Run strategy-specific tests
python -m pytest tests/test_velocity_strategy.py -v
```

## Adding New Strategies

To add a new strategy:

1. **Create the strategy file**:
```python
from src.backtest.models import Strategy

class MyNewStrategy(Strategy):
    def __init__(self):
        super().__init__(start_date_offset=60)
    
    def on_new_date(self, date, positions, add_position, remove_position):
        # Implement your strategy logic
        pass
    
    def on_end(self, positions, remove_position, date):
        # Clean up logic
        pass
```

2. **Add to strategy factory** in `src/backtest/strategy_builder.py`

3. **Create unit tests** in `tests/test_my_strategy.py`

4. **Update this README** with strategy documentation

## Best Practices

1. **Value Objects**: Use Value Objects for data transfer
2. **Error Handling**: Implement proper error handling and fallbacks
3. **Testing**: Create comprehensive unit tests
4. **Documentation**: Document strategy logic and parameters
5. **Volume Validation**: Always validate option volume before trading
6. **Risk Management**: Implement proper position sizing and risk controls

## Performance Considerations

- **Data Loading**: Strategies receive pre-loaded data to avoid repeated API calls
- **Caching**: Use cached data when available
- **Memory Efficiency**: Avoid storing large datasets in strategy instances
- **Computation**: Optimize calculations for backtesting performance

## Risk Warnings

- All strategies are for educational and research purposes
- Options trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always validate strategies thoroughly before live trading
