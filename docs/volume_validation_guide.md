# Volume Validation Configuration Guide

This guide provides comprehensive information about configuring and using the volume validation system in the backtesting framework.

## Overview

The volume validation system ensures that only liquid options are traded during backtests, improving the realism of simulation results by avoiding illiquid options that would be difficult to trade in real market conditions.

## Configuration Options

### VolumeConfig DTO

The `VolumeConfig` DTO provides all configuration options for volume validation:

```python
from src.backtest.config import VolumeConfig

# Default configuration
config = VolumeConfig()  # min_volume=10, enable_volume_validation=True

# Custom configuration
config = VolumeConfig(
    min_volume=15,                    # Minimum volume threshold
    enable_volume_validation=True     # Enable/disable validation
)
```

#### Parameters

- **`min_volume`** (int, default: 10): Minimum volume required for an option to be considered liquid
- **`enable_volume_validation`** (bool, default: True): Enable or disable volume validation

#### Validation Rules

- `min_volume` must be greater than 0
- `min_volume` cannot be negative
- Configuration is immutable once created

## Best Practices

### Volume Threshold Selection

#### Conservative Approach (High Liquidity)
```python
# For conservative strategies requiring high liquidity
config = VolumeConfig(min_volume=20, enable_volume_validation=True)
```
**Use when:**
- Trading large position sizes
- Requiring tight bid-ask spreads
- Need for quick entry/exit
- Risk-averse strategies

#### Moderate Approach (Balanced)
```python
# For balanced liquidity requirements
config = VolumeConfig(min_volume=10, enable_volume_validation=True)
```
**Use when:**
- Standard trading strategies
- Medium position sizes
- Acceptable bid-ask spreads
- Most common use case

#### Aggressive Approach (Lower Liquidity)
```python
# For strategies that can handle lower liquidity
config = VolumeConfig(min_volume=5, enable_volume_validation=True)
```
**Use when:**
- Small position sizes
- Longer holding periods
- Accepting wider spreads
- Testing edge cases

### Strategy-Specific Considerations

#### Credit Spread Strategies
```python
# Credit spreads benefit from higher volume thresholds
config = VolumeConfig(min_volume=15, enable_volume_validation=True)
```
**Reasoning:**
- Two-leg strategies require both options to be liquid
- Higher volume reduces execution risk
- Better pricing accuracy

#### Single-Leg Strategies
```python
# Single-leg strategies can use lower thresholds
config = VolumeConfig(min_volume=10, enable_volume_validation=True)
```
**Reasoning:**
- Only one option needs to be liquid
- Simpler execution requirements
- More flexibility in volume requirements

### Market Conditions

#### High Volatility Periods
```python
# Increase volume requirements during high volatility
config = VolumeConfig(min_volume=20, enable_volume_validation=True)
```
**Reasoning:**
- Wider bid-ask spreads
- Reduced liquidity
- Higher execution risk

#### Low Volatility Periods
```python
# Standard volume requirements during normal conditions
config = VolumeConfig(min_volume=10, enable_volume_validation=True)
```
**Reasoning:**
- Tighter bid-ask spreads
- Better liquidity
- Lower execution risk

## Usage Examples

### Basic Implementation

```python
from src.backtest.config import VolumeConfig
from src.backtest.main import BacktestEngine
from src.strategies.credit_spread_minimal import CreditSpreadStrategy

# Create strategy
strategy = CreditSpreadStrategy(lstm_model=model, options_handler=handler)

# Create backtest engine with volume validation
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=10000,
    volume_config=VolumeConfig(min_volume=10, enable_volume_validation=True)
)

# Run backtest
success = engine.run()
```

### Advanced Configuration

```python
# Custom volume validation with detailed statistics
volume_config = VolumeConfig(min_volume=15, enable_volume_validation=True)

engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=10000,
    volume_config=volume_config
)

# Run backtest
success = engine.run()

# Analyze volume validation results
summary = engine.volume_stats.get_summary()
print(f"Volume Validation Summary:")
print(f"  Options checked: {summary['options_checked']}")
print(f"  Positions rejected: {summary['positions_rejected_volume']}")
print(f"  Rejection rate: {summary['rejection_rate']:.1f}%")
```

### Disabling Volume Validation

```python
# For existing backtests or testing purposes
config = VolumeConfig(enable_volume_validation=False)

engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=10000,
    volume_config=config
)
```

## Troubleshooting

### Common Issues

#### High Rejection Rates

**Problem:** Many positions are being rejected due to insufficient volume.

**Solutions:**
1. Lower the volume threshold:
   ```python
   config = VolumeConfig(min_volume=5, enable_volume_validation=True)
   ```

2. Check market conditions:
   - High volatility periods may require lower thresholds
   - Consider the specific options being traded

3. Review strategy logic:
   - Ensure strategies are fetching volume data properly
   - Check if options data is available

#### No Volume Data Available

**Problem:** Options have no volume data, causing all positions to be rejected.

**Solutions:**
1. Check data availability:
   ```python
   # In your strategy
   if option.volume is None:
       print(f"No volume data for {option.symbol}")
   ```

2. Verify API connectivity:
   - Ensure Polygon.io API key is valid
   - Check network connectivity

3. Review data fetching logic:
   - Ensure `_ensure_volume_data()` is called
   - Check error handling in data fetching

#### Performance Issues

**Problem:** Volume validation is slowing down backtest execution.

**Solutions:**
1. Disable validation for testing:
   ```python
   config = VolumeConfig(enable_volume_validation=False)
   ```

2. Optimize data fetching:
   - Use cached data when available
   - Implement batch API calls if possible

3. Reduce validation frequency:
   - Only validate on position creation
   - Cache validation results when appropriate

### Debugging Tips

#### Enable Detailed Logging

```python
# Add logging to understand volume validation decisions
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Volume Statistics

```python
# After running backtest
stats = engine.volume_stats
print(f"Options checked: {stats.options_checked}")
print(f"Positions rejected: {stats.positions_rejected_volume}")

# Get detailed summary
summary = stats.get_summary()
for key, value in summary.items():
    print(f"{key}: {value}")
```

#### Validate Individual Options

```python
from src.backtest.volume_validator import VolumeValidator

# Test individual option validation
option = your_option_object
is_liquid = VolumeValidator.validate_option_volume(option, min_volume=10)
print(f"Option {option.symbol} is liquid: {is_liquid}")

# Get detailed status
status = VolumeValidator.get_volume_status(option)
print(f"Volume status: {status}")
```

## Performance Considerations

### Memory Usage

- Volume validation adds minimal memory overhead
- Statistics tracking uses immutable DTOs
- No persistent storage of validation results

### Execution Time

- Volume validation adds ~1-5ms per option
- Negligible impact on overall backtest performance
- Can be disabled for performance-critical scenarios

### API Usage

- Volume data is fetched by strategies, not BacktestEngine
- Caching reduces API calls
- Error handling prevents excessive retries

## Migration Guide

### From Existing Backtests

1. **Enable gradually:**
   ```python
   # Start with disabled validation
   config = VolumeConfig(enable_volume_validation=False)
   
   # Then enable with conservative threshold
   config = VolumeConfig(min_volume=20, enable_volume_validation=True)
   
   # Finally, adjust to optimal threshold
   config = VolumeConfig(min_volume=10, enable_volume_validation=True)
   ```

2. **Monitor statistics:**
   ```python
   # Track rejection rates
   summary = engine.volume_stats.get_summary()
   if summary['rejection_rate'] > 50:
       print("Warning: High rejection rate - consider adjusting threshold")
   ```

3. **Update strategies:**
   - Ensure strategies call `_ensure_volume_data()`
   - Handle cases where volume data is unavailable
   - Add appropriate error handling

### Testing Recommendations

1. **Unit Tests:**
   ```python
   # Test volume validation logic
   python -m pytest tests/test_volume_validator.py
   ```

2. **Integration Tests:**
   ```python
   # Test full backtest scenarios
   python -m pytest tests/test_volume_backtest_integration.py
   ```

3. **Manual Testing:**
   ```python
   # Test with different configurations
   for min_volume in [5, 10, 15, 20]:
       config = VolumeConfig(min_volume=min_volume)
       # Run backtest and analyze results
   ```

## Conclusion

The volume validation system provides a robust way to ensure realistic backtesting by validating option liquidity. By following the best practices outlined in this guide, you can configure the system to match your specific trading requirements while maintaining performance and reliability.

For additional support or questions, refer to the main README.md or the test files for implementation examples. 