# Upward Trend Reversal Strategy

## Feature Description

Implement a new options trading strategy that capitalizes on upward trend reversals by trading SPY put debit spreads. The strategy identifies 3-10 day upward trends and enters put debit spread positions when these trends show signs of reversal, specifically targeting single-day trend-ending drawdowns.

## Overview

This strategy is designed to profit from the mean reversion that often occurs after sustained upward price movements. It combines technical trend analysis with market regime filtering to identify high-probability reversal opportunities.

### Key Principles
- **Trend Detection**: Identifies 3-10 consecutive days of positive returns
- **Reversal Timing**: Enters positions on the first day of negative returns after an upward trend
- **Market Regime Filtering**: Avoids positions during momentum uptrend market regimes
- **Risk Management**: Uses put debit spreads with strict position sizing and exit criteria

## Strategy Logic

### 1. Upward Trend Detection

**Definition**: An upward trend consists of 3-10 consecutive days of positive daily returns (close-to-close).

**Criteria**:
- **Minimum Duration**: 3 days (filters out noise)
- **Maximum Duration**: 10 days (focuses on meaningful but not extreme trends)
- **Trend End**: Immediately after one day of negative returns
- **Price Movement**: Must show net positive price increase over the trend period

**Example**:
```
Day 1: +0.5% (trend starts)
Day 2: +1.2% 
Day 3: +0.8%
Day 4: -0.3% (trend ends - 3-day upward trend)
```

### 2. Single-Day Trend-Ending Drawdown

**Definition**: The drawdown measured from the trend's end price to the close price on the day that ends the trend.

**Calculation**:
```
Drawdown = (Close_negative_day - Close_last_positive_day) / Close_last_positive_day
```

**Example**:
- Thursday close (last positive day): $500
- Friday close (first negative day): $495
- Drawdown = ($495 - $500) / $500 = -1.00%

### 3. Signal Generation

**Entry Signal Requirements**:
1. **Trend Criteria**:
   - Upward trend of 3-4 days duration
   - Net positive price increase over trend period
   - First day of negative returns after trend

2. **Market Regime Filter**:
   - Market must NOT be in MOMENTUM_UPTREND regime
   - Uses existing HMM market state classification

3. **Options Criteria**:
   - SPY Put Debit Spread
   - Width ≤ $6
   - 5-10 DTE (Days to Expiration)
   - Risk/Reward ratio validation

### 4. Position Management

**Position Sizing**:
- Maximum 1 open position at a time
- Position size based on maximum risk per trade of 20%
- Risk calculated as: (Strike Width - Net Debit) × 100 × Quantity

**Exit Criteria**:
1. **Stop Loss**: If stop loss threshold is met
2. **Profit Target**: If profit target is achieved
3. **Time-Based**: Close if holding period ≥ 2 days after trend reversal
4. **Expiration**: Close at 0 DTE

## Implementation Plan

### Phase 0: Debit Spread Support (Prerequisite)

**CRITICAL**: The current backtesting engine does NOT support put debit spreads. This must be implemented first.

#### Current Limitations
- **StrategyType Enum**: Only supports `CALL_CREDIT_SPREAD` and `PUT_CREDIT_SPREAD`
- **P&L Calculations**: Only handle credit spread logic in `Position` class methods
- **Position Management**: `BacktestEngine._add_position()` only handles credit spreads
- **Exit Price Calculations**: Only support credit spread pricing logic

#### Required Changes

1. **Add Debit Spread Strategy Types**:
   ```python
   # In src/backtest/models.py - StrategyType enum
   class StrategyType(Enum):
       CALL_CREDIT_SPREAD = "call_credit_spread"
       PUT_CREDIT_SPREAD = "put_credit_spread"
       CALL_DEBIT_SPREAD = "call_debit_spread"    # NEW
       PUT_DEBIT_SPREAD = "put_debit_spread"      # NEW
       # ... other types
   ```

2. **Update Position P&L Calculations**:
   ```python
   # In Position.get_return_dollars() method
   elif self.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
       # For debit spreads: entry_price = net debit paid when opening
       # exit_price = credit received when closing
       # Return = Credit received - Debit paid
       return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
   ```

3. **Update Position Management Logic**:
   ```python
   # In BacktestEngine._add_position() method
   elif position.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
       # For debit spreads: Deduct the net debit paid from capital
       debit_paid = position.entry_price * position.quantity * 100
       if self.capital < debit_paid:
           raise ValueError("Not enough capital to add debit spread position")
       self.capital -= debit_paid
   ```

4. **Update Exit Price Calculations**:
   ```python
   # In Position.calculate_exit_price() method
   elif self.strategy_type == StrategyType.PUT_DEBIT_SPREAD:
       # For put debit spread: buy ATM put, sell OTM put
       # Current net credit = OTM put price - ATM put price
       current_net_credit = current_otm_price - current_atm_price
       return current_net_credit
   elif self.strategy_type == StrategyType.CALL_DEBIT_SPREAD:
       # For call debit spread: buy OTM call, sell ATM call  
       # Current net credit = ATM call price - OTM call price
       current_net_credit = current_atm_price - current_otm_price
       return current_net_credit
   ```

5. **Update Assignment P&L Calculations**:
   ```python
   # In Position.get_return_dollars_from_assignment() method
   elif self.strategy_type == StrategyType.PUT_DEBIT_SPREAD:
       # Long ATM put, Short OTM put
       long_strike = atm_option.strike  # ATM strike
       short_strike = otm_option.strike # OTM strike (lower)
       
       # Calculate intrinsic values at expiration
       long_intrinsic = max(0, long_strike - underlying_price)
       short_intrinsic = max(0, short_strike - underlying_price)
       
       # Net P&L = Long leg value - Short leg cost - Initial debit
       net_pnl = long_intrinsic - short_intrinsic - self.entry_price
   ```

#### Testing Requirements
- Unit tests for debit spread P&L calculations
- Integration tests with backtest engine
- Validation of position sizing and capital management
- Test assignment scenarios at expiration

### Phase 1: Core Strategy Class

**File**: `src/strategies/upward_trend_reversal_strategy.py`

**Key Components**:

1. **Trend Detection Engine**:
   ```python
   def _detect_upward_trends(self, data: pd.DataFrame) -> List[TrendInfo]:
       """
       Detect upward trends in the data.
       
       Returns:
           List of TrendInfo objects containing:
           - start_date: datetime
           - end_date: datetime  
           - duration: int (days)
           - start_price: float
           - end_price: float
           - net_return: float
           - reversal_drawdown: float
       """
   ```

2. **Market Regime Integration**:
   ```python
   def _is_momentum_uptrend_regime(self, date: datetime) -> bool:
       """
       Check if market is in momentum uptrend regime.
       
       Uses existing HMM market state classification.
       """
   ```

3. **Options Chain Analysis**:
   ```python
   def _find_put_debit_spread(self, date: datetime, current_price: float) -> Optional[SpreadInfo]:
       """
       Find suitable put debit spread with:
       - Width ≤ $6
       - 5-10 DTE
       - Valid risk/reward ratio
       """
   ```

4. **Position Management**:
   ```python
   def _calculate_position_size(self, max_risk: float, spread_risk: float) -> int:
       """
       Calculate position size based on 20% max risk per trade.
       """
   ```

### Phase 2: Data Structures

**New Classes**:

1. **TrendInfo**:
   ```python
   @dataclass
   class TrendInfo:
       start_date: datetime
       end_date: datetime
       duration: int
       start_price: float
       end_price: float
       net_return: float
       reversal_drawdown: float
       reversal_date: datetime
   ```

2. **SpreadInfo**:
   ```python
   @dataclass
   class SpreadInfo:
       atm_put: Option
       otm_put: Option
       net_debit: float
       width: float
       max_risk: float
       max_reward: float
       risk_reward_ratio: float
       dte: int
   ```

### Phase 3: Strategy Integration

**Integration Points**:

1. **Strategy Builder**:
   - Add `UpwardTrendReversalStrategyBuilder` to `strategy_builder.py`
   - Register with `StrategyFactory`

2. **Backtest Engine**:
   - Ensure compatibility with existing backtesting framework
   - Support for position tracking and performance metrics

3. **CLI Integration**:
   - Add strategy option to `recommend_cli.py`
   - Support for interactive recommendations

### Phase 4: Configuration and Testing

**Configuration Options**:
```python
class UpwardTrendReversalStrategy(Strategy):
    def __init__(
        self,
        min_trend_duration: int = 3,
        max_trend_duration: int = 4,
        max_spread_width: float = 6.0,
        min_dte: int = 5,
        max_dte: int = 10,
        max_risk_per_trade: float = 0.20,
        max_holding_days: int = 2,
        profit_target: float = None,
        stop_loss: float = None,
        start_date_offset: int = 60
    ):
```

**Testing Strategy**:
1. **Unit Tests**: Test trend detection, spread selection, position sizing
2. **Integration Tests**: Test with historical data and market regime classification
3. **Backtest Validation**: Compare against benchmark and existing strategies

## Risk Management

### Position Sizing
- **Maximum Risk**: 20% of account per trade
- **Risk Calculation**: (Strike Width - Net Debit) × 100 × Quantity
- **Position Limit**: 1 open position at a time

### Exit Management
- **Stop Loss**: Configurable percentage-based stop loss
- **Profit Target**: Configurable profit target
- **Time Decay**: Close at 0 DTE to avoid assignment risk
- **Holding Period**: Maximum 2 days after trend reversal

### Market Regime Protection
- **Momentum Filter**: Avoid positions during momentum uptrend regimes
- **Volatility Consideration**: Monitor implied volatility for spread selection

## Performance Metrics

### Key Metrics to Track
1. **Win Rate**: Percentage of profitable trades
2. **Average Win/Loss**: Average profit vs average loss
3. **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio
4. **Maximum Drawdown**: Largest peak-to-trough decline
5. **Trade Frequency**: Number of trades per month/quarter

### Benchmark Comparison
- **SPY Buy & Hold**: Compare absolute returns
- **Existing Strategies**: Compare against velocity momentum strategy
- **Risk-Adjusted**: Compare Sharpe ratios and maximum drawdowns

## Implementation Timeline

### Week 1: Debit Spread Support (Prerequisite)
- [ ] Add `PUT_DEBIT_SPREAD` and `CALL_DEBIT_SPREAD` to StrategyType enum
- [ ] Update Position P&L calculations for debit spreads
- [ ] Update BacktestEngine position management for debit spreads
- [ ] Update exit price calculations for debit spreads
- [ ] Update assignment P&L calculations for debit spreads
- [ ] Create comprehensive tests for debit spread functionality

### Week 2: Core Strategy Implementation
- [ ] Implement `UpwardTrendReversalStrategy` class
- [ ] Create trend detection engine
- [ ] Implement market regime integration
- [ ] Basic unit tests

### Week 3: Options Integration
- [ ] Implement put debit spread selection
- [ ] Add position sizing logic
- [ ] Integrate with options handler
- [ ] Spread validation and risk/reward calculations

### Week 4: Backtest Integration
- [ ] Add strategy builder
- [ ] Integrate with backtest engine
- [ ] Add CLI support
- [ ] Performance metrics implementation

### Week 5: Testing and Validation
- [ ] Comprehensive backtesting
- [ ] Performance analysis
- [ ] Documentation updates
- [ ] Strategy optimization

## Dependencies

### Existing Components
- **Market State Classifier**: For regime filtering
- **Options Handler**: For options chain data
- **Backtest Engine**: For strategy execution
- **Position Management**: For position tracking

### New Dependencies
- **Debit Spread Support**: Core backtesting engine support for debit spreads (CRITICAL PREREQUISITE)
- **Trend Analysis**: Custom trend detection algorithms
- **Spread Selection**: Put debit spread optimization
- **Risk Management**: Position sizing and exit logic

## Success Criteria

### Functional Requirements
- [ ] **Debit Spread Support**: Backtesting engine fully supports put debit spreads
- [ ] Correctly identifies 3-10 day upward trends
- [ ] Accurately calculates trend-ending drawdowns
- [ ] Properly filters out momentum uptrend regimes
- [ ] Selects appropriate put debit spreads
- [ ] Manages position sizing and exits correctly

### Performance Requirements
- [ ] Positive risk-adjusted returns over backtest period
- [ ] Win rate ≥ 35% as specified
- [ ] Maximum drawdown < 15%
- [ ] Sharpe ratio > 1.0
- [ ] Consistent performance across different market conditions

### Integration Requirements
- [ ] Seamless integration with existing backtest framework
- [ ] Compatible with CLI recommendation system
- [ ] Proper logging and debugging capabilities
- [ ] Comprehensive test coverage

## Future Enhancements

### Potential Improvements
1. **Dynamic Trend Duration**: Adjust min/max trend duration based on market volatility
2. **Multi-Asset Support**: Extend beyond SPY to other liquid ETFs
3. **Machine Learning Integration**: Use LSTM predictions to enhance entry timing
4. **Volatility Filtering**: Add implied volatility considerations
5. **Earnings Calendar Integration**: Avoid positions around earnings events

### Advanced Features
1. **Portfolio Management**: Support for multiple concurrent positions
2. **Dynamic Position Sizing**: Adjust size based on market conditions
3. **Alternative Spreads**: Support for iron condors or other complex spreads
4. **Real-time Execution**: Integration with live trading systems
