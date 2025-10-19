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

6. **Update Max Risk Calculation**:
   ```python
   # In Position.get_max_risk() method
   def get_max_risk(self):
       """
       Determine the max loss for a position.
       """
       atm_option, otm_option = self.spread_options
       width = abs(atm_option.strike - otm_option.strike)
       
       if self.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
           # Credit spread: Max risk = Width - Net Credit
           net_credit = atm_option.last_price - otm_option.last_price
           return (width - net_credit) * 100
       elif self.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
           # Debit spread: Max risk = Net Debit paid
           net_debit = abs(atm_option.last_price - otm_option.last_price)
           return net_debit * 100
       else:
           # Fallback for other position types
           net_credit = atm_option.last_price - otm_option.last_price
           return (width - net_credit) * 100
   ```

7. **Update Recommendation Engine for Debit Spreads**:

   The recommendation engine (`src/prediction/`) currently assumes all spreads are credit spreads. Changes needed:

   **a) Update ProposedPositionRequest DTO**:
   
   Rename `credit` field to `premium` and use signed values:
   - Positive for credit spreads (money received)
   - Negative for debit spreads (money paid)
   
   ```python
   # In src/prediction/decision_store.py
   @dataclass(frozen=True)
   class ProposedPositionRequest:
       symbol: str
       strategy_type: StrategyType
       legs: Tuple[Option, ...]
       premium: float  # Renamed from 'credit', positive=credit, negative=debit
       width: float
       probability_of_profit: float
       confidence: float
       expiration_date: str
       created_at: str
   ```
   
   Update all references to `credit` throughout the codebase to use `premium`.

   **b) Update InteractiveStrategyRecommender**:
   ```python
   # In src/prediction/recommendation_engine.py
   
   def _format_open_summary(self, proposal: ProposedPositionRequest, best: dict) -> str:
       # Update display logic to show "Credit" vs "Debit" based on premium sign
       if proposal.premium < 0:
           cost_label = "Debit"
           cost_value = abs(proposal.premium)  # Display as positive
       else:
           cost_label = "Credit"
           cost_value = proposal.premium
       
       return (
           f"Symbol: {proposal.symbol}\n"
           f"Strategy: {proposal.strategy_type.value}\n"
           f"Legs: {legs_str}\n"
           f"{cost_label}: ${cost_value:.2f}  Width: {proposal.width}  R/R: {rr}  Prob: {proposal.probability_of_profit:.0%}"
       )
   
   def _position_from_decision(self, rec: DecisionResponse) -> Position:
       # Add logic for debit spread strike determination
       if rec.proposal.strategy_type == StrategyType.PUT_DEBIT_SPREAD:
           # Long is higher strike (ATM), short is lower strike (OTM)
           strike = max(leg.strike for leg in legs)
       elif rec.proposal.strategy_type == StrategyType.CALL_DEBIT_SPREAD:
           # Long is higher strike (OTM), short is lower strike (ATM)
           strike = max(leg.strike for leg in legs)
   ```

   **c) Update Capital Management Display**:
   ```python
   # In recommendation_engine.py _format_close_summary()
   def _format_close_summary(self, position: Position, exit_price: float, rationale: str) -> str:
       if position.strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
           # For debit spreads: Return = Exit Credit - Entry Debit
           pnl_dollars = position.get_return_dollars(exit_price)
           pnl_pct = position._get_return(exit_price) * 100
       else:
           # For credit spreads: Return = Entry Credit - Exit Cost
           pnl_dollars = position.get_return_dollars(exit_price)
           pnl_pct = position._get_return(exit_price) * 100
   ```

   **d) Update CLI Documentation**:
   - Update help text in `recommend_cli.py` to mention support for both credit and debit spreads
   - Update example output in documentation to show debit spread examples

#### Testing Requirements
- Unit tests for debit spread P&L calculations
- Integration tests with backtest engine
- Validation of position sizing and capital management
- Test assignment scenarios at expiration
- **Test recommendation engine with debit spread proposals**
- **Test decision store serialization/deserialization with debit spreads**
- **Test CLI display formatting for debit vs credit spreads**

### Phase 1: Core Strategy Class

**File**: `src/strategies/upward_trend_reversal_strategy.py`

**Critical Requirements**:
- **MUST use** `src/common/options_handler.py` (OptionsHandler)
- **DO NOT use** `src/model/options_handler.py` (legacy, deprecated)
- Follow existing strategy patterns from `velocity_signal_momentum_strategy.py`

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
       
       Uses src/common/options_handler.py for fetching option chain data.
       Uses src/common/options_helpers.py for strike selection and filtering.
       """
   ```

4. **Position Management**:
   ```python
   def _calculate_position_size(self, position: Position, max_risk_pct: float = 0.20) -> int:
       """
       Calculate position size based on max risk percentage per trade.
       
       Uses Position.get_max_risk() to calculate per-contract risk.
       
       Args:
           position: Position object with spread_options set
           max_risk_pct: Maximum risk as percentage of capital (default: 0.20 = 20%)
           
       Returns:
           int: Number of contracts to trade
       """
       max_risk_dollars = self.capital * max_risk_pct
       per_contract_risk = position.get_max_risk()  # Uses built-in method
       quantity = int(max_risk_dollars / per_contract_risk)
       return max(1, quantity)  # At least 1 contract
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

### Phase 4: HMM Training Options for Backtest Engine

**Motivation**: To avoid look-ahead bias and ensure realistic market state classifications, add the ability to train the HMM model on data prior to the backtest period rather than always loading a pre-trained model.

**Implementation**: `src/backtest/main.py`

**New Command-Line Arguments**:
```python
parser.add_argument('--train-hmm', action='store_true', 
                   help='Train a new HMM model on 2 years of data prior to backtest start date. '
                        'If not specified, loads pre-trained model from MODEL_SAVE_BASE_PATH.')

parser.add_argument('--hmm-training-years', type=int, default=2,
                   help='Number of years of historical data to use for HMM training (default: 2)')

parser.add_argument('--save-trained-hmm', action='store_true',
                   help='Save the newly trained HMM model. Only applies when --train-hmm is used.')
```

**Training Function** (follows existing pattern from `src/model/main.py:prepare_data()`):
```python
def train_hmm_for_backtest(data_retriever, start_date, training_years=2):
    """
    Train a new HMM model on historical data prior to backtest start date.
    
    Uses the same training pattern as StockPredictor.prepare_data() in src/model/main.py
    
    Args:
        data_retriever: DataRetriever instance
        start_date: Backtest start date
        training_years: Number of years of historical data to use
        
    Returns:
        MarketStateClassifier: Trained HMM model
    """
    from datetime import timedelta
    from src.model.market_state_classifier import MarketStateClassifier
    
    # Calculate training period: N years before backtest start
    hmm_training_start = start_date - timedelta(days=training_years * 365)
    
    print(f"\n📈 Preparing HMM training data from {hmm_training_start.date()} to {start_date.date()}")
    
    # Fetch historical data (same as StockPredictor.prepare_data)
    hmm_data = data_retriever.fetch_data_for_period(hmm_training_start, 'hmm')
    
    # Filter to only include data before backtest start
    hmm_data = hmm_data[hmm_data.index < start_date]
    
    # Calculate features (same as StockPredictor.prepare_data)
    data_retriever.calculate_features_for_data(hmm_data)
    
    print(f"🎯 Training HMM on market data ({len(hmm_data)} samples)")
    
    # Train HMM model (same as StockPredictor.prepare_data)
    hmm_model = MarketStateClassifier()
    n_states = hmm_model.train_hmm_model(hmm_data)
    
    print(f"✅ HMM model trained with {n_states} optimal states")
    
    return hmm_model
```

**Saving Function** (reuses existing `save_model()` from `src/model/main.py`):
```python
def save_hmm_only(hmm_model, mode='backtest_hmm', symbol='SPY'):
    """
    Save just the HMM model using the existing save infrastructure.
    
    Follows the same pattern as save_model() in src/model/main.py but saves only HMM.
    """
    import pickle
    import os
    from datetime import datetime
    
    base_dir = os.environ.get('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp_dir = os.path.join(base_dir, mode, symbol, timestamp)
    latest_dir = os.path.join(base_dir, mode, symbol, 'latest')
    os.makedirs(timestamp_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    
    # Save HMM model and scaler (same format as save_model())
    hmm_data = {
        'hmm_model': hmm_model.hmm_model,
        'scaler': hmm_model.scaler,
        'n_states': hmm_model.n_states,
        'max_states': hmm_model.max_states
    }
    
    hmm_path = os.path.join(timestamp_dir, 'hmm_model.pkl')
    hmm_latest_path = os.path.join(latest_dir, 'hmm_model.pkl')
    
    with open(hmm_path, 'wb') as f:
        pickle.dump(hmm_data, f)
    with open(hmm_latest_path, 'wb') as f:
        pickle.dump(hmm_data, f)
    
    print(f"✅ HMM model saved to {hmm_path} and {hmm_latest_path}")
```

**Updated Main Logic**:
```python
if args.train_hmm:
    # Train new HMM on data prior to backtest
    hmm_model = train_hmm_for_backtest(data_retriever, start_date, args.hmm_training_years)
    
    if args.save_trained_hmm:
        # Save using existing pattern from src/model/main.py:save_model()
        save_hmm_only(hmm_model, mode='backtest_hmm', symbol=args.symbol)
else:
    # Load pre-trained HMM using existing loader from src/common/functions.py
    hmm_model = load_hmm_model(model_dir)
```

**Usage Examples**:
```bash
# Default: Load pre-trained HMM
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01

# Train new HMM on 2 years of prior data
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01 --train-hmm

# Train on 3 years and save
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01 \
    --train-hmm --hmm-training-years 3 --save-trained-hmm
```

**Benefits**:
- **Avoids Look-Ahead Bias**: HMM only sees data before backtest period
- **Flexibility**: Easy experimentation with different training periods
- **Reproducibility**: Clear separation of training/testing data
- **Strategy-Specific**: Can train HMM specifically for reversal strategy testing

### Phase 5: Strategy Configuration and Testing

**Strategy Configuration Options**:
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
4. **HMM Training Tests**: Verify HMM trains correctly on prior data without look-ahead bias

## Risk Management

### Position Sizing
- **Maximum Risk**: 20% of account per trade
- **Risk Calculation**: Uses `Position.get_max_risk()` method
  - For debit spreads: `(width - net_credit) * 100` per contract
  - Note: `get_max_risk()` currently uses credit spread formula; needs update for debit spreads
- **Position Limit**: 1 open position at a time
- **Quantity Calculation**: `floor(capital * 0.20 / position.get_max_risk())`

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

## Dependencies

### Existing Components
- **Market State Classifier**: For regime filtering (`src/model/market_state_classifier.py`)
- **Options Handler**: For options chain data
  - **REQUIREMENT**: Use `src/common/options_handler.py` ONLY
  - **DO NOT USE**: `src/model/options_handler.py` (legacy, being phased out)
- **Backtest Engine**: For strategy execution (`src/backtest/main.py`)
- **Position Management**: For position tracking (`src/backtest/models.py`)

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
