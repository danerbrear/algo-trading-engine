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

### Phase 4: HMM-Enabled Strategy Base Class

**Motivation**: To avoid look-ahead bias and ensure realistic market state classifications, create an `HMMStrategy` base class that provides HMM training capabilities. Strategies that need market regime filtering can inherit from this class, promoting code reuse and consistency.

**Implementation**: 
- New base class: `src/strategies/hmm_strategy.py` - `HMMStrategy` class
- Strategy implementation: `src/strategies/upward_trend_reversal_strategy.py`

**Key Design Principles**:
1. **Inheritance Hierarchy**: `HMMStrategy` inherits from `Strategy`, provides HMM capabilities
2. **Code Reuse**: All HMM-related code lives in one place
3. **Flexible Configuration**: Strategies can customize HMM training parameters
4. **No Global Side Effects**: Each strategy instance has its own HMM
5. **Backward Compatible**: Strategies that don't need HMM continue to inherit from `Strategy`

**New Base Class** (`src/backtest/models.py`):
```python
class HMMStrategy(Strategy):
    """
    Base class for strategies that use HMM for market regime classification.
    
    Provides HMM training, prediction, and persistence capabilities.
    Strategies that need market state filtering should inherit from this class.
    """
    
    def __init__(
        self,
        data_retriever: DataRetriever = None,
        train_hmm: bool = False,
        hmm_training_years: int = 2,
        save_trained_hmm: bool = False,
        hmm_model_dir: str = None,
        **kwargs
    ):
        """
        Initialize HMM-enabled strategy.
        
        Args:
            data_retriever: DataRetriever for fetching historical data
            train_hmm: Whether to train HMM on historical data
            hmm_training_years: Number of years of historical data for training
            save_trained_hmm: Whether to save trained HMM model
            hmm_model_dir: Directory for saving/loading HMM models
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(**kwargs)
        self.data_retriever = data_retriever
        self.train_hmm = train_hmm
        self.hmm_training_years = hmm_training_years
        self.save_trained_hmm = save_trained_hmm
        self.hmm_model_dir = hmm_model_dir
        self.hmm_model = None  # Trained HMM instance
    
    def set_data(self, data: pd.DataFrame, hmm_model=None, treasury_rates=None):
        """
        Set data for strategy and optionally train HMM.
        
        Overrides base Strategy.set_data() to add HMM training capability.
        """
        self.data = data
        self.treasury_rates = treasury_rates
        
        # Get first date from data
        if not data.empty:
            start_date = data.index[0]
            
            # Train HMM if requested (before validation)
            self._train_hmm_if_requested(start_date)
    
    def _train_hmm_if_requested(self, start_date: datetime):
        """
        Train HMM on historical data if training is enabled.
        
        Called during set_data() before data validation.
        
        Args:
            start_date: First date in backtest data
        """
        if not self.train_hmm or not self.data_retriever:
            return
        
        from datetime import timedelta
        from src.model.market_state_classifier import MarketStateClassifier
        
        # Calculate training period: N years before backtest start
        hmm_training_start = start_date - timedelta(days=self.hmm_training_years * 365)
        
        print(f"\n🎓 [{self.__class__.__name__}] Training HMM on {self.hmm_training_years} years of historical data")
        print(f"   Training period: {hmm_training_start.date()} to {start_date.date()}")
        
        # Fetch historical data
        hmm_data = self.data_retriever.fetch_data_for_period(hmm_training_start, 'hmm')
        
        # Filter to only include data before backtest start (avoid look-ahead bias)
        hmm_data = hmm_data[hmm_data.index < start_date]
        
        # Calculate features
        self.data_retriever.calculate_features_for_data(hmm_data)
        
        print(f"   Training on {len(hmm_data)} samples")
        
        # Train HMM model
        self.hmm_model = MarketStateClassifier()
        n_states = self.hmm_model.train_hmm_model(hmm_data)
        
        print(f"   ✅ HMM trained with {n_states} optimal states")
        
        # Apply trained HMM to backtest data
        self._apply_hmm_to_data()
        
        # Save if requested
        if self.save_trained_hmm:
            self._save_hmm_model()
    
    def _apply_hmm_to_data(self):
        """Apply trained HMM to predict market states for backtest data."""
        if self.data is None or self.hmm_model is None:
            return
        
        # Get features needed for HMM
        required_features = ['Returns', 'Volatility', 'Volume_Change']
        
        # Check if features exist
        missing_features = [f for f in required_features if f not in self.data.columns]
        if missing_features:
            print(f"   ⚠️  Warning: Missing features {missing_features}, cannot apply HMM")
            return
        
        features = self.hmm_model.scaler.transform(self.data[required_features])
        
        # Predict market states
        market_states = self.hmm_model.hmm_model.predict(features)
        
        # Add to data
        self.data['Market_State'] = market_states
        
        print(f"   ✅ Applied HMM predictions to {len(self.data)} backtest days")
    
    def _save_hmm_model(self):
        """Save trained HMM model for future use."""
        if self.hmm_model is None:
            return
        
        import pickle
        import os
        from datetime import datetime
        
        base_dir = self.hmm_model_dir or os.environ.get('MODEL_SAVE_BASE_PATH', 'Trained_Models')
        
        # Use strategy class name for mode (e.g., 'upward_trend_reversal_hmm')
        strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
        mode = f'{strategy_name}_hmm'
        
        # Get symbol from data_retriever or options_handler
        symbol = getattr(self.data_retriever, 'symbol', 'SPY')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp_dir = os.path.join(base_dir, mode, symbol, timestamp)
        latest_dir = os.path.join(base_dir, mode, symbol, 'latest')
        os.makedirs(timestamp_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        
        # Save HMM model and scaler with metadata
        hmm_data = {
            'hmm_model': self.hmm_model.hmm_model,
            'scaler': self.hmm_model.scaler,
            'n_states': self.hmm_model.n_states,
            'max_states': self.hmm_model.max_states,
            'training_years': self.hmm_training_years,
            'strategy': self.__class__.__name__,
            'timestamp': timestamp
        }
        
        hmm_path = os.path.join(timestamp_dir, 'hmm_model.pkl')
        hmm_latest_path = os.path.join(latest_dir, 'hmm_model.pkl')
        
        with open(hmm_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        with open(hmm_latest_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        print(f"   ✅ HMM model saved to {latest_dir}")
    
    def _get_hmm_mode_name(self) -> str:
        """
        Get the mode name for HMM model storage.
        
        Subclasses can override to customize storage location.
        
        Returns:
            str: Mode name for model storage
        """
        strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
        return f'{strategy_name}_hmm'
```

**Strategy Implementation** (`src/strategies/upward_trend_reversal_strategy.py`):
```python
class UpwardTrendReversalStrategy(HMMStrategy):  # Inherit from HMMStrategy
    """
    Strategy that trades put debit spreads on upward trend reversals.
    
    Inherits HMM training capabilities from HMMStrategy base class.
    """
    
    def __init__(
        self,
        options_handler: OptionsHandler,
        data_retriever: DataRetriever = None,
        train_hmm: bool = False,
        hmm_training_years: int = 2,
        save_trained_hmm: bool = False,
        hmm_model_dir: str = None,
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
        # Pass HMM parameters to base class
        super().__init__(
            data_retriever=data_retriever,
            train_hmm=train_hmm,
            hmm_training_years=hmm_training_years,
            save_trained_hmm=save_trained_hmm,
            hmm_model_dir=hmm_model_dir,
            profit_target=profit_target,
            stop_loss=stop_loss,
            start_date_offset=start_date_offset
        )
        
        # Strategy-specific parameters
        self.options_handler = options_handler
        self.min_trend_duration = min_trend_duration
        self.max_trend_duration = max_trend_duration
        self.max_spread_width = max_spread_width
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_risk_per_trade = max_risk_per_trade
        self.max_holding_days = max_holding_days
        
        # Track detected trends for analysis
        self.detected_trends: List[TrendInfo] = []
    
    # Strategy-specific methods remain the same
    # HMM training is handled by HMMStrategy base class
```

**HMM Strategy Builder Base** (`src/backtest/strategy_builder.py`):
```python
class HMMStrategyBuilder(StrategyBuilder):
    """
    Base builder for HMM-enabled strategies.
    
    Provides builder methods for HMM configuration.
    Subclasses should call super().__init__() and use these methods.
    """
    
    def __init__(self):
        super().__init__()
        self._data_retriever = None
        self._train_hmm = False
        self._hmm_training_years = 2
        self._save_trained_hmm = False
        self._hmm_model_dir = None
    
    def set_data_retriever(self, data_retriever: DataRetriever):
        """Set DataRetriever for HMM training."""
        self._data_retriever = data_retriever
        return self
    
    def set_train_hmm(self, train_hmm: bool):
        """Enable/disable HMM training."""
        self._train_hmm = train_hmm
        return self
    
    def set_hmm_training_years(self, years: int):
        """Set number of years for HMM training."""
        self._hmm_training_years = years
        return self
    
    def set_save_trained_hmm(self, save: bool):
        """Enable/disable saving trained HMM."""
        self._save_trained_hmm = save
        return self
    
    def set_hmm_model_dir(self, model_dir: str):
        """Set directory for loading/saving HMM models."""
        self._hmm_model_dir = model_dir
        return self
```

**Upward Trend Reversal Strategy Builder**:
```python
class UpwardTrendReversalStrategyBuilder(HMMStrategyBuilder):
    """Builder for UpwardTrendReversalStrategy."""
    
    def __init__(self):
        super().__init__()  # Initialize HMM builder parameters
        self._min_trend_duration = 3
        self._max_trend_duration = 4
        self._max_spread_width = 6.0
        self._min_dte = 5
        self._max_dte = 10
        self._max_risk_per_trade = 0.20
        self._max_holding_days = 2
    
    def set_min_trend_duration(self, duration: int):
        """Set minimum trend duration in days."""
        self._min_trend_duration = duration
        return self
    
    def set_max_trend_duration(self, duration: int):
        """Set maximum trend duration in days."""
        self._max_trend_duration = duration
        return self
    
    def set_max_spread_width(self, width: float):
        """Set maximum spread width."""
        self._max_spread_width = width
        return self
    
    def set_min_dte(self, dte: int):
        """Set minimum days to expiration."""
        self._min_dte = dte
        return self
    
    def set_max_dte(self, dte: int):
        """Set maximum days to expiration."""
        self._max_dte = dte
        return self
    
    def set_max_risk_per_trade(self, risk: float):
        """Set maximum risk per trade as fraction."""
        self._max_risk_per_trade = risk
        return self
    
    def set_max_holding_days(self, days: int):
        """Set maximum holding days."""
        self._max_holding_days = days
        return self
    
    def build(self) -> Strategy:
        """Build the strategy with all configured parameters."""
        if self._options_handler is None:
            raise ValueError("Missing required parameter: options_handler")
        
        strategy = UpwardTrendReversalStrategy(
            options_handler=self._options_handler,
            data_retriever=self._data_retriever,
            train_hmm=self._train_hmm,
            hmm_training_years=self._hmm_training_years,
            save_trained_hmm=self._save_trained_hmm,
            hmm_model_dir=self._hmm_model_dir,
            min_trend_duration=self._min_trend_duration,
            max_trend_duration=self._max_trend_duration,
            max_spread_width=self._max_spread_width,
            min_dte=self._min_dte,
            max_dte=self._max_dte,
            max_risk_per_trade=self._max_risk_per_trade,
            max_holding_days=self._max_holding_days,
            profit_target=self._profit_target,
            stop_loss=self._stop_loss,
            start_date_offset=self._start_date_offset
        )
        
        self.reset()
        return strategy
```

**Command-Line Integration** (`src/backtest/strategy_builder.py`):
```python
def create_strategy_from_args(
    strategy_name: str,
    data_retriever: DataRetriever,  # Pass data_retriever
    **kwargs
) -> Strategy:
    """Create strategy with HMM training support."""
    
    # Extract HMM training arguments
    train_hmm = kwargs.pop('train_hmm', False)
    hmm_training_years = kwargs.pop('hmm_training_years', 2)
    save_trained_hmm = kwargs.pop('save_trained_hmm', False)
    hmm_model_dir = kwargs.pop('hmm_model_dir', None)
    
    # Build strategy with HMM training options
    strategy = StrategyFactory.create_strategy(
        strategy_name,
        data_retriever=data_retriever,
        train_hmm=train_hmm,
        hmm_training_years=hmm_training_years,
        save_trained_hmm=save_trained_hmm,
        hmm_model_dir=hmm_model_dir,
        **kwargs
    )
    
    return strategy
```

**Updated Main Logic** (`src/backtest/main.py`):
```python
# Parse HMM training arguments
parser.add_argument('--train-hmm', action='store_true',
                   help='Train strategy-specific HMM on N years of prior data')
parser.add_argument('--hmm-training-years', type=int, default=2,
                   help='Years of historical data for HMM training (default: 2)')
parser.add_argument('--save-trained-hmm', action='store_true',
                   help='Save trained HMM model for future use')

# Create strategy with HMM training options
strategy = create_strategy_from_args(
    strategy_name=args.strategy,
    data_retriever=data_retriever,  # NEW: Pass data_retriever
    options_handler=options_handler,
    lstm_model=lstm_model,
    lstm_scaler=scaler,
    train_hmm=args.train_hmm,  # NEW
    hmm_training_years=args.hmm_training_years,  # NEW
    save_trained_hmm=args.save_trained_hmm,  # NEW
    stop_loss=args.stop_loss,
    profit_target=args.profit_target,
    start_date_offset=args.start_date_offset
)

# Set data (strategy will train HMM if requested)
strategy.set_data(data, hmm_model=None, treasury_rates=data_retriever.treasury_rates)
```

**Usage Examples**:
```bash
# Default: Use existing Market_State column from pre-trained HMM
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01

# Train strategy-specific HMM on 2 years of prior data
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01 \
    --train-hmm

# Train on 3 years and save for future use
python -m src.backtest.main --strategy upward_trend_reversal \
    --start-date 2023-01-01 --end-date 2024-01-01 \
    --train-hmm --hmm-training-years 3 --save-trained-hmm
```

**Benefits**:
- **Code Reuse**: All HMM logic in one base class, no duplication
- **Clean Inheritance**: `Strategy` → `HMMStrategy` → `UpwardTrendReversalStrategy`
- **Strategy Ownership**: Each strategy instance controls its HMM configuration
- **Avoids Look-Ahead Bias**: HMM only sees data before backtest period
- **Strategy-Specific Market Regimes**: Different strategies can define different regime criteria
- **No Side Effects**: Training doesn't affect other strategies
- **Flexible Configuration**: Easy per-strategy customization
- **Future-Proof**: New strategies can easily inherit HMM capabilities
- **Backward Compatible**: Non-HMM strategies continue to inherit from `Strategy`

**Inheritance Hierarchy**:
```
Strategy (base)
├── CreditSpreadStrategy (no HMM)
├── VelocityMomentumStrategy (no HMM)
└── HMMStrategy (HMM-enabled base)
    └── UpwardTrendReversalStrategy (uses HMM)
    └── Future HMM strategies...
```

**Testing Requirements**:
1. **HMMStrategy Base Class Tests**:
   - Test HMM training with various training periods
   - Test data filtering to avoid look-ahead bias
   - Test HMM application to backtest data
   - Test HMM model saving/loading
   - Test with missing features (error handling)
   - Test with no data_retriever provided

2. **UpwardTrendReversalStrategy Integration Tests**:
   - Test strategy inherits HMM capabilities correctly
   - Test strategy works with and without HMM training
   - Test HMM training doesn't interfere with strategy logic
   - Test strategy-specific HMM storage paths

3. **Builder Tests**:
   - Test HMMStrategyBuilder provides HMM configuration
   - Test UpwardTrendReversalStrategyBuilder inherits HMM builder methods
   - Test builder method chaining
   - Test builder passes all parameters correctly

**Migration Path for Existing Code**:
1. Remove HMM training from `src/backtest/main.py`
2. Create `HMMStrategy` base class in `src/strategies/hmm_strategy.py`
3. Update `UpwardTrendReversalStrategy` to inherit from `HMMStrategy`
4. Create `HMMStrategyBuilder` base class in `src/backtest/strategy_builder.py`
5. Update `UpwardTrendReversalStrategyBuilder` to inherit from `HMMStrategyBuilder`
6. Update `create_strategy_from_args()` to pass `data_retriever`
7. Update tests to reflect new architecture

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
- **Regime identification**: Use the bullish_regime_drawdown_analysis.py to model the market states we want to train the HMM against

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
