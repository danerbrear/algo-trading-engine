# Capital Allocation and Risk Management Feature

## Overview
This feature implements capital allocation per strategy with risk-based position sizing when running `recommend_cli`. It ensures positions are only opened if the maximum risk is within the defined percentage of capital allocated for each strategy. Additionally, the equity curve visualization is updated to show remaining capital instead of cumulative P&L.

## Requirements

### 1. Capital Allocation Configuration
- Each strategy must have a defined capital allocation amount
- Each strategy must have a maximum risk percentage (e.g., 5% of allocated capital per position)
- Configuration should be stored in a JSON file for easy modification
- Support multiple strategies with independent capital allocations

### 2. Risk-Based Position Opening
- Before opening a position, calculate the maximum risk for that position
- Only open the position if: `max_risk < (allocated_capital * max_risk_percentage)`
- Reject position recommendations that exceed the risk threshold with a clear message
- Display available capital and risk check results in the recommendation summary

### 3. Capital Tracking
- Track capital remaining for each strategy separately
- Start with allocated capital amount for each strategy
- **When opening positions:**
  - **Credit strategies** (credit spreads, naked calls/puts sold): Add premium received to capital
  - **Debit strategies** (debit spreads, long calls/puts): Deduct premium paid from capital
  - Must ensure max risk is available (checked before opening, but not deducted upfront)
- **When closing positions:**
  - Apply realized P&L to capital (can be positive or negative)
  - Realized P&L = (premium received - premium paid) for closed position
- Maintain separate capital pools per strategy
- Support all option strategy types: credit spreads, debit spreads, naked calls/puts, iron condors, butterflies, etc.

### 4. Equity Curve Updates
- Change y-axis from "Cumulative P&L ($)" to "Capital Remaining ($)"
- For each strategy, track capital over time:
  - Start with allocated capital
  - Show capital remaining after each position closes
- Display capital remaining per strategy, not overall P&L
- Update plot labels and statistics to reflect capital tracking

### 5. Directory Reorganization
- Keep decision JSON files in `predictions/decisions/` (no change needed)
- Create new `config/strategies/capital_allocations.json` for capital configuration

## Implementation Details

### Configuration Structure

**File: `config/strategies/capital_allocations.json`**
```json
{
  "strategies": {
    "credit_spread": {
      "allocated_capital": 10000.0,
      "max_risk_percentage": 0.05
    },
    "velocity_momentum": {
      "allocated_capital": 15000.0,
      "max_risk_percentage": 0.03
    }
  }
}
```

### Risk Calculation
- Max risk calculation depends on strategy type:
  - **Credit spreads**: `(spread_width - net_credit) * 100 * quantity`
  - **Debit spreads**: `(spread_width - net_debit) * 100 * quantity` (max loss is the debit paid)
  - **Naked options**: Variable based on strike and margin requirements
- Risk check: `max_risk <= (allocated_capital * max_risk_percentage)`
- **Capital availability check:**
  - **Credit strategies**: Must have sufficient capital to cover max loss (max_risk) in case of adverse move
  - **Debit strategies**: Must have sufficient capital to pay premium + cover max loss
- If check fails, reject the position with a clear message showing:
  - Requested max risk
  - Allocated capital
  - Max risk percentage
  - Maximum allowed risk

### Capital Tracking
- Maintain a `CapitalManager` class that:
  - Loads capital allocations from config
  - **Derives remaining capital by calculating from decisions JSON files** (single source of truth)
  - Calculation method:
    1. Start with allocated capital from `capital_allocations.json`
    2. Load all decision JSON files for the strategy
    3. For each accepted decision:
       - **Opening credit positions**: Add premium received (`entry_price`) to capital
       - **Opening debit positions**: Deduct premium paid (`entry_price`) from capital
       - **Closing credit positions**: Subtract premium paid to close (`exit_price`) from capital
       - **Closing debit positions**: Add premium received when closing (`exit_price`) to capital
    4. For open positions (no `closed_at`): Only apply opening premium, no closing adjustment yet
  - Determines position type from `StrategyType` enum (credit vs debit)
  - Supports all strategy types: credit spreads, debit spreads, naked calls/puts, iron condors, butterflies, etc.
  - **No separate state file needed** - always derived from decisions (ensures accuracy even if decisions are manually edited)

**Capital Calculation Formula:**
```
Remaining Capital = Allocated Capital (from config)
                  + Sum(entry_price for all credit positions opened)
                  - Sum(entry_price for all debit positions opened)
                  - Sum(exit_price for all closed credit positions)  // Pay to close
                  + Sum(exit_price for all closed debit positions)    // Receive when closing
                  
Note: Realized P&L for closed positions:
  - Credit spreads: entry_price - exit_price (credit received - debit paid)
  - Debit spreads: exit_price - entry_price (credit received - debit paid)
```

**Calculation Method:**
1. Load all decision JSON files via `JsonDecisionStore`
2. Filter decisions by `strategy_name` from proposal
3. Filter to `outcome == "accepted"`
4. For each decision:
   - If credit strategy: add `entry_price` to running total (premium received when opening)
   - If debit strategy: subtract `entry_price` from running total (premium paid when opening)
   - If `closed_at` exists (position closed):
     - Credit strategy: subtract `exit_price` from running total (premium paid to close)
     - Debit strategy: add `exit_price` to running total (premium received when closing)

**Example:**
- Allocated capital: $10,000 (from config)
- Decision 1: Credit spread opened, entry_price $1.86 (received credit) → Capital = $10,000 + $1.86 = $10,186
- Decision 2: Debit spread opened, entry_price $1.50 (paid debit) → Capital = $10,186 - $1.50 = $10,036
- Decision 1 closed: exit_price $2.95 (paid debit to close) → Capital = $10,036 - $2.95 = $9,741
  - Realized P&L: $1.86 - $2.95 = -$1.09 (loss)
- Decision 2 closed: exit_price $2.00 (received credit when closing) → Capital = $9,741 + $2.00 = $9,743
  - Realized P&L: $2.00 - $1.50 = +$0.50 (profit)

### Changes to Existing Components

#### `recommend_cli.py`
- Load capital allocation configuration
- Initialize `CapitalManager` with allocations
- Pass capital manager to `InteractiveStrategyRecommender`
- Display capital status in output

#### `recommendation_engine.py`
- Add capital manager dependency (which loads decision store)
- Check risk before accepting position recommendations
- Capital is recalculated from decisions each time (no manual updates needed)
- When opening/closing positions, decisions are saved to JSON, and capital manager will recalculate on next call
- Determine strategy type (credit vs debit) from `StrategyType` enum
- Display capital and risk information in recommendation summaries (capital recalculated from decisions)
- Support all option strategy types

#### `plot_equity_curve.py`
- Change from P&L tracking to capital tracking
- Load capital allocations to determine starting capital
- Calculate capital remaining per strategy (based on realized P&L only):
  - Start with allocated capital
  - For each closed position: apply realized P&L
    - Credit spreads: `entry_price - exit_price` (credit received - debit paid to close)
    - Debit spreads: `exit_price - entry_price` (credit received when closing - debit paid to open)
  - Open positions (not closed yet) do not affect the equity curve
- Update plot labels: "Capital Remaining ($)" instead of "Cumulative P&L ($)"
- Update statistics to show capital remaining instead of total P&L
- Track capital changes over time showing only realized gains/losses

#### `decision_store.py`
- No changes needed - existing structure is sufficient
- Capital manager will read from existing decision fields: `entry_price`, `exit_price`, `outcome`, `closed_at`, `proposal.strategy_type`, `proposal.strategy_name`

## File Structure

```
config/
  strategies/
    capital_allocations.json    # NEW: Capital allocation configuration

predictions/
  decisions/                   # Keep existing location
    decisions_*.json

src/
  prediction/
    capital_manager.py         # NEW: Capital tracking and risk checking
    recommend_cli.py           # MODIFY: Load and use capital manager
    recommendation_engine.py   # MODIFY: Risk checking before opening positions
    plot_equity_curve.py       # MODIFY: Show capital remaining instead of P&L
    decision_store.py          # MODIFY: Optional capital tracking in decisions
```

## API Changes

### New: `CapitalManager` Class
```python
class CapitalManager:
    def __init__(self, allocations_config: dict, decision_store: JsonDecisionStore)
    def get_allocated_capital(self, strategy_name: str) -> float
    def get_remaining_capital(self, strategy_name: str) -> float  # Calculated from decisions
    def check_risk_threshold(self, strategy_name: str, max_risk: float) -> bool
    def is_credit_strategy(self, strategy_type: StrategyType) -> bool
    def _calculate_remaining_capital(self, strategy_name: str) -> float
    # No apply_premium/apply_realized_pnl needed - always calculated from decisions
```

**Strategy Type Detection:**
- Credit strategies (receive premium, add to capital):
  - `PUT_CREDIT_SPREAD`: Receive credit when opening
  - `CALL_CREDIT_SPREAD`: Receive credit when opening
  - `SHORT_CALL`: Naked short call, receive premium
  - `SHORT_PUT`: Naked short put, receive premium
- Debit strategies (pay premium, subtract from capital):
  - `CALL_DEBIT_SPREAD`: Pay debit when opening (buy call, sell call at higher strike)
  - `PUT_DEBIT_SPREAD`: Pay debit when opening (buy put, sell put at lower strike)
  - `LONG_CALL`: Pay premium when opening
  - `LONG_PUT`: Pay premium when opening
  - `LONG_STOCK`: Pay for stock (equivalent to debit)
- Premium handling:
  - Credit strategies: Premium received adds to capital
  - Debit strategies: Premium paid subtracts from capital
- **Note**: If `CALL_DEBIT_SPREAD` and `PUT_DEBIT_SPREAD` are not yet in the `StrategyType` enum, they should be added
- Future strategy types (iron condors, butterflies, etc.) should be categorized based on whether they receive or pay net premium

### Modified: `InteractiveStrategyRecommender`
- Add `capital_manager: CapitalManager` parameter to `__init__`
- Modify `recommend_open_position` to check risk before accepting
- Modify `recommend_close_positions` to update capital tracking

### Modified: `plot_equity_curve.py`
- Change `ClosedPosition` to include capital tracking fields
- Modify `calculate_equity_curve` to use capital remaining instead of cumulative P&L
- Update plot labels and statistics

## Migration Notes
- No backward compatibility fallbacks required per user request
- Existing decision JSON files are the source of truth for capital tracking
- Capital is always calculated from decisions, ensuring accuracy even if files are manually edited
- Users must create `config/strategies/capital_allocations.json` before using feature
- Default behavior: if config missing, print error and exit
- No separate capital state file needed - always derived from decisions

## Testing Considerations
- Test risk threshold rejection when max_risk exceeds limit
- Test capital calculation from decisions JSON files (credit spreads)
- Test capital calculation from decisions JSON files (debit spreads)
- Test capital calculation with multiple positions of different types
- Test capital recalculation after manually editing decision JSON files
- Test equity curve with capital remaining calculation
- Test with multiple strategies having different allocations
- Test capital depletion scenarios
- Test strategy type detection for credit vs debit strategies
- Test with various strategy types: iron condors, butterflies, naked options, etc.
- Verify capital matches expected values after adding/removing decisions manually

## Example Usage

### Running with Capital Allocation
```bash
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --date 2025-10-30
```

### Expected Output - Credit Spread
```
📊 Capital Allocation: $15,000.00 for velocity_momentum
💰 Remaining Capital: $15,000.00
📈 Max Risk Per Position: $450.00 (3.0% of allocated capital)

Open recommendation:
Symbol: SPY
Strategy: put_credit_spread
Legs: PUT 685 exp 2025-11-04, PUT 679 exp 2025-11-04
Credit: $1.86  Width: 6.0  R/R: 0.31  Prob: 70%
Max Risk: $414.00
Premium received: $186.00 (credit strategy)
✅ Risk check passed ($414.00 <= $450.00)

Open this position? [y/N]: y
✅ Position opened. Capital updated: $15,186.00 (+$186.00 premium received)
```

### Expected Output - Debit Spread
```
📊 Capital Allocation: $10,000.00 for credit_spread
💰 Remaining Capital: $10,186.00
📈 Max Risk Per Position: $500.00 (5.0% of allocated capital)

Open recommendation:
Symbol: SPY
Strategy: call_debit_spread
Legs: CALL 685 exp 2025-11-04, CALL 695 exp 2025-11-04
Debit: $1.50  Width: 10.0  R/R: 0.67  Prob: 65%
Max Risk: $850.00
Premium paid: $150.00 (debit strategy)
❌ Risk check failed: $850.00 exceeds maximum allowed risk of $500.00 (5.0% of $10,000.00)
Position rejected due to risk threshold.
```

### Closing Position Example
```
Closing position:
Position: SPY PUT 685/679 credit spread
Entry: $1.86 (received $186.00)
Exit: $2.95 (paid $295.00)
Realized P&L: -$109.00
💰 Capital updated: $15,077.00 ($15,186.00 - $109.00 P&L)
```

## Future Enhancements (Out of Scope)
- Automatic position sizing to maximize capital usage within risk limits
- Capital rebalancing between strategies
- Historical capital allocation tracking
- Capital withdrawal/deposit tracking

