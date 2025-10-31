# Prediction Module

The prediction module provides tools for generating trading recommendations and analyzing performance through equity curve visualization.

## Overview

This module contains two main components:

1. **Interactive Recommendation CLI** (`recommend_cli.py`) - Generate and manage trading recommendations
2. **Equity Curve Plotter** (`plot_equity_curve.py`) - Visualize and analyze trading performance

---

## 📊 Interactive Recommendation CLI

The `recommend_cli.py` script provides an interactive interface for generating trading recommendations based on your strategies.

### Basic Usage

```bash
# Basic run (interactive prompts)
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum

# Specify a specific date (YYYY-MM-DD)
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --date 2025-10-15

# Non-interactive (auto-accept all recommendations)
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --yes

# Auto-close open positions (uses previous day's prices)
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --auto-close
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `SPY` | Underlying symbol to trade |
| `--strategy` | `credit_spread` | Strategy to use (`credit_spread`, `velocity_momentum`) |
| `--date` | Today | Run date in `YYYY-MM-DD` format |
| `--yes` | `False` | Auto-accept all prompts (non-interactive mode) |
| `--auto-close` | `False` | Automatically close positions recommended to close |
| `--verbose` | `False` | Enable verbose output |
| `-f, --free` | `False` | Use free tier rate limiting (13s timeout between requests) |

### Workflow

1. **Check for Open Positions**: 
   - If open positions exist, the CLI displays their current status (P&L, DTE, days held)
   - Only the close flow runs if positions are open

2. **Generate Recommendations**:
   - Strategy analyzes current market conditions
   - Generates position recommendations with confidence scores
   - Displays proposal details (strikes, expiration, premium, probability of profit)

3. **User Decision**:
   - Interactive: User accepts or rejects each recommendation
   - Non-interactive (`--yes`): All recommendations are automatically accepted

4. **Decision Storage**:
   - Accepted decisions are saved to `predictions/decisions/decisions_YYYYMMDD.json`
   - Each decision includes entry price, strategy name, and rationale

5. **Close Positions**:
   - Check if open positions should be closed (stop loss, profit target, expiration, etc.)
   - Calculate exit prices and P&L
   - Update decision records with `closed_at` and `exit_price`

### Decision JSON Format

Decisions are stored in JSON files with the following structure:

```json
{
  "id": "unique_decision_id",
  "decided_at": "2025-10-27T21:15:24.433232+00:00",
  "outcome": "accepted",
  "entry_price": 2.55,
  "exit_price": 2.01,
  "closed_at": "2025-10-28T19:20:54.553040",
  "quantity": 1,
  "rationale": "strategy_confidence=0.70",
  "proposal": {
    "symbol": "SPY",
    "strategy_name": "velocity_signal_momentum",
    "strategy_type": "put_credit_spread",
    "legs": [...],
    "credit": 2.55,
    "width": 10.0,
    "confidence": 0.7,
    "probability_of_profit": 0.7,
    "expiration_date": "2025-11-04",
    "created_at": "2025-10-27T21:11:49.523546+00:00"
  }
}
```

### Example Output

```
No open positions found, running open flow
📊 Analyzing market conditions for SPY on 2025-10-27...

Open recommendation:
Symbol: SPY
Strategy: Put Credit Spread
Strike: 685/675
Expiration: 2025-11-04
Premium: $2.55
Width: $10.00
Probability of Profit: 70.0%
Confidence: 0.70

Open this position? (y/n): y
✅ Position opened and saved to decisions_20251027.json
```

---

## 📈 Equity Curve Plotter

The `plot_equity_curve.py` script generates equity curves and performance statistics from closed trading positions.

### Basic Usage

```bash
# Plot all strategies
python -m src.prediction.plot_equity_curve

# Plot specific strategy
python -m src.prediction.plot_equity_curve --strategy velocity_signal_momentum

# Print summary statistics only (no plot)
python -m src.prediction.plot_equity_curve --summary-only

# Save plot to file without displaying
python -m src.prediction.plot_equity_curve --output equity_curve.png --no-show
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--strategy` | Filter by strategy name (e.g., `velocity_signal_momentum`, `upward_trend_reversal`) |
| `--output, -o` | Save plot to file (e.g., `equity_curve.png`) |
| `--no-show` | Don't display plot interactively (only save to file) |
| `--summary-only` | Print statistics only, don't generate plot |
| `--decisions-dir` | Directory containing decision JSON files (default: `predictions/decisions`) |

### Features

#### Equity Curve Visualization

- **Cumulative P&L over time** - See how your strategy performance evolves
- **Multi-strategy comparison** - Compare different strategies side-by-side
- **Per-strategy filtering** - Focus on specific strategy performance
- **Performance metrics** - Win rate, total P&L, trade count displayed in legend

#### Summary Statistics

The script provides comprehensive performance analysis:

```
================================================================================
TRADING SUMMARY
================================================================================

Overall Performance:
  Total P&L: $289.00
  Total Trades: 2
  Wins: 2 | Losses: 0
  Win Rate: 100.0%
  Avg P&L per Trade: $144.50

By Strategy:
  Velocity Signal Momentum:
    P&L: $289.00
    Trades: 2
    Win Rate: 100.0%
    Avg P&L: $144.50
```

### P&L Calculation

The script automatically calculates P&L based on strategy type:

- **Credit Spreads** (e.g., Put Credit Spread):
  ```
  P&L = (Entry Premium - Exit Premium) × Quantity × 100
  ```
  Profit when exit premium < entry premium (spread loses value)

- **Debit Spreads** (e.g., Put Debit Spread):
  ```
  P&L = (Exit Premium - Entry Premium) × Quantity × 100
  ```
  Profit when exit premium > entry premium (spread gains value)

### Example Output

```
📂 Loading closed positions from predictions/decisions...
✅ Loaded 2 closed positions

================================================================================
TRADING SUMMARY
================================================================================
[... statistics ...]

📈 Generating equity curve plot...
✅ Plot saved to: equity_curve.png
```

---

## 🔄 Integration with VS Code Tasks

Both scripts are available as VS Code tasks for easy access:

- **Equity Curve: Plot All Strategies** - Generate full equity curve visualization
- **Equity Curve: Summary Only** - Get statistics without plotting
- **Equity Curve: Velocity Momentum** - Filter to specific strategy

Access via: `Cmd+Shift+P` → "Tasks: Run Task"

---

## 📁 File Structure

```
src/prediction/
├── README.md                 # This file
├── recommend_cli.py         # Interactive recommendation CLI
├── plot_equity_curve.py     # Equity curve plotting tool
├── recommendation_engine.py  # InteractiveStrategyRecommender class
└── decision_store.py        # JSON decision storage

predictions/                  # Output directory (workspace root)
└── decisions/                # Decision JSON files
    ├── decisions_20251027.json
    ├── decisions_20251030.json
    └── ...
```

---

## 💡 Best Practices

### Recommendation CLI

1. **Daily Workflow**: Run the CLI once per day to check for new recommendations and close positions
   ```bash
   python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum
   ```

2. **Auto-Close Mode**: Use `--auto-close` for automated position management
   ```bash
   python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --auto-close --yes
   ```

3. **Historical Analysis**: Test strategies on historical dates
   ```bash
   python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --date 2025-09-15
   ```

### Equity Curve Analysis

1. **Regular Monitoring**: Track performance regularly
   ```bash
   python -m src.prediction.plot_equity_curve --summary-only
   ```

2. **Strategy Comparison**: Compare multiple strategies
   ```bash
   python -m src.prediction.plot_equity_curve  # Shows all strategies
   ```

3. **Export for Reporting**: Save plots for documentation
   ```bash
   python -m src.prediction.plot_equity_curve --output reports/equity_oct_2025.png --no-show
   ```

---

## 🔧 Troubleshooting

### "No closed positions found"
- Ensure decision files exist in `predictions/decisions/`
- Check that positions have `closed_at` field set
- Verify `outcome` is `"accepted"` for positions to be included

### "No open positions found"
- This is normal when starting fresh or after all positions are closed
- The CLI will proceed to the open flow to generate new recommendations

### Matplotlib/NumPy Issues on macOS
If you encounter floating point exceptions when generating plots:
- Use `--summary-only` flag to get statistics without plotting
- Save to file instead: `--output equity.png --no-show`
- Update NumPy: `pip install --upgrade numpy`

### Strategy Name Not Found
- Ensure decision files have been migrated: Check for `strategy_name` field in proposal
- Re-run migration if needed (though new decisions automatically include it)

---

## 📚 Related Documentation

- **[Strategy Documentation](../strategies/README.md)** - Trading strategy implementations
- **[Backtest Documentation](../backtest/README.md)** - Backtesting framework
- **[Model Documentation](../model/README.md)** - ML models and training

---

## 🚀 Quick Reference

### Recommendation CLI Commands

```bash
# Interactive mode (default)
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum

# Auto-accept mode
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --yes

# Auto-close mode
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --auto-close

# Specific date
python -m src.prediction.recommend_cli --symbol SPY --strategy velocity_momentum --date 2025-10-15
```

### Equity Curve Commands

```bash
# Plot all
python -m src.prediction.plot_equity_curve

# Filter by strategy
python -m src.prediction.plot_equity_curve --strategy velocity_signal_momentum

# Statistics only
python -m src.prediction.plot_equity_curve --summary-only

# Save to file
python -m src.prediction.plot_equity_curve --output equity.png --no-show
```

