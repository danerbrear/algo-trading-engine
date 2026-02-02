# Options Trading System

This project provides a comprehensive backtesting and prediction platform for options trading strategies. The system enables users to develop, test, and evaluate trading strategies using historical data, while also providing real-time predictions for current market conditions. It includes a robust backtesting framework with volume validation, risk management tools, and machine learning-powered strategy prediction capabilities.

## 📦 Usage

### Installation

Add `algo-trading-engine` to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "algo-trading-engine @ git+https://github.com/danerbrear/algo-trading-engine.git@main",
    # Or for a specific version/tag:
    # "algo-trading-engine @ git+https://github.com/danerbrear/algo-trading-engine.git@v0.0.3",
]
```

Or install directly:
```bash
pip install git+https://github.com/danerbrear/algo-trading-engine.git@main
```

### Basic Backtesting Example

```python
from datetime import datetime
from algo_trading_engine import BacktestEngine, BacktestConfig

# Create configuration
config = BacktestConfig(
    initial_capital=3000,
    start_date=datetime(2025, 1, 2),
    end_date=datetime(2026, 1, 2),
    symbol="SPY",
    strategy_type="velocity_momentum",  # Built-in strategy
    api_key="your-polygon-api-key"
)

# Run backtest
engine = BacktestEngine.from_config(config)
engine.run()

# Get results
metrics = engine.get_performance_metrics()
print(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Win Rate: {metrics.win_rate:.1f}%")
```

### Basic Paper Trading Example

```python
from algo_trading_engine import PaperTradingEngine, PaperTradingConfig

# Create configuration
# Note: Capital is managed via config/strategies/capital_allocations.json
config = PaperTradingConfig(
    symbol="SPY",
    strategy_type="velocity_momentum",
    api_key="your-polygon-api-key"
)

# Run paper trading
engine = PaperTradingEngine.from_config(config)
engine.run()
```

See [examples/](examples/) for complete working examples including custom strategies.

### Bar Interval Support

The system supports multiple bar intervals for backtesting and paper trading:

```python
from algo_trading_engine import BacktestConfig, BarTimeInterval

# Daily bars (default) - best for swing/position trading
config = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2026, 1, 1),
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.DAY  # Default
)

# Hourly bars - best for intraday trading
# Note: yfinance limits hourly data to 729 days (~2 years)
config = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),  # Within 729 day limit
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.HOUR
)

# Minute bars - best for day trading/scalping
# Note: yfinance limits minute data to 59 days
config = BacktestConfig(
    initial_capital=10000,
    start_date=datetime(2024, 11, 1),
    end_date=datetime(2024, 12, 20),  # Within 59 day limit
    symbol="SPY",
    strategy_type="velocity_momentum",
    bar_interval=BarTimeInterval.MINUTE
)
```

**Key Points:**
- **Daily bars**: No date range limit, best for position/swing trading
- **Hourly bars**: Max 729 days (yfinance limit), ~6.5x more data points than daily
- **Minute bars**: Max 59 days (yfinance limit), ~390x more data points than daily
- **Cache structure**: Interval-specific subdirectories (`daily/`, `hourly/`, `minute/`)
- **Performance**: Daily bars are fastest; minute bars require significantly more processing

See [examples/backtest/](examples/backtest/) for complete examples of each interval type.

## 📁 Project Structure

### Core Components

- **`algo_trading_engine/ml_models/`** - Machine learning models and training logic
  - Market analysis machine learning models
  - Strategy prediction machine learning models
  - See [Model Documentation](algo_trading_engine/ml_models/README.md) for details

- **`algo_trading_engine/strategies/`** - Trading strategy implementations
  - Various options trading strategies
  - See [Strategy Documentation](algo_trading_engine/strategies/README.md) for details

- **`algo_trading_engine/backtest/`** - Backtesting framework
  - Volume validation system
  - Performance analysis tools
  - Strategy evaluation

- **`algo_trading_engine/common/`** - Shared utilities and data models
  - Data retrieval and caching
  - Value Objects and DTOs
  - Common functions

- **`algo_trading_engine/prediction/`** - Prediction pipeline
  - Interactive recommendations and decision capture
  - JSON decision store
  - CLI for open/close flows
  - Equity curve visualization
  - See [Prediction Documentation](algo_trading_engine/prediction/README.md) for details

### File Structure

```
/
├── src/algo_trading_engine/      # Source code directory (Python package)
│   ├── ml_models/            # ML models and training
│   │   ├── README.md         # Model-specific documentation
│   │   ├── main.py           # Training entry point
│   │   ├── lstm_model.py     # Strategy prediction model
│   │   ├── market_state_classifier.py # Market analysis model
│   │   └── ...
│   ├── strategies/           # Trading strategies
│   │   ├── velocity_signal_momentum_strategy.py
│   │   ├── credit_spread_minimal.py
│   │   └── README.md         # Strategy documentation
│   ├── backtest/             # Backtesting framework
│   │   ├── main.py           # Backtesting engine
│   │   ├── volume_validator.py
│   │   └── ...
│   ├── common/               # Shared utilities
│   │   ├── data_retriever.py # Data fetching
│   │   ├── models.py         # Value Objects
│   │   ├── cache/            # Caching system
│   │   └── ...
│   └── prediction/           # Prediction pipeline
│       ├── recommend_cli.py          # CLI entrypoint for recommendations
│       ├── recommendation_engine.py  # InteractiveStrategyRecommender
│       ├── decision_store.py         # JSON decision store
│       └── ...
├── data_cache/                          # Cached market data
│   ├── stocks/                          # Stock price data
│   ├── options/                         # Options chain data
│   ├── treasury/                        # Treasury yield data
│   └── calendar/                        # Economic calendar data
├── config/                              # Configuration files
│   ├── strategies/                      # Config specific to custom strategies
│   |   ├── capital_allocations.json     # Specify amount of capital and max position risk for each custom strategy
│   │   └── ...
├── tests/                               # Unit tests
├── docs/                                # Documentation
├── predictions/                         # Prediction outputs
├── Trained_Models/                      # Saved models
├── pyproject.toml                       # Package configuration and dependencies
├── setup_env.py                         # Environment setup
└── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher (required for yfinance 1.0+)
- Virtual environment (recommended)
- Polygon.io API key

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd algo-trading-engine
python setup_env.py
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Configure environment:**

      Edit .env file with your Polygon.io API key


3. **Train models:**
```bash
python -m src.ml_models.main --free --save
```
5. **Run backtests:**
```bash
python -m src.backtest.main --strategy velocity_momentum
```

6. **Analyze performance:**
```bash
# View equity curve and statistics
python -m src.prediction.plot_equity_curve --summary-only
```

7. **Run Trading Engine**
```bash
python -m src.prediction.recommend_cli --strategy velocity_momentum
```

## 📊 Key Features

### Market State Classification
- **HMM-based regime detection** with 3-5 distinct market states
- **Real-time state identification** for current market conditions
- **State transition analysis** for trend prediction

### Options Strategy Prediction
- **ML-based strategy selection** from multiple classes:
  - Hold (no position)
  - Call Credit Spread (bearish/neutral)
  - Put Credit Spread (bullish/neutral)
- **Risk-adjusted returns** using Sharpe ratio calculations
- **Treasury rate integration** for realistic risk-free rates

### Backtesting System
- **Adjusted Historical Returns** to reduce survivorship bias
- **Volume validation** for realistic trading simulation
- **Comprehensive performance metrics** including Sharpe ratios
- **Strategy comparison** and optimization tools

### Data Management
- **Intelligent caching** for API efficiency
- **Value Objects** for type-safe data handling
- **Treasury rate integration** for accurate risk calculations

## 🔧 Configuration

### Environment Variables
- `POLYGON_API_KEY`: Polygon.io API key for options data
- `MODEL_SAVE_BASE_PATH`: Path for saving trained models

### Capital Allocation and Risk Management
Each strategy has independent capital allocation with risk-based position sizing. Configure in `config/strategies/capital_allocations.json`:

```json
{
  "strategies": {
    "credit_spread": {
      "allocated_capital": 10000.0,
      "max_risk_percentage": 0.05
    },
    "velocity_momentum": {
      "allocated_capital": 2000.0,
      "max_risk_percentage": 0.20
    }
  }
}
```

- **`allocated_capital`**: Starting capital for the strategy
- **`max_risk_percentage`**: Maximum risk per position as % of allocated capital (e.g., 0.05 = 5%)

Positions are only opened if the maximum risk is within the defined percentage. See [Capital Allocation Feature Documentation](features/capital_allocation_and_risk_management.md) for details.

### Volume Validation
- Configurable minimum volume thresholds
- Real-time volume checking for position entry/exit
- Comprehensive volume statistics tracking

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_treasury_rates.py -v
python -m pytest tests/test_velocity_strategy.py -v
```

## 📚 Documentation

### API Documentation
- **[Public API Organization](docs/public_api_organization.md)** - Guide to the public API structure and usage patterns

### Component Documentation
- **[Model Documentation](algo_trading_engine/ml_models/README.md)** - ML models, training, and evaluation
- **[Strategy Documentation](algo_trading_engine/strategies/README.md)** - Trading strategy implementations
- **[Prediction Documentation](algo_trading_engine/prediction/README.md)** - Recommendation CLI and equity curve analysis
- **[Backtest Documentation](algo_trading_engine/backtest/README.md)** - Backtesting framework and usage

### Feature Guides
- **[Capital Allocation & Risk Management](features/capital_allocation_and_risk_management.md)** - Per-strategy capital allocation and risk-based position sizing
- **[Volume Validation](docs/volume_validation_guide.md)** - Real-time volume checking for position entry/exit
- **[MA Velocity Analysis](docs/ma_velocity_analysis_guide.md)** - Moving average velocity/elasticity analysis
- **[Strategy Builder](docs/strategy_builder_guide.md)** - Custom strategy development guide

## ⚠️ Limitations

1. **Market Regime Stability**: States evolve over time
2. **API Dependencies**: Requires Polygon.io access with rate limits
4. **Trading Risk**: Educational/research purposes only

## 📄 License

This project is for educational and research purposes. Always do your own research and consider your risk tolerance before trading.

---

**Note**: This system is for educational and research purposes. Options trading involves substantial risk of loss. Past performance does not guarantee future results.
