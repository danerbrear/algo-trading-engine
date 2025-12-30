# Options Trading System

This project provides a comprehensive backtesting and prediction platform for options trading strategies. The system enables users to develop, test, and evaluate trading strategies using historical data, while also providing real-time predictions for current market conditions. It includes a robust backtesting framework with volume validation, risk management tools, and machine learning-powered strategy prediction capabilities.

## 🏗️ Project Architecture

The system operates in two main stages:
1. **Market Analysis** - Identifies market conditions and trends
2. **Strategy Prediction** - Determines optimal trading strategies

### Core Components

- **`algo_trading_engine/model/`** - Machine learning models and training logic
  - Market analysis models
  - Strategy prediction models
  - See [Model Documentation](algo_trading_engine/model/README.md) for details

- **`algo_trading_engine/strategies/`** - Trading strategy implementations
  - Various options trading strategies
  - Risk management and position sizing
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

## 📁 Project Structure

```
lstm_poc/
├── algo_trading_engine/      # Source code directory (Python package)
│   ├── model/                # ML models and training
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
├── data_cache/               # Cached market data
│   ├── stocks/               # Stock price data
│   ├── options/              # Options chain data
│   ├── treasury/             # Treasury yield data
│   └── calendar/             # Economic calendar data
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── predictions/              # Prediction outputs
├── Trained_Models/           # Saved models
├── requirements.txt          # Dependencies
├── setup_env.py              # Environment setup
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python version >3.9 and <3.12
- Virtual environment (recommended)
- Polygon.io API key

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd lstm_poc
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
python setup_env.py
# Edit .env file with your Polygon.io API key
```

3. **Train models:**
```bash
python -m src.model.main --free --save
```

4. **Run the prediction engine (interactive recommender):**
```bash
# Basic run (interactive prompts)
python -m src.prediction.recommend_cli --symbol SPY --strategy credit_spread

# Specify a historical or specific date (YYYY-MM-DD)
python -m src.prediction.recommend_cli --symbol SPY --strategy credit_spread --date 2025-09-10

# Non-interactive (auto-accept recommendations; useful for batch runs)
python -m src.prediction.recommend_cli --symbol SPY --strategy credit_spread --yes
```

Flags:
- `--symbol`: underlying symbol (default `SPY`)
- `--strategy`: `credit_spread` (default) or `velocity_momentum`
- `--date`: run date in `YYYY-MM-DD`; defaults to today
- `--yes`: auto-accept prompts (non-interactive)

Outputs:
- Accepted decisions are written to `predictions/decisions/decisions_YYYYMMDD.json`.
- When open positions exist, the CLI prints their current status and will only run the close flow for that day.

5. **Run backtests:**
```bash
python -m src.backtest.main --strategy velocity_signal_momentum
```

6. **Analyze performance:**
```bash
# View equity curve and statistics
python -m src.prediction.plot_equity_curve --summary-only
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

- **[Model Documentation](algo_trading_engine/model/README.md)** - ML models, training, and evaluation
- **[Strategy Documentation](algo_trading_engine/strategies/README.md)** - Trading strategy implementations
- **[Prediction Documentation](algo_trading_engine/prediction/README.md)** - Recommendation CLI and equity curve analysis
- **[Backtest Documentation](algo_trading_engine/backtest/README.md)** - Backtesting framework and usage
- **[API Documentation](docs/)** - Detailed API and usage guides

## ⚠️ Limitations

1. **Market Regime Stability**: States evolve over time
2. **API Dependencies**: Requires Polygon.io access with rate limits
4. **Trading Risk**: Educational/research purposes only

## 📄 License

This project is for educational and research purposes. Always do your own research and consider your risk tolerance before trading.

---

**Note**: This system is for educational and research purposes. Options trading involves substantial risk of loss. Past performance does not guarantee future results.
