# LSTM POC - Options Trading Strategy System

This project provides a comprehensive backtesting and prediction platform for options trading strategies. The system enables users to develop, test, and evaluate trading strategies using historical data, while also providing real-time predictions for current market conditions. It includes a robust backtesting framework with volume validation, risk management tools, and machine learning-powered strategy prediction capabilities.

## 🏗️ Project Architecture

The system operates in two main stages:
1. **Market Analysis** - Identifies market conditions and trends
2. **Strategy Prediction** - Determines optimal trading strategies

### Core Components

- **`src/model/`** - Machine learning models and training logic
  - Market analysis models
  - Strategy prediction models
  - See [Model Documentation](src/model/README.md) for details

- **`src/strategies/`** - Trading strategy implementations
  - Various options trading strategies
  - Risk management and position sizing
  - See [Strategy Documentation](src/strategies/README.md) for details

- **`src/backtest/`** - Backtesting framework
  - Volume validation system
  - Performance analysis tools
  - Strategy evaluation

- **`src/common/`** - Shared utilities and data models
  - Data retrieval and caching
  - Value Objects and DTOs
  - Common functions

- **`src/prediction/`** - Prediction pipeline
  - Interactive recommendations and decision capture
  - JSON decision store
  - CLI for open/close flows

- **`src/analysis/`** - Market analysis tools
  - Moving average velocity analysis
  - Upward trend drawdown analysis
  - Daily drawdown likelihood analysis
  - Independent statistical analyses

## 📁 Project Structure

```
lstm_poc/
├── src/                      # Source code directory
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
│   ├── prediction/           # Prediction pipeline
│   │   ├── recommend_cli.py          # CLI entrypoint for recommendations
│   │   ├── recommendation_engine.py  # InteractiveStrategyRecommender
│   │   ├── decision_store.py         # JSON decision store
│   │   └── ...
│   └── analysis/             # Market analysis tools
│       ├── ma_velocity_analysis.py   # Moving average analysis
│       ├── upward_trend_drawdown_analysis.py  # Drawdown analysis
│       ├── daily_drawdown_likelihood_analysis.py  # Daily likelihood analysis
│       ├── run_ma_analysis.py        # MA analysis entry point
│       ├── run_drawdown_analysis.py  # Drawdown analysis entry point
│       └── run_daily_likelihood_analysis.py  # Daily likelihood entry point
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

- Python 3.9 or higher
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

6. **Run market analysis:**
```bash
# Moving average velocity analysis
python -m src.analysis.run_ma_analysis

# Upward trend drawdown analysis (default: 12 months)
python -m src.analysis.run_drawdown_analysis

# Daily drawdown likelihood analysis
python -m src.analysis.run_daily_likelihood_analysis

# Custom analysis period
python -m src.analysis.run_drawdown_analysis --months 6

# Skip plotting (console output only)
python -m src.analysis.run_drawdown_analysis --no-plot
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

## 📈 Performance

The system achieves:
- **Market Analysis**: Clear identification of market conditions
- **Strategy Prediction Accuracy**: ~70-75% on test data
- **Risk-Adjusted Returns**: Sharpe ratio optimization
- **Volume Validation**: Realistic trading simulation

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

- **[Model Documentation](src/model/README.md)** - ML models, training, and evaluation
- **[Strategy Documentation](src/strategies/README.md)** - Trading strategy implementations
- **[Backtest Documentation](src/backtest/README.md)** - Backtesting framework and usage
- **[Analysis Documentation](docs/upward_trend_drawdown_analysis_implementation_summary.md)** - Market analysis tools
- **[API Documentation](docs/)** - Detailed API and usage guides

### Analysis Tools

- **Moving Average Velocity Analysis** - Identifies optimal MA combinations for trend signals
  - Specification: `features/ma_analysis.md`
  - Implementation: `src/analysis/ma_velocity_analysis.py`
  
- **Upward Trend Drawdown Analysis** - Analyzes drawdowns during 3-10 day upward trends
  - Specification: `features/upward_trend_drawdown_analysis.md`
  - Implementation: `src/analysis/upward_trend_drawdown_analysis.py`
  - Summary: `docs/upward_trend_drawdown_analysis_implementation_summary.md`
  
- **Daily Drawdown Likelihood Analysis** - Analyzes likelihood of drawdowns on each day of upward trends
  - Specification: `features/daily_drawdown_likelihood_analysis.md`
  - Implementation: `src/analysis/daily_drawdown_likelihood_analysis.py`
  - Summary: `docs/daily_drawdown_likelihood_analysis_summary.md`

## ⚠️ Limitations

1. **Market Regime Stability**: States evolve over time
2. **API Dependencies**: Requires Polygon.io access with rate limits
3. **Computational Requirements**: GPU recommended for training
4. **Trading Risk**: Educational/research purposes only

## 🤝 Contributing

1. Follow the established code patterns and Value Object rules
2. Create unit tests for new functionality
3. Update relevant documentation
4. Use the virtual environment for development

## 📄 License

This project is for educational and research purposes. Always do your own research and consider your risk tolerance before trading.

---

**Note**: This system is for educational and research purposes. Options trading involves substantial risk of loss. Past performance does not guarantee future results.