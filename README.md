# Market Regime Classification System

This project implements a sophisticated market regime classification system using a combination of Hidden Markov Models (HMM) and Long Short-Term Memory (LSTM) networks. The system identifies distinct market states and predicts optimal options trading strategies using both price action and options market data.

## Overview

The system operates in two main stages:
1. Market State Identification using HMM
2. Options Strategy Prediction using LSTM

### Market States

The HMM identifies distinct market regimes based on core market features. The system can identify multiple market states (typically 3-5 states) with characteristics such as:

- **Low Volatility Uptrend**: Low volatility with steady price increases
- **Momentum Uptrend**: Medium volatility with strong upward momentum  
- **Consolidation**: Low volatility with sideways price movement
- **High Volatility Downtrend**: High volatility with declining prices
- **High Volatility Rally**: Very high volatility with sharp price increases

Each state has distinct characteristics in terms of:
- Average returns
- Volatility levels
- Price relative to moving averages
- Volume patterns

### Options Trading Strategies

The LSTM model predicts optimal options trading strategies from three classes:

1. **Hold** - No options position recommended
   - Market conditions suggest holding cash
   - Wait for better opportunities or reduce position sizes

2. **Call Credit Spread** - Bearish/Neutral strategy
   - Structure: Sell ATM Call + Buy ATM+5 Call
   - Direction: Profit when stock goes down or stays flat
   - Max Profit: Premium received
   - Max Loss: $5 - Premium received

3. **Put Credit Spread** - Bullish/Neutral strategy
   - Structure: Sell ATM Put + Buy ATM-5 Put
   - Direction: Profit when stock goes up or stays flat
   - Max Profit: Premium received
   - Max Loss: $5 - Premium received

## Project Files

### Core Components

- **`src/model/main.py`** - Main training and evaluation script
- **`src/model/data_retriever.py`** - Market data fetching and feature engineering
- **`src/model/options_handler.py`** - Options data processing and strategy calculations
- **`src/model/market_state_classifier.py`** - HMM model for market state classification
- **`src/model/lstm_model.py`** - LSTM model for strategy prediction
- **`src/model/config.py`** - Configuration settings
- **`src/common/cache/cache_manager.py`** - Data caching utilities
- **`src/model/api_retry_handler.py`** - API request retry logic
- **`progress_tracker.py`** - Training progress tracking

### New Files

- **`src/prediction/predict_today.py`** - Script for making predictions on today's market data
  - Loads pretrained HMM and LSTM models
  - Fetches recent market data (prioritizes cached data)
  - Calculates features and makes predictions
  - Provides specific option recommendations
  - Saves results to `predictions/` directory

- **`src/common/cache/examine_cache.py`** - Utility script for examining cached data files
  - Inspects pickle files in the cache directory
  - Shows data structure, date ranges, and statistics
  - Useful for debugging and data validation

- **`setup_env.py`** - Environment setup script
  - Creates `.env` file with required API keys
  - Sets up configuration directories

### Data and Output Directories

- **`data_cache/`** - Cached market and options data
  - `data_cache/stocks/` - Stock price data
  - `data_cache/options/` - Options chain data
  - `data_cache/treasury/` - Treasury yield data

- **`predictions/`** - Prediction output files
  - Daily prediction results saved as timestamped text files
  - Format: `prediction_SPY_YYYYMMDD.txt`

- **`Trained_Models/`** - Saved model files (when using --save flag)
  - Timestamped and latest model versions
  - Includes both LSTM and HMM models

## Technical Architecture

### Data Retrieval and Processing (`data_retriever.py`)

The DataRetriever class handles:
- Market data fetching using yfinance
- Feature engineering and calculation
- Options data processing
- State classification using HMM

Key Features Used for State Classification:
- Returns and volatility metrics
- Price relative to moving averages
- Volume analysis
- Trend indicators

Additional Features for LSTM Training:
- Options market indicators
- Momentum indicators (RSI, MACD)
- Advanced volume metrics (OBV)

### LSTM Model Architecture (`lstm_model.py`)

The model uses a sophisticated architecture:
- Bidirectional LSTM layers for sequence learning
- Layer normalization for training stability
- Dropout and L2 regularization for preventing overfitting
- Class weights to handle strategy imbalance

Model Structure:
```
1. Input Layer
2. Bidirectional LSTM (128 units) + LayerNorm + Dropout(0.4)
3. Bidirectional LSTM (64 units) + LayerNorm + Dropout(0.4)
4. LSTM (32 units) + LayerNorm + Dropout(0.3)
5. Dense (64 units) + LayerNorm + Dropout(0.3)
6. Dense (32 units) + LayerNorm + Dropout(0.2)
7. Output Layer (3 units, softmax) - Hold, Call Credit Spread, Put Credit Spread
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)
- Polygon.io API key (get one at https://polygon.io/dashboard/signup)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
python setup_env.py
```

4. Edit the created `.env` file and replace `your_api_key_here` with your actual Polygon.io API key.

### Configuration

The system uses the following environment variables:
- `POLYGON_API_KEY`: Your Polygon.io API key for fetching historical options data

You can set these variables in two ways:
1. In the `.env` file (recommended)
2. Directly in the code when initializing OptionsHandler:
```python
handler = OptionsHandler(symbol='SPY', api_key='YOUR_KEY')
```

### Running the System

#### Training and Evaluation

1. Run the main training program:
```bash
# From the project root directory
python -m src.model.main
```

2. **Free Tier Rate Limiting:**

If you're using a free Polygon.io API account, use the `-f` or `--free` flag to enable 13-second delays between API requests to stay within rate limits:
```bash
python -m src.model.main --free
```
This prevents hitting the free tier limit of 5 API calls per minute. Without this flag, the system will make requests without delays (suitable for paid API tiers).

3. **Output Verbosity Control:**

By default, the system runs in quiet mode with a clean progress bar. Use the `-v` or `--verbose` flag to show detailed logging:
```bash
# Default: clean progress bar (quiet mode)
python -m src.model.main

# Detailed logging for debugging
python -m src.model.main --verbose

# Explicitly enable quiet mode (same as default)
python -m src.model.main --quiet
```
In quiet mode (default), detailed API call messages are suppressed and only a clean progress bar with essential information is displayed. Use `--verbose` for full logging when debugging.

4. **Saving the Trained Model:**

You can save the trained model after training by using the `-s` or `--save` flag:
```bash
python -m src.model.main --save
```
- The model will be saved in Keras format (`.keras`) to the directory specified by the `MODEL_SAVE_BASE_PATH` variable in your `.env` file (default: `Trained_Models`).
- The folder structure will be:
  - `<MODEL_SAVE_BASE_PATH>/<mode>/<timestamp>/model.keras` (timestamped)
  - `<MODEL_SAVE_BASE_PATH>/<mode>/latest/model.keras` (latest, always overwritten)
- You can specify a custom mode label for the subfolder using `--mode`:
```bash
python -m src.model.main --save --mode my_experiment
```

- You can combine flags. For example, to use free tier rate limiting, quiet mode, and save the model:
```bash
python -m src.model.main --free --quiet --save
```

5. **Common Usage Examples:**

```bash
# Basic usage (clean progress bar by default)
python -m src.model.main

# Detailed logging for debugging
python -m src.model.main --verbose

# Free tier with clean progress (default)
python -m src.model.main --free

# Production run with all options
python -m src.model.main --free --save --mode production

# Debug mode with full logging
python -m src.model.main --verbose --free --save
```

#### Making Predictions

After training and saving models, you can make predictions for today's market. The script can be run in multiple ways:

**Option 1: Run as a module (recommended)**
```bash
# From the project root directory
python -m src.prediction.predict_today

# With custom symbol
python -m src.prediction.predict_today --symbol QQQ
```

**Option 2: Run directly**
```bash
# From the project root directory
python src/prediction/predict_today.py

# With custom symbol
python src/prediction/predict_today.py --symbol QQQ
```

**Option 3: Run from the prediction directory**
```bash
cd src/prediction
python predict_today.py
```

The prediction script will:
- Load pretrained HMM and LSTM models
- Fetch recent market data (prioritizes cached data)
- Calculate technical features
- Predict market state and optimal strategy
- Provide specific option recommendations
- Save results to `predictions/` directory

**Note:** The script automatically handles import paths for both module execution (`-m` flag) and direct execution.

#### Examining Cached Data

To inspect cached data files:

```bash
# From the project root directory
python -m src.common.cache.examine_cache

# Or run directly
python src/common/cache/examine_cache.py

# Modify the script to examine different files
# Edit src/common/cache/examine_cache.py and change the file path
```

### Development and Testing

For development or testing specific components:
```python
from options_handler import OptionsHandler
import pandas as pd

# Initialize the handler
handler = OptionsHandler(symbol='SPY')

# Load your price data
data = pd.read_csv('spy_data.csv', index_col='Date', parse_dates=True)

# Calculate options features
data = handler.calculate_option_features(data)
```

## Model Performance

The system achieves:
- HMM State Classification: Clear separation of market regimes with distinct characteristics
- LSTM Prediction Performance:
  - Training Accuracy: ~90%
  - Testing/Validation Accuracy: ~70-75%
  - Strong performance in identifying specific strategies:
    - Hold strategy: High precision for conservative periods
    - Call Credit Spread: Good performance during bearish/neutral markets
    - Put Credit Spread: Strong performance during bullish/neutral markets

Key Performance Characteristics:
- Handles class imbalance through weighted training
- Prevents overfitting using multiple regularization techniques
- Captures both short and long-term market dynamics
- Provides actionable options trading recommendations

## Data Requirements

The system requires:
- Daily OHLCV data
- Options chain data from Polygon.io
  - Minimum volume: 10 contracts
  - Minimum price: $0.10
  - Greeks and implied volatility data
- Minimum 2 years of historical data recommended

### Options Data Features

The system calculates the following options-related features:
- Implied Volatility (IV) for calls and puts
- Option Greeks (Delta, Gamma, Theta, Vega)
- Volume and Open Interest
- Put/Call Ratio
- Option Volume Ratio

### Data Caching

To optimize API usage and performance:
- Options data is automatically cached
- Cache location: `data_cache/options/<symbol>/`
- Cache files are date-based and stored in pickle format
- Cache is automatically used when available
- Stock data is also cached in `data_cache/stocks/<symbol>/`

## Limitations and Considerations

1. Market Regime Stability
   - States are not permanent and can evolve over time
   - Major market events can create new regimes

2. Options Data Dependency
   - Requires Polygon.io API access
   - API rate limits apply: use `--free` flag for free tier (5 calls/minute) or upgrade to paid tier for faster processing
   - Some historical dates may have incomplete data

3. Computational Requirements
   - HMM state optimization is computationally intensive
   - LSTM training benefits from GPU acceleration
   - Options data caching helps reduce API calls

4. Trading Risk
   - This system is for educational and research purposes
   - Always do your own research and consider your risk tolerance
   - Options trading involves substantial risk of loss
   - Past performance does not guarantee future results

## File Structure

```
lstm_poc/
├── src/                      # Source code directory
│   ├── __init__.py           # Makes src a Python package
│   ├── model/                # Model-related modules
│   │   ├── __init__.py       # Makes model a package
│   │   ├── main.py           # Main training script
│   │   ├── data_retriever.py # Data fetching and processing
│   │   ├── options_handler.py # Options data handling
│   │   ├── market_state_classifier.py # HMM model
│   │   ├── lstm_model.py     # LSTM model
│   │   ├── config.py         # Configuration
│   │   └── api_retry_handler.py # API retry logic
│   ├── prediction/           # Prediction modules
│   │   ├── __init__.py       # Makes prediction a package
│   │   └── predict_today.py  # Daily prediction script
│   └── common/               # Common utilities
│       ├── __init__.py       # Makes common a package
│       └── cache/            # Caching utilities
│           ├── __init__.py   # Makes cache a package
│           ├── cache_manager.py # Caching utilities
│           └── examine_cache.py # Cache inspection utility
├── progress_tracker.py       # Progress tracking
├── setup_env.py              # Environment setup
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .env                      # Environment variables (created by setup)
├── data_cache/               # Cached data
│   ├── stocks/               # Stock price data
│   ├── options/              # Options data
│   └── treasury/             # Treasury data
├── predictions/              # Prediction outputs
└── Trained_Models/           # Saved models (when using --save)
```