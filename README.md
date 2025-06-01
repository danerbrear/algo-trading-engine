# Market Regime Classification System

This project implements a sophisticated market regime classification system using a combination of Hidden Markov Models (HMM) and Long Short-Term Memory (LSTM) networks. The system identifies distinct market states and predicts regime transitions using both price action and options market data.

## Overview

The system operates in two main stages:
1. Market State Identification using HMM
2. State Prediction using LSTM

### Market States

The HMM identifies 5 distinct market regimes based on core market features:

1. Low Volatility Uptrend (22.30%)
   - Average Return: +0.0905%
   - Low Volatility: 0.0044
   - Price above MA (1.0103)
   - Normal Volume (1.01)

2. Momentum Uptrend (21.87%)
   - Average Return: +0.0941%
   - Medium Volatility: 0.0073
   - Strong Price above MA (1.0119)
   - Slightly Below Average Volume (0.98)

3. Consolidation (20.78%)
   - Average Return: +0.0915%
   - Medium Volatility: 0.0076
   - Price near MA (1.0050)
   - Low Volume (0.94)

4. High Volatility Downtrend (29.94%)
   - Average Return: -0.0598%
   - High Volatility: 0.0133
   - Price below MA (0.9949)
   - High Volume (1.09)

5. High Volatility Rally (5.11%)
   - Average Return: +0.2340%
   - Very High Volatility: 0.0274
   - Price near MA (1.0037)
   - Low Volume (0.93)

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
- Class weights to handle regime imbalance

Model Structure:
```
1. Input Layer
2. Bidirectional LSTM (128 units) + LayerNorm + Dropout(0.4)
3. Bidirectional LSTM (64 units) + LayerNorm + Dropout(0.4)
4. LSTM (32 units) + LayerNorm + Dropout(0.3)
5. Dense (64 units) + LayerNorm + Dropout(0.3)
6. Dense (32 units) + LayerNorm + Dropout(0.2)
7. Output Layer (5 units, softmax)
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

1. Run the main program:
```bash
python main.py
```

2. For development or testing specific components:
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
  - Training Accuracy: 92%
  - Testing/Validation Accuracy: 73%
  - Strong performance in identifying specific regimes:
    - Bullish Low Vol States: 82% precision, 83% recall (State 1)
    - Consolidation States: 90% precision, 67% recall (State 2)
    - Volatile Downtrends: 62% precision, 69% recall (State 3)
    - High Vol Rally: 62% precision, 100% recall (State 4)

Key Performance Characteristics:
- Handles class imbalance through weighted training
- Prevents overfitting using multiple regularization techniques
- Captures both short and long-term market dynamics
- Macro-average F1 score: 0.67
- Weighted-average F1 score: 0.74

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

## Limitations and Considerations

1. Market Regime Stability
   - States are not permanent and can evolve over time
   - Major market events can create new regimes

2. Options Data Dependency
   - Requires Polygon.io API access
   - API rate limits apply (consider paid tier for production)
   - Some historical dates may have incomplete data

3. Computational Requirements
   - HMM state optimization is computationally intensive
   - LSTM training benefits from GPU acceleration
   - Options data caching helps reduce API calls