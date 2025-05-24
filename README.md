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

## Installation and Usage

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the program:
```bash
python main.py
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
- Options chain data
- Minimum 2 years of historical data recommended

## Limitations and Considerations

1. Market Regime Stability
   - States are not permanent and can evolve over time
   - Major market events can create new regimes

2. Options Data Dependency
   - Requires reliable options market data
   - Some instruments may have limited options data

3. Computational Requirements
   - HMM state optimization is computationally intensive
   - LSTM training benefits from GPU acceleration

## Future Improvements

1. Model Enhancements
   - Dynamic state number optimization
   - Attention mechanisms for LSTM
   - Ensemble methods for prediction

2. Feature Engineering
   - Additional options Greeks
   - Market sentiment indicators
   - Cross-asset correlation features

3. Production Deployment
   - Real-time prediction pipeline
   - Model retraining framework
   - Performance monitoring system 