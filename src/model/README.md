# Machine Learning Models

This directory contains the machine learning models and training logic for the LSTM POC system.

## Overview

The system uses two main machine learning models:
1. **Hidden Markov Model (HMM)** for market state classification
2. **Long Short-Term Memory (LSTM)** for options strategy prediction

## Model Architecture

### 1. Market State Classifier (HMM)

**File**: `market_state_classifier.py`

The HMM identifies distinct market regimes based on core market features.

#### Market States
- **Low Volatility Uptrend**: Low volatility with steady price increases
- **Momentum Uptrend**: Medium volatility with strong upward momentum  
- **Consolidation**: Low volatility with sideways price movement
- **High Volatility Downtrend**: High volatility with declining prices
- **High Volatility Rally**: Very high volatility with sharp price increases

#### Features Used
- Returns and volatility metrics
- Price relative to moving averages
- Volume analysis
- Trend indicators

#### Training Process
```python
from src.model.market_state_classifier import MarketStateClassifier

# Initialize classifier
classifier = MarketStateClassifier()

# Train on historical data
classifier.train(data)

# Predict market states
states = classifier.predict_states(data)
```

### 2. Strategy Prediction Model (LSTM)

**File**: `lstm_model.py`

The LSTM predicts optimal options trading strategies from three classes.

#### Strategy Classes
1. **Hold** - No options position recommended
2. **Call Credit Spread** - Bearish/Neutral strategy
3. **Put Credit Spread** - Bullish/Neutral strategy

#### Model Architecture
```
1. Input Layer
2. Bidirectional LSTM (128 units) + LayerNorm + Dropout(0.4)
3. Bidirectional LSTM (64 units) + LayerNorm + Dropout(0.4)
4. LSTM (32 units) + LayerNorm + Dropout(0.3)
5. Dense (64 units) + LayerNorm + Dropout(0.3)
6. Dense (32 units) + LayerNorm + Dropout(0.2)
7. Output Layer (3 units, softmax)
```

#### Features Used
- Market state predictions from HMM
- Options market indicators
- Momentum indicators (RSI, MACD)
- Advanced volume metrics (OBV)
- Economic calendar features

#### Training Process
```python
from src.model.lstm_model import LSTMModel

# Initialize model
model = LSTMModel(sequence_length=60, num_features=50)

# Train model
history = model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)
```

## Training Pipeline

### Main Training Script

**File**: `main.py`

The main training script orchestrates the entire training process:

```bash
# Basic training
python -m src.model.main

# Training with options
python -m src.model.main --symbol QQQ --free --save --verbose
```

#### Command Line Options
- `--symbol SYMBOL`: Stock symbol to train on (default: SPY)
- `-s, --save`: Save trained models
- `--mode MODE`: Mode label for model saving
- `-f, --free`: Use free tier rate limiting
- `-q, --quiet`: Suppress detailed output
- `-v, --verbose`: Show detailed output

### Data Preparation

**File**: `data_retriever.py`

Handles data fetching, feature engineering, and preparation:

```python
from src.common.data_retriever import DataRetriever

# Initialize retriever
retriever = DataRetriever(
    symbol='SPY',
    hmm_start_date='2010-01-01',
    lstm_start_date='2020-01-01'
)

# Prepare data for training
data, options_data = retriever.prepare_data_for_lstm(
    sequence_length=60,
    state_classifier=hmm_model
)
```

### Options Data Processing

**File**: `options_handler.py`

Processes options data and calculates strategy-specific features:

```python
from src.model.options_handler import OptionsHandler
from src.model.lstm_model import LSTMModel

# Initialize handler
handler = OptionsHandler(symbol='SPY')

# Calculate options features
data = handler.calculate_option_features(data)

# Calculate strategy signals (moved to LSTMModel)
data = LSTMModel.calculate_option_signals(data)
```

## Model Performance

### HMM Performance
- **State Classification**: Clear separation of market regimes
- **State Stability**: Consistent state identification across time periods
- **Feature Importance**: Volatility and trend indicators most predictive

### LSTM Performance
- **Training Accuracy**: ~90%
- **Testing Accuracy**: ~70-75%
- **Strategy Performance**:
  - Hold strategy: High precision for conservative periods
  - Call Credit Spread: Good performance during bearish/neutral markets
  - Put Credit Spread: Strong performance during bullish/neutral markets

### Key Performance Characteristics
- Handles class imbalance through weighted training
- Prevents overfitting using multiple regularization techniques
- Captures both short and long-term market dynamics
- Provides actionable options trading recommendations

## Model Saving and Loading

### Saving Models
```bash
# Save models during training
python -m src.model.main --save --mode production
```

### Loading Models
```python
from src.common.functions import load_hmm_model, load_lstm_model

# Load HMM model
hmm_model = load_hmm_model(model_dir)

# Load LSTM model
lstm_model, scaler = load_lstm_model(model_dir, return_lstm_instance=True)
```

### Model Directory Structure
```
Trained_Models/
├── lstm_poc/
│   └── SPY/
│       ├── latest/
│       │   ├── model.keras
│       │   ├── lstm_scaler.pkl
│       │   └── hmm_model.pkl
│       └── 2024-01-15_14-30-00/
│           ├── model.keras
│           ├── lstm_scaler.pkl
│           └── hmm_model.pkl
```

## Visualization

**File**: `plots.py`

Comprehensive visualization capabilities:

- **Confusion Matrix**: Model prediction accuracy across strategy classes
- **Signal Distribution**: Actual vs predicted signals over time
- **Returns Comparison**: Strategy returns vs market returns
- **Training History**: Loss and accuracy curves
- **Feature Importance**: Feature contributions
- **Market States**: State transitions over time

## Configuration

**File**: `config.py`

Model configuration settings:

```python
# LSTM Configuration
LSTM_CONFIG = {
    'sequence_length': 60,
    'num_features': 50,
    'lstm_units': [128, 64, 32],
    'dropout_rates': [0.4, 0.4, 0.3, 0.3, 0.2],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# HMM Configuration
HMM_CONFIG = {
    'n_components': 5,
    'covariance_type': 'full',
    'random_state': 42
}
```

## Data Requirements

### Minimum Data Requirements
- **Daily OHLCV data**: At least 2 years recommended
- **Options chain data**: From Polygon.io API
- **Treasury rates**: For risk-free rate calculations
- **Economic calendar**: CPI and Consumer Confidence data

### Options Data Features
- Implied Volatility (IV) for calls and puts
- Option Greeks (Delta, Gamma, Theta, Vega)
- Volume and Open Interest
- Put/Call Ratio
- Option Volume Ratio

## API Integration

**File**: `api_retry_handler.py`

Handles API requests with retry logic and rate limiting:

```python
from src.model.api_retry_handler import APIRetryHandler

# Initialize handler
handler = APIRetryHandler(api_key='your_key', use_free_tier=True)

# Make API requests with retry logic
data = handler.get_options_chain(symbol, date)
```

## Testing

Run model-specific tests:

```bash
# Run all model tests
python -m pytest tests/test_model/ -v

# Run specific model tests
python -m pytest tests/test_lstm_model.py -v
python -m pytest tests/test_market_state_classifier.py -v
```

## Best Practices

1. **Data Quality**: Ensure high-quality, clean data for training
2. **Feature Engineering**: Use domain knowledge for feature creation
3. **Validation**: Use proper train/validation/test splits
4. **Regularization**: Prevent overfitting with dropout and L2 regularization
5. **Monitoring**: Track training progress and model performance
6. **Versioning**: Save model versions with timestamps
7. **Documentation**: Document model architecture and hyperparameters

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or sequence length
2. **Overfitting**: Increase dropout rates or add more regularization
3. **Poor Performance**: Check data quality and feature engineering
4. **API Rate Limits**: Use free tier flag for rate limiting

### Performance Optimization

1. **GPU Acceleration**: Use GPU for LSTM training
2. **Data Caching**: Cache API responses to reduce calls
3. **Batch Processing**: Process data in batches for efficiency
4. **Model Optimization**: Use model quantization for inference

## Future Enhancements

1. **Ensemble Methods**: Combine multiple models for better predictions
2. **Attention Mechanisms**: Add attention layers to LSTM
3. **Transformer Models**: Experiment with transformer architectures
4. **Online Learning**: Implement online model updates
5. **Multi-Asset**: Extend to multiple assets and correlations
