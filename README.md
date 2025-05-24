# LSTM Stock Price Prediction

This project implements an LSTM (Long Short-Term Memory) neural network to predict stock prices using historical data from the SPY ETF.

## Features

- Fetches historical stock data using yfinance
- Implements a deep LSTM model using TensorFlow
- Includes data preprocessing and scaling
- Provides visualization of predictions
- Calculates RMSE (Root Mean Square Error) for model evaluation

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script can be run directly:

```bash
python main.py
```

Or you can use the classes in your own code:

```python
from main import StockPredictor

# Initialize predictor
predictor = StockPredictor(start_date='2015-01-01')

# Prepare data
predictor.prepare_data()

# Train model
history = predictor.train_model(epochs=50)

# Evaluate and visualize results
results = predictor.evaluate_model()
predictor.plot_results(results)
```

## Project Structure

- `main.py`: Main class that orchestrates the entire process
- `data_retriever.py`: Handles data fetching and preprocessing
- `lstm_model.py`: Implements the LSTM model architecture
- `requirements.txt`: Project dependencies

## Model Architecture

The LSTM model consists of:
- 3 LSTM layers with 50 units each
- Dropout layers (0.2) for regularization
- Dense output layer
- Adam optimizer with learning rate 0.001
- Mean Squared Error loss function

## Customization

You can customize various parameters:
- Sequence length (default: 60 days)
- Training epochs (default: 50)
- Batch size (default: 32)
- Start date for historical data
- Stock symbol (default: 'SPY') 