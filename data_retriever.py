import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import os
import json
import pickle
from pathlib import Path
from market_state_classifier import MarketStateClassifier
from options_handler import OptionsHandler

class DataRetriever:
    def __init__(self, symbol='SPY', start_date='2010-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.ticker = None
        self.state_classifier = MarketStateClassifier()
        self.options_handler = OptionsHandler(symbol, start_date=start_date)
        
        # Create cache directories
        self.cache_dir = Path('data_cache')
        self.options_cache_dir = self.cache_dir / 'options' / symbol
        self.options_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, sequence_length=60):
        """Prepare data for LSTM model with enhanced features"""
        # Fetch and calculate features
        self.fetch_data()
        self.calculate_features()
        
        # Calculate option features
        self.data = self.options_handler.calculate_option_features(self.data)
        
        # Find optimal number of states and label the data
        states = self.state_classifier.find_optimal_states(self.data)
        self.data['Market_State'] = self.state_classifier.predict_states(self.data)
        print(f"Optimal number of market states found: {states}")
        
        # Calculate option trading signals
        self.data = self.options_handler.calculate_option_signals(self.data)
        
        # Prepare feature matrix for LSTM (including all features and market states)
        feature_columns = [
            'Returns', 'Log_Returns', 'Volatility', 'High_Low_Range',
            'Price_to_SMA20', 'SMA20_to_SMA50',
            'RSI', 'MACD', 'MACD_Hist',
            'Volume_Ratio', 'OBV',
            'ATM_Call_Return', 'ATM_Put_Return',
            'Call_Put_Ratio', 'Option_Volume_Ratio',
            'Market_State'  # Add market state as a feature
        ]
        
        # Scale features
        self.features = self.scaler.fit_transform(self.data[feature_columns])
        
        # Create sequences using numpy operations for better performance
        n_samples = len(self.features) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_samples, sequence_length, len(feature_columns)))
        y = np.zeros(n_samples)
        
        # Fill arrays using vectorized operations
        for i in range(sequence_length, len(self.features)):
            X[i-sequence_length] = self.features[i-sequence_length:i]
            y[i-sequence_length] = self.data['Option_Signal'].iloc[i]
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        return X_train, y_train, X_test, y_test

    def fetch_data(self):
        """Fetch data from Yahoo Finance"""
        print(f"\nFetching data from {self.start_date} onwards...")
        self.ticker = yf.Ticker(self.symbol)
        
        # Get data and ensure index is timezone-naive
        self.data = self.ticker.history(start=self.start_date)
        if self.data.empty:
            raise ValueError(f"No data retrieved for {self.symbol} from {self.start_date}")
            
        self.data.index = self.data.index.tz_localize(None)
        print(f"Initial data range: {self.data.index[0]} to {self.data.index[-1]}")
        
        # Filter data to start from the specified date
        start_date = pd.Timestamp(self.start_date).tz_localize(None)
        mask = self.data.index >= start_date
        self.data = self.data[mask].copy()
        print(f"Filtered data range: {self.data.index[0]} to {self.data.index[-1]}")
        
        if self.data.empty:
            raise ValueError(f"No data available after filtering for dates >= {self.start_date}")
            
        return self.data

    def calculate_features(self, window=20):
        """Calculate technical features"""
        # Basic returns and volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Volatility measures
        self.data['Volatility'] = self.data['Returns'].rolling(window=window).std()
        self.data['High_Low_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        
        # Trend indicators
        self.data['SMA20'] = self.data['Close'].rolling(window=window).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Price_to_SMA20'] = self.data['Close'] / self.data['SMA20']
        self.data['SMA20_to_SMA50'] = self.data['SMA20'] / self.data['SMA50']
        
        # Momentum indicators
        self.data['RSI'] = self._calculate_rsi(self.data['Close'], window)
        self.data['MACD'], self.data['MACD_Signal'] = self._calculate_macd(self.data['Close'])
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # Volume features
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=window).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        self.data['OBV'] = self._calculate_obv(self.data['Close'], self.data['Volume'])
        
        # Drop NaN values
        self.data = self.data.dropna()

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_obv(self, prices, volume):
        """Calculate On-Balance Volume"""
        price_change = prices.diff()
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(prices)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def get_state_description(self, state_id):
        """Generate description for each state based on its characteristics"""
        state_data = self.data[self.data['Market_State'] == state_id]
        
        avg_return = state_data['Returns'].mean()
        avg_vol = state_data['Volatility'].mean()
        avg_volume = state_data['Volume_Ratio'].mean()
        avg_price_sma = state_data['Price_to_SMA20'].mean()
        avg_sma_trend = state_data['SMA20_to_SMA50'].mean()
        
        # Determine state characteristics
        trend = "Bullish" if avg_return > 0 else "Bearish"
        volatility = "High" if avg_vol > self.data['Volatility'].mean() else "Low"
        volume = "High" if avg_volume > 1 else "Low"
        price_trend = "Above MA" if avg_price_sma > 1 else "Below MA"
        ma_trend = "Upward" if avg_sma_trend > 1 else "Downward"
        
        return (f"State {state_id}: {trend} with {volatility} volatility, {volume} volume, "
                f"Price {price_trend}, Moving Averages trending {ma_trend}") 