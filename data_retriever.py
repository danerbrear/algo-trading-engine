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
from cache_manager import CacheManager

class DataRetriever:
    def __init__(self, symbol='SPY', hmm_start_date='2010-01-01', lstm_start_date='2020-01-01'):
        """Initialize DataRetriever with separate date ranges for HMM and LSTM
        
        Args:
            symbol: Stock symbol to analyze
            hmm_start_date: Start date for HMM training data (market state classification)
            lstm_start_date: Start date for LSTM training data (options signal prediction)
        """
        self.symbol = symbol
        self.hmm_start_date = hmm_start_date
        self.lstm_start_date = lstm_start_date
        self.start_date = lstm_start_date  # Backward compatibility
        self.scaler = StandardScaler()
        self.data = None
        self.hmm_data = None  # Separate data for HMM training
        self.lstm_data = None  # Separate data for LSTM training
        self.features = None
        self.ticker = None
        self.state_classifier = MarketStateClassifier()
        self.cache_manager = CacheManager()
        self.options_handler = OptionsHandler(symbol, start_date=lstm_start_date, cache_dir=self.cache_manager.base_dir)
        
        print(f"🔄 DataRetriever Configuration:")
        print(f"   📊 HMM training data: {hmm_start_date} onwards (for market state classification)")
        print(f"   🎯 LSTM training data: {lstm_start_date} onwards (for options signal prediction)")
        
    def prepare_data(self, sequence_length=60):
        """Prepare data for LSTM model with enhanced features using separate date ranges"""
        print(f"\n📈 Phase 1: Preparing HMM training data from {self.hmm_start_date}")
        # Fetch HMM training data (longer history for market state patterns)
        self.hmm_data = self.fetch_data_for_period(self.hmm_start_date, 'hmm')
        self.calculate_features_for_data(self.hmm_data)
        
        print(f"\n🎯 Phase 2: Training HMM on market data ({len(self.hmm_data)} samples)")
        # Train HMM on the longer historical data
        states = self.state_classifier.find_optimal_states(self.hmm_data)
        print(f"✅ Optimal number of market states found: {states}")
        
        print(f"\n📊 Phase 3: Preparing LSTM training data from {self.lstm_start_date}")
        # Fetch LSTM training data (more recent data for options trading)
        self.lstm_data = self.fetch_data_for_period(self.lstm_start_date, 'lstm')
        self.calculate_features_for_data(self.lstm_data)
        
        # Calculate option features for LSTM data
        self.lstm_data = self.options_handler.calculate_option_features(self.lstm_data)
        
        print(f"\n🔮 Phase 4: Applying trained HMM to LSTM data")
        # Apply the trained HMM to the LSTM data
        self.lstm_data['Market_State'] = self.state_classifier.predict_states(self.lstm_data)
        
        print(f"\n💰 Phase 5: Generating option signals for LSTM data")
        # Calculate option trading signals for LSTM data
        self.lstm_data = self.options_handler.calculate_option_signals(self.lstm_data)
        
        # Use LSTM data as the main dataset for training
        self.data = self.lstm_data
        
        print(f"\n🚀 Phase 6: Preparing LSTM features ({len(self.lstm_data)} samples)")
        # Prepare feature matrix for LSTM (including all features and market states)
        feature_columns = [
            'Returns', 'Log_Returns', 'Volatility', 'High_Low_Range',
            'Price_to_SMA20', 'SMA20_to_SMA50',
            'RSI', 'MACD', 'MACD_Hist',
            'Volume_Ratio', 'OBV',
            'ATM_Call_Return', 'ATM_Put_Return',
            'Put_Call_Ratio', 'Option_Volume_Ratio',
            'Market_State'  # Add market state as a feature
        ]
        
        # Scale features
        self.features = self.scaler.fit_transform(self.lstm_data[feature_columns])
        
        # Critical: Check for NaN or infinite values that cause training to fail
        print(f"🔍 Data validation:")
        print(f"   NaN values in features: {np.isnan(self.features).sum()}")
        print(f"   Infinite values in features: {np.isinf(self.features).sum()}")
        print(f"   NaN values in labels: {self.lstm_data['Option_Signal'].isna().sum()}")
        
        # Clean the data - remove rows with NaN or infinite values
        valid_mask = ~(np.isnan(self.features).any(axis=1) | np.isinf(self.features).any(axis=1))
        self.features = self.features[valid_mask]
        clean_lstm_data = self.lstm_data[valid_mask].copy()
        
        print(f"   📊 Cleaned data: {len(self.features)} samples (removed {(~valid_mask).sum()} invalid samples)")
        
        # Create sequences using numpy operations for better performance
        n_samples = len(self.features) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_samples, sequence_length, len(feature_columns)))
        y = np.zeros(n_samples)
        
        # Fill arrays using vectorized operations
        for i in range(sequence_length, len(self.features)):
            X[i-sequence_length] = self.features[i-sequence_length:i]
            y[i-sequence_length] = clean_lstm_data['Option_Signal'].iloc[i]
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        print(f"✅ Data preparation complete:")
        print(f"   🏋️ Training samples: {len(X_train)}")
        print(f"   🧪 Testing samples: {len(X_test)}")
        print(f"   📊 Features per sample: {len(feature_columns)}")
        
        return X_train, y_train, X_test, y_test

    def fetch_data_for_period(self, start_date: str, data_type: str = 'general'):
        """Fetch data for a specific period with caching"""
        cache_suffix = f'_{data_type}_data'
        
        # Try to load from cache first
        cached_data = self.cache_manager.load_date_from_cache(
            pd.Timestamp(start_date),
            cache_suffix,
            'stocks',
            self.symbol
        )
        
        if cached_data is not None:
            print(f"📋 Loading cached {data_type.upper()} data from {start_date} ({len(cached_data)} samples)")
            return cached_data
            
        print(f"🌐 Fetching {data_type.upper()} data from {start_date} onwards...")
        if self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        
        # Get data and ensure index is timezone-naive
        data = self.ticker.history(start=start_date)
        if data.empty:
            raise ValueError(f"No data retrieved for {self.symbol} from {start_date}")
            
        data.index = data.index.tz_localize(None)
        print(f"📊 Initial {data_type} data range: {data.index[0]} to {data.index[-1]}")
        
        # Filter data to start from the specified date
        start_date_ts = pd.Timestamp(start_date).tz_localize(None)
        mask = data.index >= start_date_ts
        data = data[mask].copy()
        print(f"✂️ Filtered {data_type} data range: {data.index[0]} to {data.index[-1]} ({len(data)} samples)")
        
        if data.empty:
            raise ValueError(f"No data available after filtering for dates >= {start_date}")
            
        # Cache the filtered data
        self.cache_manager.save_date_to_cache(
            pd.Timestamp(start_date),
            data,
            cache_suffix,
            'stocks',
            self.symbol
        )
        
        return data

    def calculate_features_for_data(self, data: pd.DataFrame, window=20):
        """Calculate technical features for a given dataset"""
        # Basic returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility measures
        data['Volatility'] = data['Returns'].rolling(window=window).std()
        data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
        
        # Trend indicators
        data['SMA20'] = data['Close'].rolling(window=window).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['Price_to_SMA20'] = data['Close'] / data['SMA20']
        data['SMA20_to_SMA50'] = data['SMA20'] / data['SMA50']
        
        # Momentum indicators
        data['RSI'] = self._calculate_rsi(data['Close'], window)
        data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Volume features
        data['Volume_SMA'] = data['Volume'].rolling(window=window).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['OBV'] = self._calculate_obv(data['Close'], data['Volume'])
        
        # Drop NaN values
        data.dropna(inplace=True)

    def fetch_data(self):
        """Legacy method for backward compatibility - uses LSTM start date"""
        return self.fetch_data_for_period(self.lstm_start_date, 'lstm')

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