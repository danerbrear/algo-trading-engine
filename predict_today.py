#!/usr/bin/env python3
"""
Script to make predictions for today's option chain using pretrained HMM and LSTM models
"""

import os
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import pandas as pd
import keras
import yfinance as yf
from market_state_classifier import MarketStateClassifier
from lstm_model import LSTMModel
from cache_manager import CacheManager

class TodayPredictor:
    """Predictor class for making daily options trading predictions using pretrained HMM and LSTM models."""
    
    def __init__(self, symbol='SPY'):
        """Initialize the predictor with pretrained models
        
        Args:
            symbol: Stock symbol to analyze
        """
        self.symbol = symbol
        
        # Load model directory from environment variable
        model_save_base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
        self.model_dir = os.path.join(model_save_base_path, 'lstm_poc', 'latest')
            
        self.lstm_model = None
        self.hmm_model = None
        self.lstm_scaler = None
        self.sequence_length = 60
        self.n_features = 11
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load pretrained HMM and LSTM models"""
        print(f"🔄 Loading models from {self.model_dir}...")
        
        # Load LSTM model
        lstm_path = os.path.join(self.model_dir, 'model.keras')
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
        
        self.lstm_model = LSTMModel(sequence_length=self.sequence_length, n_features=self.n_features)
        self.lstm_model.model = keras.models.load_model(lstm_path)
        print(f"✅ LSTM model loaded from {lstm_path}")
        
        # Load LSTM scaler
        scaler_path = os.path.join(self.model_dir, 'lstm_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.lstm_scaler = pickle.load(f)
            print(f"✅ LSTM scaler loaded from {scaler_path}")
        else:
            print(f"⚠️  LSTM scaler not found at {scaler_path}")
        
        # Load HMM model
        hmm_path = os.path.join(self.model_dir, 'hmm_model.pkl')
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(f"HMM model not found at {hmm_path}")
        
        with open(hmm_path, 'rb') as f:
            hmm_data = pickle.load(f)
        
        self.hmm_model = MarketStateClassifier(max_states=hmm_data['max_states'])
        self.hmm_model.hmm_model = hmm_data['hmm_model']
        self.hmm_model.scaler = hmm_data['scaler']
        self.hmm_model.n_states = hmm_data['n_states']
        
        print(f"✅ HMM model loaded from {hmm_path}")
        print(f"   Number of states: {self.hmm_model.n_states}")
        
    def fetch_recent_data(self, days=90):
        """Fetch recent market data for prediction
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Increase the fetch period to ensure we have enough data after feature calculation
        # We need at least sequence_length + some buffer for feature calculation
        fetch_days = max(days, self.sequence_length + 50)  # Add 50 days buffer for feature calculation
        
        print(f"📊 Fetching {fetch_days} days of {self.symbol} data (need {self.sequence_length} for LSTM + buffer)...")
        
        # Load from cache
        cache_manager = CacheManager()
        
        # Look for cached data files in data_cache/stocks/SPY
        cache_dir = cache_manager.get_cache_dir('stocks', self.symbol)
        if not cache_dir.exists():
            raise ValueError(f"No cache directory found for {self.symbol}")
        
        print(f"🔍 Looking for cached data in {cache_dir}")
        
        # Get all cached files and sort by date (newest first)
        cached_files = []
        for file_path in cache_dir.glob('*.pkl'):
            try:
                # Try to extract date from filename (format: YYYY-MM-DD_suffix.pkl)
                filename = file_path.stem
                if '_' in filename:
                    date_str = filename.split('_')[0]
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    cached_files.append((file_date, file_path))
            except ValueError:
                continue
        
        if not cached_files:
            raise ValueError(f"No cached files found for {self.symbol}")
        
        # Sort by date (newest first) and use the first available file
        cached_files.sort(key=lambda x: x[0], reverse=True)
        file_date, file_path = cached_files[0]
        
        print(f"📁 Using cached file: {file_path.name} (contains data from {file_date.date()})")
        
        # Load the cached data
        cached_data = cache_manager.load_from_cache(file_path.name, 'stocks', self.symbol)
        if cached_data is None or cached_data.empty:
            raise ValueError(f"Failed to load data from {file_path.name}")
        
        print(f"   Cached data range: {cached_data.index[0].date()} to {cached_data.index[-1].date()}")
        
        # Get the most recent fetch_days from the cached data
        if len(cached_data) >= fetch_days:
            # Take the last fetch_days from the data
            filtered_data = cached_data.tail(fetch_days).copy()
        else:
            # Use all available data if we don't have enough
            filtered_data = cached_data.copy()
            print(f"   ⚠️  Only {len(filtered_data)} days available, using all cached data")
        
        print(f"✅ Loaded {len(filtered_data)} days from cache: {file_path.name}")
        print(f"   Data range: {filtered_data.index[0].date()} to {filtered_data.index[-1].date()}")
        return filtered_data
    
    def calculate_features(self, data, window=20):
        """Calculate technical features for prediction
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window size
            
        Returns:
            DataFrame with calculated features
        """
        print("🔧 Calculating technical features...")
        
        # Basic returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=window).std()
        data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
        
        # Trend indicators
        data['SMA20'] = data['Close'].rolling(window=window).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['Price_to_SMA20'] = data['Close'] / data['SMA20']
        data['SMA20_to_SMA50'] = data['SMA20'] / data['SMA50']
        
        # Momentum indicators
        data['RSI'] = self.calculate_rsi(data['Close'], window=14)
        macd_line, macd_signal = self.calculate_macd(data['Close'])
        data['MACD_Hist'] = macd_line - macd_signal
        
        # Volume features
        data['Volume_SMA'] = data['Volume'].rolling(window=window).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['OBV'] = self.calculate_obv(data['Close'], data['Volume'])
        
        # Options features - fetch real options data for today
        current_price = data['Close'].iloc[-1]  # Use most recent price
        options_features = self.fetch_today_options_data(current_price)
        data['Put_Call_Ratio'] = options_features['Put_Call_Ratio']
        data['Option_Volume_Ratio'] = options_features['Option_Volume_Ratio']
        
        # Market state (will be calculated by HMM)
        data['Market_State'] = 0  # Placeholder
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        print(f"✅ Calculated features for {len(data)} samples")
        return data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def calculate_obv(self, prices, volume):
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
    
    def fetch_today_options_data(self, current_price):
        """Fetch today's option chain data to calculate real options features
        
        Args:
            current_price: Current stock price
            
        Returns:
            Dict with Put_Call_Ratio and Option_Volume_Ratio
        """
        try:
            print("📊 Fetching today's option chain data...")
            
            # Fetch option chain for today using yfinance
            ticker = yf.Ticker(self.symbol)
            option_chain = ticker.option_chain()
            
            if not option_chain or not hasattr(option_chain, 'calls') or not hasattr(option_chain, 'puts'):
                print("⚠️  No option chain data available for today")
                return {'Put_Call_Ratio': 1.0, 'Option_Volume_Ratio': 0.1}
            
            # Get ATM options
            atm_call, atm_put = self.get_atm_options(option_chain, current_price)
            
            if atm_call is not None and atm_put is not None:
                # Calculate real options features
                call_volume = atm_call.get('volume', 0) or 0
                put_volume = atm_put.get('volume', 0) or 0
                
                put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
                option_volume_ratio = (call_volume + put_volume) / 1000000  # Normalize by 1M
                
                print(f"✅ Real options features calculated:")
                print(f"   Put/Call Ratio: {put_call_ratio:.3f}")
                print(f"   Option Volume Ratio: {option_volume_ratio:.3f}")
                
                return {
                    'Put_Call_Ratio': put_call_ratio,
                    'Option_Volume_Ratio': option_volume_ratio
                }
            else:
                print("⚠️  Could not get ATM options data")
                return {'Put_Call_Ratio': 1.0, 'Option_Volume_Ratio': 0.1}
                
        except Exception as e:
            print(f"⚠️  Error fetching options data: {e}")
            print("   Using placeholder values...")
            return {'Put_Call_Ratio': 1.0, 'Option_Volume_Ratio': 0.1}
    
    def get_atm_options(self, option_chain, current_price):
        """Get ATM call and put options from the option chain"""
        try:
            # Find ATM call and put options (closest to current price)
            calls = option_chain.calls
            puts = option_chain.puts
            
            # Find the closest strike to current price
            atm_strike = round(current_price)
            
            # Get options at ATM strike
            atm_calls = calls[calls['strike'] == atm_strike]
            atm_puts = puts[puts['strike'] == atm_strike]
            
            # If no exact match, find closest
            if atm_calls.empty:
                closest_call_idx = (calls['strike'] - current_price).abs().idxmin()
                atm_calls = calls.loc[[closest_call_idx]]
            
            if atm_puts.empty:
                closest_put_idx = (puts['strike'] - current_price).abs().idxmin()
                atm_puts = puts.loc[[closest_put_idx]]
            
            if not atm_calls.empty and not atm_puts.empty:
                # Get the first ATM call and put options
                atm_call = atm_calls.iloc[0]
                atm_put = atm_puts.iloc[0]
                
                return atm_call, atm_put
            else:
                return None, None
                
        except Exception as e:
            print(f"⚠️  Error getting ATM options: {e}")
            return None, None
    
    def check_current_data_availability(self, data):
        """Check if we have stock and option data for the current date"""
        print("🔍 Checking data availability for current date...")
        
        # Check stock data for current date
        today = datetime.now().date()
        latest_stock_date = data.index[-1].date()
        
        if latest_stock_date < today:
            print(f"⚠️  WARNING: Stock data is not current!")
            print(f"   Latest stock data: {latest_stock_date}")
            print(f"   Current date: {today}")
            print(f"   Data is {(today - latest_stock_date).days} days old")
        else:
            print(f"✅ Stock data is current (latest: {latest_stock_date})")
        
        # Check option data availability
        try:
            ticker = yf.Ticker(self.symbol)
            option_chain = ticker.option_chain()
            
            if option_chain and hasattr(option_chain, 'calls') and hasattr(option_chain, 'puts'):
                # Check if we have options for today
                calls = option_chain.calls
                puts = option_chain.puts
                
                if not calls.empty and not puts.empty:
                    print(f"✅ Option chain data available")
                    print(f"   Calls: {len(calls)} contracts")
                    print(f"   Puts: {len(puts)} contracts")
                else:
                    print(f"⚠️  WARNING: Option chain data is incomplete!")
                    print(f"   Calls available: {len(calls) if not calls.empty else 0}")
                    print(f"   Puts available: {len(puts) if not puts.empty else 0}")
            else:
                print(f"⚠️  WARNING: No option chain data available!")
        except Exception as e:
            print(f"⚠️  WARNING: Could not fetch option chain data: {e}")
        
        print("=" * 50)
    
    def predict_market_state(self, data):
        """Predict market state using HMM model"""
        print("🔮 Predicting market state...")
        
        # Prepare HMM features
        feature_matrix = np.column_stack([
            data['Returns'],
            data['Volatility'],
            data['Price_to_SMA20'],
            data['SMA20_to_SMA50'],
            data['Volume_Ratio']
        ])
        
        # Scale features
        scaled_features = self.hmm_model.scaler.transform(feature_matrix)
        
        # Predict states
        market_states = self.hmm_model.hmm_model.predict(scaled_features)
        
        # Update market states in data
        data['Market_State'] = market_states
        
        current_state = market_states[-1]
        print(f"✅ Current market state: {current_state}")
        
        return data, current_state
    
    def prepare_lstm_features(self, data):
        """Prepare features for LSTM prediction"""
        print("🧠 Preparing LSTM features...")
        
        # Define feature columns (same as in training)
        feature_columns = [
            'Log_Returns', 'Volatility', 'High_Low_Range',
            'SMA20_to_SMA50', 'RSI', 'MACD_Hist',
            'Volume_Ratio', 'OBV', 'Put_Call_Ratio', 
            'Option_Volume_Ratio', 'Market_State'
        ]
        
        # Scale features
        if self.lstm_scaler is not None:
            features = self.lstm_scaler.transform(data[feature_columns])
        else:
            # Create new scaler if not available
            scaler = StandardScaler()
            features = scaler.fit_transform(data[feature_columns])
        
        return features
    
    def create_sequences(self, features):
        """Create sequences for LSTM prediction"""
        if len(features) < self.sequence_length:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Use the last sequence_length samples
        sequence = features[-self.sequence_length:]
        X = sequence.reshape(1, self.sequence_length, self.n_features)
        
        return X
    
    def make_prediction(self):
        """Make prediction for today's market"""
        print(f"\n🎯 Making prediction for {self.symbol} today...")
        print("=" * 60)
        
        # Fetch recent data
        data = self.fetch_recent_data(days=90)
        
        # Check data availability for current date
        self.check_current_data_availability(data)
        
        # Calculate features
        data = self.calculate_features(data)
        
        # Check if we have enough data after feature calculation
        if len(data) < self.sequence_length:
            print(f"⚠️  Warning: Only {len(data)} samples available after feature calculation")
            print(f"   Need at least {self.sequence_length} samples for LSTM prediction")
            print(f"   This might indicate insufficient historical data or data quality issues")
            
            # Try to fetch more data if we don't have enough
            if len(data) < self.sequence_length:
                print(f"🔄 Attempting to fetch more data...")
                # Fetch more data (double the original request)
                data = self.fetch_recent_data(days=180)
                data = self.calculate_features(data)
                
                if len(data) < self.sequence_length:
                    raise ValueError(f"Insufficient data: {len(data)} samples available, need {self.sequence_length} for LSTM prediction")
        
        print(f"✅ Have {len(data)} samples available for prediction")
        
        # Predict market state
        data, market_state = self.predict_market_state(data)
        
        # Prepare LSTM features
        features = self.prepare_lstm_features(data)
        
        # Create sequence for prediction
        X = self.create_sequences(features)
        
        # Make LSTM prediction
        prediction = self.lstm_model.predict(X)[0]
        probabilities = self.lstm_model.predict_proba(X)[0]
        
        # Strategy labels
        strategy_labels = ['Hold', 'Call Credit Spread', 'Put Credit Spread']
        predicted_strategy = strategy_labels[prediction]
        confidence = probabilities[prediction]
        
        # Get current market data
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1].date()
        
        # Display results
        print(f"\n📊 Prediction Results for {current_date}:")
        print(f"   Current {self.symbol} Price: ${current_price:.2f}")
        print(f"   Market State: {market_state}")
        print(f"   Predicted Strategy: {predicted_strategy}")
        print(f"   Confidence: {confidence:.2%}")
        
        # Display all probabilities
        print(f"\n📈 Strategy Probabilities:")
        for i, (label, prob) in enumerate(zip(strategy_labels, probabilities)):
            print(f"   {i}: {label:18} {prob:.2%}")
        
        # Provide option recommendations
        self.recommend_options(predicted_strategy, current_price, confidence)
        
        return {
            'date': current_date,
            'price': current_price,
            'market_state': market_state,
            'predicted_strategy': predicted_strategy,
            'confidence': confidence,
            'probabilities': dict(zip(strategy_labels, probabilities))
        }
    
    def recommend_options(self, strategy, current_price, confidence):
        """Provide specific option recommendations"""
        print(f"\n💡 Option Recommendations:")
        print("=" * 40)
        
        if strategy == 'Hold':
            print("   🛑 No options recommended - market conditions suggest holding cash")
            print("   💡 Consider waiting for better opportunities or reducing position sizes")
            
        elif strategy == 'Call Credit Spread':
            print("   📉 Strategy: Call Credit Spread (Bearish/Neutral)")
            print("   📋 Structure: Sell ATM Call + Buy ATM+5 Call")
            print("   🎯 Direction: Profit when stock goes down or stays flat")
            print("   💰 Max Profit: Premium received")
            print("   ⚠️  Max Loss: $5 - Premium received")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            # Specific recommendations
            atm_strike = round(current_price)
            otm_strike = atm_strike + 5
            
            print(f"\n   📝 Specific Recommendations:")
            print(f"      • Sell {self.symbol} {atm_strike} Call")
            print(f"      • Buy {self.symbol} {otm_strike} Call")
            print(f"      • Target expiration: 30-45 days")
            print(f"      • Look for high implied volatility for better premiums")
            
        elif strategy == 'Put Credit Spread':
            print("   📈 Strategy: Put Credit Spread (Bullish/Neutral)")
            print("   📋 Structure: Sell ATM Put + Buy ATM-5 Put")
            print("   🎯 Direction: Profit when stock goes up or stays flat")
            print("   💰 Max Profit: Premium received")
            print("   ⚠️  Max Loss: $5 - Premium received")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            # Specific recommendations
            atm_strike = round(current_price)
            otm_strike = atm_strike - 5
            
            print(f"\n   📝 Specific Recommendations:")
            print(f"      • Sell {self.symbol} {atm_strike} Put")
            print(f"      • Buy {self.symbol} {otm_strike} Put")
            print(f"      • Target expiration: 30-45 days")
            print(f"      • Look for high implied volatility for better premiums")

def main():
    parser = argparse.ArgumentParser(description="Make option predictions for today")
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol to analyze')
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = TodayPredictor(symbol=args.symbol)
        
        # Make prediction
        result = predictor.make_prediction()
        
        # Create predictions directory if it doesn't exist
        predictions_dir = 'predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save results to file in predictions directory
        output_file = os.path.join(predictions_dir, f"prediction_{args.symbol}_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Prediction Results for {args.symbol} - {result['date']}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Price: ${result['price']:.2f}\n")
            f.write(f"Market State: {result['market_state']}\n")
            f.write(f"Predicted Strategy: {result['predicted_strategy']}\n")
            f.write(f"Confidence: {result['confidence']:.2%}\n")
            f.write(f"\nAll Probabilities:\n")
            for strategy, prob in result['probabilities'].items():
                f.write(f"  {strategy}: {prob:.2%}\n")
        
        print(f"\n💾 Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 