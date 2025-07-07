#!/usr/bin/env python3
"""
Script to make predictions for today's option chain using pretrained HMM and LSTM models
"""

import os
import sys
import argparse
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras
import yfinance as yf

# Add the src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

# Import with try/except to handle both direct execution and module execution
try:
    from model.market_state_classifier import MarketStateClassifier
    from model.lstm_model import LSTMModel
    from common.cache.cache_manager import CacheManager
    from common.data_retriever import DataRetriever
    from prediction.calendar_config_manager import CalendarConfigManager
    from prediction.models import Predictor
except ImportError:
    # Fallback for module execution
    from src.model.market_state_classifier import MarketStateClassifier
    from src.model.lstm_model import LSTMModel
    from src.common.cache.cache_manager import CacheManager
    from src.common.data_retriever import DataRetriever
    from src.prediction.calendar_config_manager import CalendarConfigManager
    from src.prediction.models import Predictor

class TodayPredictor(Predictor):
    """Predictor class for making daily options trading predictions using pretrained HMM and LSTM models."""

    def __init__(self, symbol='SPY'):
        """Initialize the predictor with pretrained models
        
        Args:
            symbol: Stock symbol to analyze
        """
        self.symbol = symbol

        # Initialize DataRetriever for data fetching and feature calculation
        self.data_retriever = DataRetriever(symbol=symbol, quiet_mode=True)

        # Load model directory from environment variable
        model_save_base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
        self.model_dir = os.path.join(model_save_base_path, 'lstm_poc', symbol, 'latest')

        self.lstm_model = None
        self.hmm_model = None
        self.lstm_scaler = None
        self.sequence_length = 60
        self.n_features = 29  # Updated to match the saved model (29 features including time series features)

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
        """Fetch recent market data for prediction using DataRetriever
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Calculate start date based on days needed
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 Fetching {days} days of {self.symbol} data using DataRetriever...")
        
        # Use DataRetriever to fetch data with caching
        data = self.data_retriever.fetch_data_for_period(
            start_date.strftime('%Y-%m-%d'),
            'prediction'
        )
        
        # Ensure we have enough data for LSTM sequence
        if len(data) < self.sequence_length + 10:
            print(f"⚠️  Warning: Only {len(data)} days available, need at least {self.sequence_length + 10}")
            # Try to fetch more data
            extended_start_date = end_date - timedelta(days=days * 2)
            data = self.data_retriever.fetch_data_for_period(
                extended_start_date.strftime('%Y-%m-%d'), 
                'prediction'
            )
        
        print(f"✅ Loaded {len(data)} days from DataRetriever")
        print(f"   Data range: {data.index[0].date()} to {data.index[-1].date()}")
        return data
    
    def calculate_features(self, data, window=20):
        """Calculate technical features for prediction using DataRetriever
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window size
            
        Returns:
            DataFrame with calculated features
        """
        print("🔧 Calculating technical features using DataRetriever...")
        
        # Use DataRetriever to calculate technical features
        self.data_retriever.calculate_features_for_data(data, window)
        
        # Add options features - fetch real options data for today
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
        
        # Load and validate calendar configuration - REQUIRED for prediction
        try:
            calendar_config = CalendarConfigManager()
            calendar_config.print_validation_report()
        except Exception as e:
            print(f"❌ ERROR: Could not load calendar configuration: {e}")
            print(f"   Calendar configuration is required for prediction.")
            print(f"   Please ensure calendar_config.json exists and is valid.")
            print(f"   Run 'python update_calendar_config.py --show' to check current config.")
            sys.exit(1)
        
        # Add calendar features if not already present
        if ('Days_Until_Next_CPI' not in data.columns or 'Days_Since_Last_CPI' not in data.columns or
            'Days_Until_Next_CC' not in data.columns or 'Days_Since_Last_CC' not in data.columns or
            'Days_Until_Next_FFR' not in data.columns or 'Days_Since_Last_FFR' not in data.columns):
            print("📅 Adding economic calendar features...")
            
            # Use calendar config for prediction (no fallback to historical processor)
            today = datetime.now()
            
            # Calculate calendar features for today
            calendar_features = calendar_config.get_calendar_features_for_date(today)
            
            # Add calendar features to all data points (they're the same for all rows)
            for feature_name, feature_value in calendar_features.items():
                data[feature_name] = feature_value
            
            print("✅ Calendar features added using configuration file")
        
        # Define feature columns (same as in training)
        # Note: Using 29 features to match the saved model architecture (including time series features)
        feature_columns = [
            'High_Low_Range',
            'SMA20_to_SMA50', 'SMA20_to_SMA50_Lag1', 'SMA20_to_SMA50_Lag5', 
            'SMA20_to_SMA50_MA5', 'SMA20_to_SMA50_MA10', 'SMA20_to_SMA50_Std5', 'SMA20_to_SMA50_Momentum',
            'RSI', 'RSI_Lag1', 'RSI_Lag5', 'RSI_MA5', 'RSI_MA10', 'RSI_Std5', 'RSI_Momentum', 
            'RSI_Overbought', 'RSI_Oversold',
            'MACD_Hist',
            'Volume_Ratio', 'OBV', 'Put_Call_Ratio', 
            'Option_Volume_Ratio', 'Market_State',
            'Days_Until_Next_CPI', 'Days_Since_Last_CPI',
            'Days_Until_Next_CC', 'Days_Since_Last_CC',
            'Days_Until_Next_FFR', 'Days_Since_Last_FFR'
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
    
    def _find_best_spread(self, ticker, current_price, strategy_type, confidence, today, min_days=20, max_days=40):
        """Find the best spread (expiry and width) minimizing risk/reward and maximizing probability of profit"""
        candidates = []
        total_evaluated = 0
        total_rejected = 0
        expirations = ticker.options
        
        print(f"   🔍 Evaluating spreads for {strategy_type} strategy...")
        print(f"   📅 Available expirations: {len(expirations)} total")
        
        for expiry_str in expirations:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            days_to_expiry = (expiry_date - today).days
            if not (min_days <= days_to_expiry <= max_days):
                continue
            try:
                chain = ticker.option_chain(expiry_str)
            except Exception as e:
                print(f"   ⚠️  Could not fetch chain for {expiry_str}: {e}")
                continue
                
            print(f"   📊 Evaluating {expiry_str} ({days_to_expiry} days)...")
            
            for width in [3, 4, 5, 6, 7, 8, 10]:
                total_evaluated += 1
                
                if strategy_type == 'call_credit':
                    atm_strike = round(current_price)
                    otm_strike = atm_strike + width
                    calls = chain.calls
                    atm_row = calls[calls['strike'] == atm_strike]
                    otm_row = calls[calls['strike'] == otm_strike]
                    if atm_row.empty or otm_row.empty:
                        total_rejected += 1
                        continue
                    sell = atm_row.iloc[0]
                    buy = otm_row.iloc[0]
                    credit = sell['lastPrice'] - buy['lastPrice']
                    max_risk = width - credit
                    direction = 'bearish'
                else:
                    atm_strike = round(current_price)
                    otm_strike = atm_strike - width
                    puts = chain.puts
                    atm_row = puts[puts['strike'] == atm_strike]
                    otm_row = puts[puts['strike'] == otm_strike]
                    if atm_row.empty or otm_row.empty:
                        total_rejected += 1
                        continue
                    sell = atm_row.iloc[0]
                    buy = otm_row.iloc[0]
                    credit = sell['lastPrice'] - buy['lastPrice']
                    max_risk = width - credit
                    direction = 'bullish'
                    
                if max_risk <= 0 or credit <= 0:
                    total_rejected += 1
                    continue
                    
                risk_reward = credit / max_risk
                prob_profit = self._estimate_probability_of_profit(confidence, direction, width, atm_strike, otm_strike, current_price, days_to_expiry)
                
                # Calculate minimum required risk/reward ratio
                min_risk_reward = (1 - prob_profit) / prob_profit if prob_profit > 0 else float('inf')
                
                # Only include spreads that meet the minimum risk/reward requirement
                if risk_reward >= min_risk_reward:
                    candidates.append({
                        'expiry': expiry_date,
                        'width': width,
                        'atm_strike': atm_strike,
                        'otm_strike': otm_strike,
                        'sell': sell,
                        'buy': buy,
                        'credit': credit,
                        'max_risk': max_risk,
                        'risk_reward': risk_reward,
                        'prob_profit': prob_profit,
                        'min_risk_reward': min_risk_reward,
                        'days': days_to_expiry
                    })
                    print(f"      ✅ {width}pt spread: R/R={risk_reward:.2f}, Prob={prob_profit:.1%}, Min={min_risk_reward:.2f}")
                else:
                    total_rejected += 1
                    print(f"      ❌ {width}pt spread: R/R={risk_reward:.2f} < Min={min_risk_reward:.2f} (Prob={prob_profit:.1%})")
        
        print(f"   📈 Evaluation Summary:")
        print(f"      • Total spreads evaluated: {total_evaluated}")
        print(f"      • Spreads rejected: {total_rejected}")
        print(f"      • Spreads meeting criteria: {len(candidates)}")
        
        # Sort by risk/reward ascending (minimize), then probability of profit descending (maximize)
        if candidates:
            candidates.sort(key=lambda x: (x['risk_reward'], -x['prob_profit']))
            best = candidates[0]
            print(f"   🏆 Best spread selected:")
            print(f"      • {best['width']}pt spread expiring {best['expiry'].strftime('%Y-%m-%d')}")
            print(f"      • Risk/Reward: 1:{best['risk_reward']:.2f}, Probability: {best['prob_profit']:.1%}")
            return best
        else:
            print(f"   ❌ No spreads meet the minimum criteria")
            return None

    def _recommend_call_credit_spread(self, ticker, current_price, confidence, today):
        print("   📋 Structure: Sell ATM Call + Buy OTM Call")
        print("   🎯 Direction: Profit when stock goes down or stays flat")
        best = self._find_best_spread(ticker, current_price, 'call_credit', confidence, today)
        if best:
            print(f"\n   📝 Optimal Recommendation:")
            print(f"      • Sell {self.symbol} {best['atm_strike']} Call @ ${best['sell']['lastPrice']:.2f}")
            print(f"      • Buy {self.symbol} {best['otm_strike']} Call @ ${best['buy']['lastPrice']:.2f}")
            print(f"      • Spread Width: ${best['width']}")
            print(f"      • Expiration: {best['expiry'].strftime('%Y-%m-%d')} ({best['days']} days)")
            print(f"      • Credit Received: ${best['credit']:.2f}")
            print(f"      • Max Risk: ${best['max_risk']:.2f}")
            print(f"      • Risk/Reward Ratio: 1:{best['risk_reward']:.2f}")
            print(f"      • Probability of Profit: {best['prob_profit']:.1%}")
            print(f"      • Min Required R/R: 1:{best['min_risk_reward']:.2f}")
            if best['risk_reward'] > 0.3:
                print(f"      ✅ Excellent risk/reward ratio")
            elif best['risk_reward'] > 0.2:
                print(f"      ✅ Good risk/reward ratio")
            else:
                print(f"      ⚠️  Consider waiting for better pricing")
        else:
            self._print_generic_call_credit_recommendations(current_price, confidence)

    def _recommend_put_credit_spread(self, ticker, current_price, confidence, today):
        print("   📋 Structure: Sell ATM Put + Buy OTM Put")
        print("   🎯 Direction: Profit when stock goes up or stays flat")
        best = self._find_best_spread(ticker, current_price, 'put_credit', confidence, today)
        if best:
            print(f"\n   📝 Optimal Recommendation:")
            print(f"      • Sell {self.symbol} {best['atm_strike']} Put @ ${best['sell']['lastPrice']:.2f}")
            print(f"      • Buy {self.symbol} {best['otm_strike']} Put @ ${best['buy']['lastPrice']:.2f}")
            print(f"      • Spread Width: ${best['width']}")
            print(f"      • Expiration: {best['expiry'].strftime('%Y-%m-%d')} ({best['days']} days)")
            print(f"      • Credit Received: ${best['credit']:.2f}")
            print(f"      • Max Risk: ${best['max_risk']:.2f}")
            print(f"      • Risk/Reward Ratio: 1:{best['risk_reward']:.2f}")
            print(f"      • Probability of Profit: {best['prob_profit']:.1%}")
            print(f"      • Min Required R/R: 1:{best['min_risk_reward']:.2f}")
            if best['risk_reward'] > 0.3:
                print(f"      ✅ Excellent risk/reward ratio")
            elif best['risk_reward'] > 0.2:
                print(f"      ✅ Good risk/reward ratio")
            else:
                print(f"      ⚠️  Consider waiting for better pricing")
        else:
            self._print_generic_put_credit_recommendations(current_price, confidence)

    def recommend_options(self, strategy, current_price, confidence):
        print(f"\n💡 Option Recommendations:")
        print("=" * 40)
        try:
            ticker = yf.Ticker(self.symbol)
            today = datetime.now().date()
        except Exception as e:
            print(f"⚠️  Could not fetch option chain: {e}")
            ticker = None
            today = datetime.now().date()
            
        if strategy == 'Hold':
            print("   🛑 No options recommended - market conditions suggest holding cash")
            print("   💡 Consider waiting for better opportunities or reducing position sizes")
            
        elif strategy == 'Call Credit Spread':
            print(f"   🎯 Model predicts: Call Credit Spread (Bearish/Neutral)")
            print(f"   📊 Model Confidence: {confidence:.1%}")
            if ticker:
                self._recommend_call_credit_spread(ticker, current_price, confidence, today)
            else:
                self._print_generic_call_credit_recommendations(current_price, confidence)
                
        elif strategy == 'Put Credit Spread':
            print(f"   🎯 Model predicts: Put Credit Spread (Bullish/Neutral)")
            print(f"   📊 Model Confidence: {confidence:.1%}")
            if ticker:
                self._recommend_put_credit_spread(ticker, current_price, confidence, today)
            else:
                self._print_generic_put_credit_recommendations(current_price, confidence)

    def _print_generic_call_credit_recommendations(self, current_price, confidence):
        """Print generic call credit spread recommendations when specific options aren't available"""
        print(f"\n   📝 Generic Recommendations:")
        atm_strike = round(current_price)
        print(f"      • Sell {self.symbol} {atm_strike} Call")
        print(f"      • Buy {self.symbol} {atm_strike + 5} Call")
        print(f"      • Target expiration: 20-40 days")
        print(f"      • Look for high implied volatility for better premiums")
        print(f"      • Probability of Profit: {self._estimate_probability_of_profit(confidence, 'bearish'):.1%}")
    
    def _print_generic_put_credit_recommendations(self, current_price, confidence):
        """Print generic put credit spread recommendations when specific options aren't available"""
        print(f"\n   📝 Generic Recommendations:")
        atm_strike = round(current_price)
        print(f"      • Sell {self.symbol} {atm_strike} Put")
        print(f"      • Buy {self.symbol} {atm_strike - 5} Put")
        print(f"      • Target expiration: 20-40 days")
        print(f"      • Look for high implied volatility for better premiums")
        print(f"      • Probability of Profit: {self._estimate_probability_of_profit(confidence, 'bullish'):.1%}")

    def _estimate_probability_of_profit(self, confidence, direction, width=None, atm_strike=None, otm_strike=None, current_price=None, days_to_expiry=None):
        """Estimate probability of profit based on spread characteristics and model confidence"""
        base_prob = confidence
        
        # Base adjustment from model confidence
        if confidence > 0.6:
            base_prob += 0.1
        elif confidence < 0.4:
            base_prob -= 0.1
        
        # Adjust based on spread width (wider spreads = higher probability)
        if width is not None:
            width_bonus = min(0.15, (width - 3) * 0.02)  # +2% per point width, max 15%
            base_prob += width_bonus
        
        # Adjust based on distance from current price (more OTM = higher probability)
        if atm_strike is not None and otm_strike is not None and current_price is not None:
            if direction == 'bullish':  # Put credit spread
                # For put spreads, we want the stock to stay above the short put
                distance_otm = (current_price - atm_strike) / current_price
                distance_bonus = min(0.10, distance_otm * 100)  # +1% per 1% OTM, max 10%
                base_prob += distance_bonus
            else:  # Call credit spread
                # For call spreads, we want the stock to stay below the short call
                distance_otm = (atm_strike - current_price) / current_price
                distance_bonus = min(0.10, distance_otm * 100)  # +1% per 1% OTM, max 10%
                base_prob += distance_bonus
        
        # Adjust based on days to expiration (shorter = higher probability)
        if days_to_expiry is not None:
            if days_to_expiry <= 30:
                time_bonus = 0.05  # +5% for short-term
            elif days_to_expiry <= 45:
                time_bonus = 0.02  # +2% for medium-term
            else:
                time_bonus = -0.02  # -2% for long-term
            base_prob += time_bonus
        
        # Ensure probability is within reasonable bounds
        return max(0.35, min(0.85, base_prob))

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