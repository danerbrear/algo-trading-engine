import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial

class HMMStateClassifier(BaseEstimator, ClassifierMixin):
    """Custom classifier for HMM state detection with GridSearchCV support"""
    def __init__(self, n_components=3, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = None
        
    def fit(self, X, y=None):
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            n_iter=self.n_iter,
            covariance_type='full',
            random_state=42
        )
        self.model.fit(X)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y=None):
        """Return negative log likelihood score (higher is better for sklearn)"""
        return self.model.score(X)

class DataRetriever:
    def __init__(self, symbol='SPY', start_date='2010-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.hmm_model = None
        self.n_states = None
        self.ticker = None
        
    def fetch_data(self):
        """Fetch data from Yahoo Finance"""
        self.ticker = yf.Ticker(self.symbol)
        self.data = self.ticker.history(start=self.start_date)
        return self.data
    
    def get_atm_options(self, current_price, expiry_date):
        """Get at-the-money options for a given expiry date"""
        options = self.ticker.option_chain(expiry_date)
        
        # Get closest strike price to current price for calls
        calls = options.calls
        calls['Strike_Diff'] = abs(calls['strike'] - current_price)
        atm_call = calls.nsmallest(1, 'Strike_Diff').iloc[0]
        
        # Get closest strike price to current price for puts
        puts = options.puts
        puts['Strike_Diff'] = abs(puts['strike'] - current_price)
        atm_put = puts.nsmallest(1, 'Strike_Diff').iloc[0]
        
        return atm_call, atm_put
    
    def calculate_option_features(self):
        """Calculate option-related features with batch processing"""
        # Initialize option features
        self.data['ATM_Call_Return'] = np.nan
        self.data['ATM_Put_Return'] = np.nan
        self.data['Call_Put_Ratio'] = np.nan
        self.data['Option_Volume_Ratio'] = np.nan
        
        def process_date_batch(dates_batch):
            results = []
            current_expiry = None
            current_options = None
            
            for current_date in dates_batch:
                current_price = self.data.loc[current_date, 'Close']
                
                try:
                    # Only fetch new options data if we need to
                    if current_expiry is None or current_date.to_pydatetime() > datetime.strptime(current_expiry, '%Y-%m-%d').replace(tzinfo=current_date.tzinfo):
                        expiry_dates = self.ticker.options
                        next_expiry_dates = [
                            date for date in expiry_dates 
                            if datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=current_date.tzinfo) > current_date.to_pydatetime()
                        ]
                        if next_expiry_dates:
                            current_expiry = min(next_expiry_dates)
                            current_options = self.ticker.option_chain(current_expiry)
                        else:
                            raise ValueError("No valid expiry dates found")
                    
                    # Get ATM options
                    atm_call, atm_put = self._get_atm_options_cached(current_price, current_options)
                    
                    results.append({
                        'date': current_date,
                        'ATM_Call_Return': (atm_call['lastPrice'] - atm_call['lastPrice']) / atm_call['lastPrice'] 
                                          if atm_call['lastPrice'] > 0 else 0,
                        'ATM_Put_Return': (atm_put['lastPrice'] - atm_put['lastPrice']) / atm_put['lastPrice']
                                         if atm_put['lastPrice'] > 0 else 0,
                        'Call_Put_Ratio': atm_call['volume'] / atm_put['volume'] if atm_put['volume'] > 0 else 1,
                        'Option_Volume_Ratio': (atm_call['volume'] + atm_put['volume']) / 
                                             self.data.loc[current_date, 'Volume'] 
                                             if self.data.loc[current_date, 'Volume'] > 0 else 0
                    })
                except (IndexError, KeyError, ValueError) as e:
                    # Use previous values or defaults
                    if results:
                        prev_result = results[-1]
                        results.append({
                            'date': current_date,
                            'ATM_Call_Return': prev_result['ATM_Call_Return'],
                            'ATM_Put_Return': prev_result['ATM_Put_Return'],
                            'Call_Put_Ratio': prev_result['Call_Put_Ratio'],
                            'Option_Volume_Ratio': prev_result['Option_Volume_Ratio']
                        })
                    else:
                        results.append({
                            'date': current_date,
                            'ATM_Call_Return': 0,
                            'ATM_Put_Return': 0,
                            'Call_Put_Ratio': 1,
                            'Option_Volume_Ratio': 0
                        })
            
            return results
        
        # Process dates in batches using parallel processing
        dates = self.data.index[1:]
        batch_size = 20  # Adjust based on your needs
        date_batches = [dates[i:i + batch_size] for i in range(0, len(dates), batch_size)]
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            all_results = []
            for batch_results in executor.map(process_date_batch, date_batches):
                all_results.extend(batch_results)
        
        # Update the dataframe with results
        for result in all_results:
            self.data.loc[result['date'], 'ATM_Call_Return'] = result['ATM_Call_Return']
            self.data.loc[result['date'], 'ATM_Put_Return'] = result['ATM_Put_Return']
            self.data.loc[result['date'], 'Call_Put_Ratio'] = result['Call_Put_Ratio']
            self.data.loc[result['date'], 'Option_Volume_Ratio'] = result['Option_Volume_Ratio']
        
        # Fill any remaining NaN values
        self.data[['ATM_Call_Return', 'ATM_Put_Return', 
                   'Call_Put_Ratio', 'Option_Volume_Ratio']] = \
            self.data[['ATM_Call_Return', 'ATM_Put_Return', 
                       'Call_Put_Ratio', 'Option_Volume_Ratio']].fillna(method='ffill').fillna(method='bfill')

    def _get_atm_options_cached(self, current_price, options):
        """Get at-the-money options using cached option chain data"""
        # Get closest strike price to current price for calls
        calls = options.calls
        calls['Strike_Diff'] = abs(calls['strike'] - current_price)
        atm_call = calls.nsmallest(1, 'Strike_Diff').iloc[0]
        
        # Get closest strike price to current price for puts
        puts = options.puts
        puts['Strike_Diff'] = abs(puts['strike'] - current_price)
        atm_put = puts.nsmallest(1, 'Strike_Diff').iloc[0]
        
        return atm_call, atm_put
    
    def calculate_features(self, window=20):
        """Calculate technical features for regime classification"""
        # Returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Trend indicators
        self.data['SMA20'] = self.data['Close'].rolling(window=window).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        
        # Volatility
        self.data['Volatility'] = self.data['Returns'].rolling(window=window).std()
        
        # Price relative to moving averages
        self.data['Price_to_SMA20'] = self.data['Close'] / self.data['SMA20']
        self.data['SMA20_to_SMA50'] = self.data['SMA20'] / self.data['SMA50']
        
        # Volume features
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=window).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Options features
        self.calculate_option_features()
        
        # Drop NaN values
        self.data = self.data.dropna()
    
    def find_optimal_states(self, max_states=5):
        """Find optimal number of states using HMM with multiple criteria"""
        # Prepare features for HMM
        feature_matrix = np.column_stack([
            self.data['Returns'],
            self.data['Volatility'],
            self.data['Price_to_SMA20'],
            self.data['SMA20_to_SMA50'],
            self.data['Volume_Ratio'],
            self.data['ATM_Call_Return'],
            self.data['ATM_Put_Return'],
            self.data['Call_Put_Ratio'],
            self.data['Option_Volume_Ratio']
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Try different numbers of states and evaluate using multiple criteria
        best_score = float('-inf')
        best_n_states = 3  # Default
        scores = []
        
        for n_states in range(2, max_states + 1):
            try:
                # Train HMM with current number of states
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='full',
                    n_iter=1000,  # Increased iterations
                    random_state=42,
                    tol=1e-5,    # Stricter convergence
                    init_params='kmeans'  # Better initialization
                )
                
                # Fit and get scores
                model.fit(scaled_features)
                aic = model.aic(scaled_features)
                bic = model.bic(scaled_features)
                log_likelihood = model.score(scaled_features)
                
                # Get state distributions
                states = model.predict(scaled_features)
                state_counts = np.bincount(states)
                min_state_prop = np.min(state_counts) / len(states)
                
                # Calculate transition matrix stability
                transition_matrix = model.transmat_
                eigenvals = np.linalg.eigvals(transition_matrix)
                stability = np.min(np.abs(eigenvals))
                
                # Combined score considering multiple factors
                # - Higher log likelihood is better
                # - Lower AIC/BIC is better
                # - Want reasonable state proportions (not too imbalanced)
                # - Want stable transitions
                combined_score = (log_likelihood 
                                - 0.5 * (aic + bic) / len(scaled_features)
                                + 100 * min_state_prop 
                                + 50 * stability)
                
                scores.append({
                    'n_states': n_states,
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood,
                    'min_state_prop': min_state_prop,
                    'stability': stability,
                    'combined_score': combined_score,
                    'model': model
                })
                
                print(f"\nEvaluating {n_states} states:")
                print(f"AIC: {aic:.2f}")
                print(f"BIC: {bic:.2f}")
                print(f"Log Likelihood: {log_likelihood:.2f}")
                print(f"Min State Proportion: {min_state_prop:.2%}")
                print(f"Transition Stability: {stability:.2f}")
                print(f"Combined Score: {combined_score:.2f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_n_states = n_states
                    self.hmm_model = model
                    
            except Exception as e:
                print(f"Error fitting {n_states} states: {str(e)}")
                continue
        
        if not scores:
            print("Failed to fit any models. Using default 3 states.")
            self.n_states = 3
            self.hmm_model = hmm.GaussianHMM(
                n_components=3,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(scaled_features)
        else:
            self.n_states = best_n_states
            print(f"\nSelected {best_n_states} states with score {best_score:.2f}")
            
            # Print characteristics of each state
            states = self.hmm_model.predict(scaled_features)
            self.data['Market_State'] = states
            
            for state in range(best_n_states):
                state_data = self.data[self.data['Market_State'] == state]
                print(f"\nState {state} characteristics:")
                print(f"Proportion: {len(state_data) / len(self.data):.2%}")
                print(f"Average Return: {state_data['Returns'].mean():.4%}")
                print(f"Volatility: {state_data['Volatility'].mean():.4f}")
                print(f"Average Volume Ratio: {state_data['Volume_Ratio'].mean():.2f}")
                print(f"Call/Put Ratio: {state_data['Call_Put_Ratio'].mean():.2f}")
        
        return self.n_states
    
    def prepare_data(self, sequence_length=60):
        """Prepare data for LSTM model with optimized processing"""
        # Calculate features
        self.calculate_features()
        
        # Find optimal number of states and label the data
        n_states = self.find_optimal_states()
        print(f"Optimal number of market states found: {n_states}")
        
        # Prepare feature matrix
        feature_columns = [
            'Returns', 'Volatility', 'Price_to_SMA20', 
            'SMA20_to_SMA50', 'Volume_Ratio',
            'ATM_Call_Return', 'ATM_Put_Return',
            'Call_Put_Ratio', 'Option_Volume_Ratio'
        ]
        
        # Scale features
        self.features = self.scaler.fit_transform(self.data[feature_columns])
        
        # Create sequences using numpy operations for better performance
        n_samples = len(self.features) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_samples, sequence_length, self.features.shape[1]))
        y = np.zeros(n_samples)
        
        # Fill arrays using vectorized operations
        for i in range(sequence_length, len(self.features)):
            X[i-sequence_length] = self.features[i-sequence_length:i]
            y[i-sequence_length] = self.data['Market_State'].iloc[i]
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def get_state_description(self, state_id):
        """Generate description for each state based on its characteristics"""
        state_data = self.data[self.data['Market_State'] == state_id]
        
        avg_return = state_data['Returns'].mean()
        avg_vol = state_data['Volatility'].mean()
        avg_volume = state_data['Volume_Ratio'].mean()
        avg_call_return = state_data['ATM_Call_Return'].mean()
        avg_put_return = state_data['ATM_Put_Return'].mean()
        avg_cp_ratio = state_data['Call_Put_Ratio'].mean()
        
        # Determine state characteristics
        trend = "Bullish" if avg_return > 0 else "Bearish"
        volatility = "High" if avg_vol > self.data['Volatility'].mean() else "Low"
        volume = "High" if avg_volume > 1 else "Low"
        options_sentiment = "Call-heavy" if avg_cp_ratio > 1 else "Put-heavy"
        options_return = "Calls outperforming" if avg_call_return > avg_put_return else "Puts outperforming"
        
        return (f"State {state_id}: {trend} with {volatility} volatility, {volume} volume, "
                f"{options_sentiment}, {options_return}") 