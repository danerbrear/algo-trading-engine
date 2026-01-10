import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.base import BaseEstimator, ClassifierMixin

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

class MarketStateClassifier:
    """Class for identifying market states using HMM"""
    def __init__(self, max_states=5):
        self.max_states = max_states
        self.hmm_model = None
        self.n_states = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_hmm_model(self, data):
        """Train the HMM model and find optimal number of states"""
        print(f"\n🎯 Training HMM model on market data ({len(data)} samples)")
        optimal_states = self.find_optimal_states(data)
        self.is_trained = True
        print(f"✅ HMM model trained with {optimal_states} optimal states")
        return optimal_states
        
    def find_optimal_states(self, data):
        """Find optimal number of states using HMM with multiple criteria"""
        # Prepare features for HMM - using key market indicators
        # Note: Price_to_SMA20 used in HMM but excluded from LSTM to avoid correlation
        feature_matrix = np.column_stack([
            data['Returns'],
            data['Volatility'],
            data['Price_to_SMA20'],
            data['SMA20_to_SMA50'],
            data['Volume_Ratio']
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Try different numbers of states and evaluate using multiple criteria
        best_score = float('-inf')
        best_n_states = 3  # Default
        scores = []
        
        for n_states in range(2, self.max_states + 1):
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
            state_characteristics = self._get_state_characteristics(data, states)
            
            for state in range(best_n_states):
                print(f"\nState {state} characteristics:")
                chars = state_characteristics[state]
                print(f"Proportion: {chars['proportion']:.2%}")
                print(f"Average Return: {chars['avg_return']:.4%}")
                print(f"Volatility: {chars['volatility']:.4f}")
                print(f"Price/SMA20 Ratio: {chars['price_sma20']:.4f}")
                print(f"SMA20/SMA50 Ratio: {chars['sma_trend']:.4f}")
                print(f"Volume Ratio: {chars['volume_ratio']:.2f}")
        
        return self.n_states
    
    def predict_states(self, data):
        """Predict market states for given data"""
        if self.hmm_model is None or not self.is_trained:
            raise ValueError("Model not trained. Call train_hmm_model first.")
            
        # Note: Price_to_SMA20 used in HMM but excluded from LSTM to avoid correlation
        feature_matrix = np.column_stack([
            data['Returns'],
            data['Volatility'],
            data['Price_to_SMA20'],
            data['SMA20_to_SMA50'],
            data['Volume_Ratio']
        ])
        
        scaled_features = self.scaler.transform(feature_matrix)
        return self.hmm_model.predict(scaled_features)
    
    def _get_state_characteristics(self, data, states):
        """Calculate characteristics for each state"""
        state_chars = {}
        for state in range(self.n_states):
            state_data = data[states == state]
            state_chars[state] = {
                'proportion': len(state_data) / len(data),
                'avg_return': state_data['Returns'].mean(),
                'volatility': state_data['Volatility'].mean(),
                'price_sma20': state_data['Price_to_SMA20'].mean(),
                'sma_trend': state_data['SMA20_to_SMA50'].mean(),
                'volume_ratio': state_data['Volume_Ratio'].mean()
            }
        return state_chars
    
    def get_state_description(self, state_id, data, states):
        """Generate description for each state based on its characteristics"""
        state_data = data[states == state_id]
        
        avg_return = state_data['Returns'].mean()
        avg_vol = state_data['Volatility'].mean()
        avg_volume = state_data['Volume_Ratio'].mean()
        avg_price_sma = state_data['Price_to_SMA20'].mean()
        avg_sma_trend = state_data['SMA20_to_SMA50'].mean()
        
        # Determine state characteristics
        trend = "Bullish" if avg_return > 0 else "Bearish"
        volatility = "High" if avg_vol > data['Volatility'].mean() else "Low"
        volume = "High" if avg_volume > 1 else "Low"
        price_trend = "Above MA" if avg_price_sma > 1 else "Below MA"
        ma_trend = "Upward" if avg_sma_trend > 1 else "Downward"
        
        return (f"State {state_id}: {trend} with {volatility} volatility, {volume} volume, "
                f"Price {price_trend}, Moving Averages trending {ma_trend}") 