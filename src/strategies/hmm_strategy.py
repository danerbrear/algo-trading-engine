"""
HMM-Enabled Strategy Base Class

This module provides a base class for strategies that use HMM for market regime classification.
Strategies that need market state filtering should inherit from this class.
"""

from datetime import datetime
import pandas as pd
from src.backtest.models import Strategy


class HMMStrategy(Strategy):
    """
    Base class for strategies that use HMM for market regime classification.
    
    Provides HMM training, prediction, and persistence capabilities.
    Strategies that need market state filtering should inherit from this class.
    
    Example:
        class MyStrategy(HMMStrategy):
            def __init__(self, data_retriever=None, train_hmm=False, **kwargs):
                super().__init__(data_retriever=data_retriever, train_hmm=train_hmm, **kwargs)
                # Strategy-specific initialization
    """
    
    def __init__(
        self,
        data_retriever=None,
        train_hmm: bool = False,
        hmm_training_years: int = 2,
        save_trained_hmm: bool = False,
        hmm_model_dir: str = None,
        **kwargs
    ):
        """
        Initialize HMM-enabled strategy.
        
        Args:
            data_retriever: DataRetriever for fetching historical data
            train_hmm: Whether to train HMM on historical data
            hmm_training_years: Number of years of historical data for training
            save_trained_hmm: Whether to save trained HMM model
            hmm_model_dir: Directory for saving/loading HMM models
            **kwargs: Additional arguments passed to Strategy base class
        """
        super().__init__(**kwargs)
        self.data_retriever = data_retriever
        self.train_hmm = train_hmm
        self.hmm_training_years = hmm_training_years
        self.save_trained_hmm = save_trained_hmm
        self.hmm_model_dir = hmm_model_dir
        self.hmm_model = None  # Trained HMM instance
    
    def set_data(self, data: pd.DataFrame, hmm_model=None, treasury_rates=None):
        """
        Set data for strategy and optionally train HMM.
        
        Overrides base Strategy.set_data() to add HMM training capability.
        
        Args:
            data: Market data DataFrame
            hmm_model: Pre-trained HMM model (optional, unused if train_hmm=True)
            treasury_rates: Treasury rates data (optional)
        """
        self.data = data
        self.treasury_data = treasury_rates
        
        # Get first date from data
        if not data.empty:
            start_date = data.index[0]
            
            # Train HMM if requested (before validation)
            self._train_hmm_if_requested(start_date)
    
    def _train_hmm_if_requested(self, start_date: datetime):
        """
        Train HMM on historical data if training is enabled.
        
        Called during set_data() before data validation.
        
        Args:
            start_date: First date in backtest data
        """
        if not self.train_hmm or not self.data_retriever:
            return
        
        from datetime import timedelta
        from src.model.market_state_classifier import MarketStateClassifier
        
        # Calculate training period: N years before backtest start
        hmm_training_start = start_date - timedelta(days=self.hmm_training_years * 365)
        
        print(f"\n🎓 [{self.__class__.__name__}] Training HMM on {self.hmm_training_years} years of historical data")
        print(f"   Training period: {hmm_training_start.date()} to {start_date.date()}")
        
        # Fetch historical data
        hmm_data = self.data_retriever.fetch_data_for_period(hmm_training_start, 'hmm')
        
        # Filter to only include data before backtest start (avoid look-ahead bias)
        hmm_data = hmm_data[hmm_data.index < start_date]
        
        # Calculate features
        self.data_retriever.calculate_features_for_data(hmm_data)
        
        print(f"   Training on {len(hmm_data)} samples")
        
        # Train HMM model
        self.hmm_model = MarketStateClassifier()
        n_states = self.hmm_model.train_hmm_model(hmm_data)
        
        print(f"   ✅ HMM trained with {n_states} optimal states")
        
        # Apply trained HMM to backtest data
        self._apply_hmm_to_data()
        
        # Save if requested
        if self.save_trained_hmm:
            self._save_hmm_model()
    
    def _apply_hmm_to_data(self):
        """Apply trained HMM to predict market states for backtest data."""
        if self.data is None or self.hmm_model is None:
            return
        
        # Get features needed for HMM
        required_features = ['Returns', 'Volatility', 'Volume_Change']
        
        # Check if features exist
        missing_features = [f for f in required_features if f not in self.data.columns]
        if missing_features:
            print(f"   ⚠️  Warning: Missing features {missing_features}, cannot apply HMM")
            return
        
        features = self.hmm_model.scaler.transform(self.data[required_features])
        
        # Predict market states
        market_states = self.hmm_model.hmm_model.predict(features)
        
        # Add to data
        self.data['Market_State'] = market_states
        
        print(f"   ✅ Applied HMM predictions to {len(self.data)} backtest days")
    
    def _save_hmm_model(self):
        """Save trained HMM model for future use."""
        if self.hmm_model is None:
            return
        
        import pickle
        import os
        from datetime import datetime as dt
        
        base_dir = self.hmm_model_dir or os.environ.get('MODEL_SAVE_BASE_PATH', 'Trained_Models')
        
        # Use strategy class name for mode (e.g., 'upwardtrendreversal_hmm')
        strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
        mode = f'{strategy_name}_hmm'
        
        # Get symbol from data_retriever or default to SPY
        symbol = getattr(self.data_retriever, 'symbol', 'SPY')
        
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        timestamp_dir = os.path.join(base_dir, mode, symbol, timestamp)
        latest_dir = os.path.join(base_dir, mode, symbol, 'latest')
        os.makedirs(timestamp_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        
        # Save HMM model and scaler with metadata
        hmm_data = {
            'hmm_model': self.hmm_model.hmm_model,
            'scaler': self.hmm_model.scaler,
            'n_states': self.hmm_model.n_states,
            'max_states': self.hmm_model.max_states,
            'training_years': self.hmm_training_years,
            'strategy': self.__class__.__name__,
            'timestamp': timestamp
        }
        
        hmm_path = os.path.join(timestamp_dir, 'hmm_model.pkl')
        hmm_latest_path = os.path.join(latest_dir, 'hmm_model.pkl')
        
        with open(hmm_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        with open(hmm_latest_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        print(f"   ✅ HMM model saved to {latest_dir}")

