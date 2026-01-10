"""
Common utility functions used across the project.
"""

import os
import pickle
from algo_trading_engine.model.market_state_classifier import MarketStateClassifier

def load_hmm_model(model_dir):
    """
    Load a trained HMM model from the given directory.
    
    Args:
        model_dir: Directory where the HMM model is saved
        
    Returns:
        MarketStateClassifier: Loaded and configured HMM model
        
    Raises:
        FileNotFoundError: If the HMM model file is not found
        Exception: If there's an error loading the model
    """
    hmm_path = os.path.join(model_dir, 'hmm_model.pkl')
    if not os.path.exists(hmm_path):
        raise FileNotFoundError(f"HMM model not found at {hmm_path}")
    try:
        with open(hmm_path, 'rb') as f:
            hmm_data = pickle.load(f)
        hmm_model = MarketStateClassifier(max_states=hmm_data['max_states'])
        hmm_model.hmm_model = hmm_data['hmm_model']
        hmm_model.scaler = hmm_data['scaler']
        hmm_model.n_states = hmm_data['n_states']
        hmm_model.is_trained = True
        print(f"✅ HMM model loaded from {hmm_path}")
        print(f"   Number of states: {hmm_model.n_states}")
        return hmm_model
    except Exception as e:
        raise Exception(f"Error loading HMM model from {hmm_path}: {str(e)}")

def load_lstm_model(model_dir, return_lstm_instance=False):
    """
    Load a trained LSTM model from the given directory.
    
    Args:
        model_dir: Directory where the LSTM model is saved
        return_lstm_instance: If True, return LSTMModel instance instead of keras model
        
    Returns:
        tuple: (model, scaler) - Loaded LSTM model and scaler
        
    Raises:
        FileNotFoundError: If the LSTM model file is not found
        Exception: If there's an error loading the model
    """
    lstm_path = os.path.join(model_dir, 'model.keras')
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
    try:
        import keras
        keras_model = keras.models.load_model(lstm_path)
        print(f"LSTM model loaded from {lstm_path}")
        scaler = None
        scaler_path = os.path.join(model_dir, 'lstm_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"LSTM scaler loaded from {scaler_path}")
        else:
            print(f"WARNING: LSTM scaler not found at {scaler_path}")
        if return_lstm_instance:
            try:
                from algo_trading_engine.model.lstm_model import LSTMModel
                lstm_instance = LSTMModel(sequence_length=60, n_features=29)  # Default values
                lstm_instance.model = keras_model
                return lstm_instance, scaler
            except ImportError:
                from algo_trading_engine.model.lstm_model import LSTMModel
                lstm_instance = LSTMModel(sequence_length=60, n_features=29)
                lstm_instance.model = keras_model
                return lstm_instance, scaler
        return keras_model, scaler
    except Exception as e:
        raise Exception(f"Error loading LSTM model from {lstm_path}: {str(e)}")

def get_model_directory(symbol='SPY', mode='lstm_poc', base_path=None):
    """
    Get the model directory path for a given symbol and mode.
    
    Args:
        symbol: Stock symbol (default: 'SPY')
        mode: Model mode/label (default: 'lstm_poc')
        base_path: Optional base path for models
        
    Returns:
        str: Path to the model directory
    """
    if base_path is None:
        base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    return os.path.join(base_path, mode, symbol, 'latest') 