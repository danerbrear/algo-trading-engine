import os
import pandas as pd
from datetime import datetime
import pickle
from .models import Strategy, CreditSpreadStrategy, Position
from src.common.data_retriever import DataRetriever
from src.model.market_state_classifier import MarketStateClassifier

class BacktestEngine:
    """
    BacktestEngine is a class that runs a backtest on a given dataset and model.
    """

    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000,
                 start_date: datetime = datetime.now(),
                 end_date: datetime = datetime.now()):
        self.data = data
        self.strategy = strategy
        self.capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.positions = []

    def run(self):
        """
        Run the backtest.
        """
        
        try:
            self._validate_data(self.data)
        except Exception as e:
            print(f"Error validating data: {e}")

        # Generate date range (business days only)
        date_range = pd.bdate_range(start=self.start_date, end=self.end_date)
        
        # For each date in the range, simulate the strategy
        for date in date_range:
            # Convert to tuple for immutability
            positions_tuple = tuple(self.positions)
            self.strategy.on_new_date(positions_tuple, self._add_position, self._remove_position)
        
        print(f"Final capital: {self.capital}")

    def _determine_return(self):
        """
        Determine the returns of an executed options strategy.
        """

    def _validate_data(self, data: pd.DataFrame):
        """
        Validate the data given the scaler.
        """

    def _add_position(self, position: Position):
        """
        Add a position to the positions list.
        """

        if self.capital < position.entry_price * position.quantity:
            raise ValueError("Not enough capital to add position")
        
        self.capital -= position.entry_price * position.quantity
        self.positions.append(position)

    def _remove_position(self, position: Position):
        """
        Remove a position from the positions list.
        """
        self.capital += position.exit_price * position.quantity
        self.positions.remove(position)

if __name__ == "__main__":
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2025, 7, 1)

    data_retriever = DataRetriever(symbol='SPY', hmm_start_date='2010-01-01', lstm_start_date='2021-06-01', use_free_tier=False, quiet_mode=True)

    # Load model directory from environment variable
    model_save_base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    model_dir = os.path.join(model_save_base_path, 'lstm_poc', 'SPY', 'latest')

    # Load HMM model
    hmm_path = os.path.join(model_dir, 'hmm_model.pkl')
    if not os.path.exists(hmm_path):
        raise FileNotFoundError(f"HMM model not found at {hmm_path}")
    
    with open(hmm_path, 'rb') as f:
        hmm_data = pickle.load(f)
    
    hmm_model = MarketStateClassifier(max_states=hmm_data['max_states'])
    hmm_model.hmm_model = hmm_data['hmm_model']
    hmm_model.scaler = hmm_data['scaler']
    hmm_model.n_states = hmm_data['n_states']
    hmm_model.is_trained = True
    
    print(f"✅ HMM model loaded from {hmm_path}")
    print(f"   Number of states: {hmm_model.n_states}")
    
    # Then prepare the data for LSTM
    data = data_retriever.prepare_data_for_lstm(state_classifier=hmm_model)

    backtester = BacktestEngine(
        data=data, 
        strategy=CreditSpreadStrategy(),
        initial_capital=10000,
        start_date=start_date,
        end_date=end_date
    )
    backtester.run()
