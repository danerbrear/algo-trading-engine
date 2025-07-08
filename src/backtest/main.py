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

    def run(self) -> bool:
        """
        Run the backtest.
        """
        
        # Validate the data first
        if not self._validate_data(self.data):
            print("❌ Backtest aborted due to invalid data")
            return False

        # Generate date range (business days only)
        date_range = pd.bdate_range(start=self.start_date, end=self.end_date)
        
        # For each date in the range, simulate the strategy
        for date in date_range:
            # Convert to tuple for immutability
            positions_tuple = tuple(self.positions)
            self.strategy.on_new_date(positions_tuple, self._add_position, self._remove_position)
        
        print(f"Final capital: {self.capital}")
        return True

    def _determine_return(self):
        """
        Determine the returns of an executed options strategy.
        """

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data and filter it to the specified date range.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        print(f"🔍 Validating data for backtest...")
        print(f"   Original data shape: {data.shape}")
        print(f"   Date range: {self.start_date} to {self.end_date}")
        
        # Check if the data has the required columns
        required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # Basic OHLCV data
            'Returns', 'Log_Returns', 'Volatility',     # Basic technical features
            'RSI', 'MACD_Hist', 'Volume_Ratio',         # Technical indicators
            'Market_State',                             # HMM market state
            'Put_Call_Ratio', 'Option_Volume_Ratio',    # Options features
            'Days_Until_Next_CPI', 'Days_Since_Last_CPI',  # Calendar features
            'Days_Until_Next_CC', 'Days_Since_Last_CC',
            'Days_Until_Next_FFR', 'Days_Since_Last_FFR'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"⚠️  Warning: Missing columns: {missing_columns}")
            print(f"   Available columns: {list(data.columns)}")
            return False
        else:
            print(f"✅ All required columns present")
        
        # Check if data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            print("❌ Error: Data must have a datetime index for backtesting")
            return False
        
        # Filter data to the specified date range
        print(f"   Data index range: {data.index.min()} to {data.index.max()}")
        
        # Convert start_date and end_date to datetime if they're not already
        if isinstance(self.start_date, datetime):
            start_date = self.start_date
        else:
            start_date = pd.to_datetime(self.start_date)
            
        if isinstance(self.end_date, datetime):
            end_date = self.end_date
        else:
            end_date = pd.to_datetime(self.end_date)
        
        # Filter data to the specified date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data[mask].copy()
        
        if len(filtered_data) == 0:
            print(f"❌ Error: No data available for the specified date range: {start_date} to {end_date}. "
                  f"Available data range: {data.index.min()} to {data.index.max()}")
            return False
        
        # Update the data attribute
        self.data = filtered_data
        
        print(f"✅ Data validation complete:")
        print(f"   Final data shape: {self.data.shape}")
        print(f"   Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"   Business days: {len(self.data)}")
        
        return True

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
