import pandas as pd
from datetime import datetime
from .models import Strategy, CreditSpreadStrategy, Position
from src.common.data_retriever import DataRetriever

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
    data = data_retriever.prepare_data_for_lstm()

    backtester = BacktestEngine(
        data=data, 
        strategy=CreditSpreadStrategy(),
        initial_capital=10000,
        start_date=start_date,
        end_date=end_date
    )
    backtester.run()
