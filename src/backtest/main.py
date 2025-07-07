import pandas as pd
from datetime import datetime

class BacktestEngine:
    """
    BacktestEngine is a class that runs a backtest on a given dataset and model.
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000, model = None, scaler = None,
                 start_date: str = datetime.now().strftime("%Y-%m-%d"),
                 end_date: str = datetime.now().strftime("%Y-%m-%d")):
        self.data = data
        self.initial_capital = initial_capital
        self.model = model
        self.scaler = scaler
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        """
        Run the backtest.
        """
        pass

    def _determine_return(self):
        """
        Determine the returns of an executed options strategy.
        """
        pass

if __name__ == "__main__":
    backtester = BacktestEngine()
    backtester.run()
