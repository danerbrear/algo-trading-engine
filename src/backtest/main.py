import os
import pandas as pd
from datetime import datetime

from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from .models import Strategy, Position
from src.common.data_retriever import DataRetriever
from src.common.functions import load_hmm_model, load_lstm_model

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
        self.initial_capital = initial_capital  # Store initial capital for reporting
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

        # Use only dates that exist in the data (not pd.bdate_range which includes holidays)
        # Filter data to the specified date range and use the actual dates
        date_range = self.data.index
        
        print(f"📅 Running backtest on {len(date_range)} trading days")
        print(f"   Date range: {date_range[0].date()} to {date_range[-1].date()}")

        # For each date in the range, simulate the strategy
        for date in date_range:
            # Convert to tuple for immutability
            positions_tuple = tuple(self.positions)

            try:
                self.strategy.on_new_date(date, positions_tuple, self._add_position, self._remove_position)
            except Exception as e:
                print(f"Error in on_new_date: {e}")
                return False

        self._end()

        # Calculate final performance metrics
        initial_capital = self.initial_capital  # Use the initial capital from the constructor
        final_return = self.capital - initial_capital
        final_return_pct = (final_return / initial_capital) * 100
        
        print(f"\n📊 Backtest Results Summary:")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Final Capital: ${self.capital:,.2f}")
        print(f"   Total Return: ${final_return:+,.2f} ({final_return_pct:+.2f}%)")
        print(f"   Trading Days: {len(date_range)}")
        
        return True

    def _end(self):
        """
        On end, execute strategy and close any remaining positions.
        """
        print(f"\n🏁 Closing backtest - {len(self.positions)} positions remaining")
        
        # Get the last available price from the data
        last_date = self.data.index[-1]
        last_price = self.data.loc[last_date, 'Close']
        
        print(f"   Last trading date: {last_date.date()}")
        print(f"   Last closing price: ${last_price:.2f}")
        
        # Execute strategy's on_end method
        self.strategy.on_end(self.positions)
        
        # Close all remaining positions with the last available price
        total_pnl = 0
        for position in self.positions[:]:  # Create a copy to avoid modification during iteration
            try:
                # Calculate the return for this position
                position_return = position.get_return_dollars(last_price)
                total_pnl += position_return
                
                print(f"   Closing position: {position}")
                print(f"     Exit price: ${last_price:.2f}")
                print(f"     Return: ${position_return:+.2f}")
                
                # Remove the position and update capital
                self._remove_position(position, last_price)
                
            except Exception as e:
                print(f"   Error closing position {position}: {e}")
        
        print(f"   Total P&L from closing positions: ${total_pnl:+.2f}")
        print(f"   Final capital: ${self.capital:.2f}")

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data and filter it to the specified date range.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        print(f"\n🔍 Validating data for backtest...")
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
            print(f"❌ Error: No data available for the specified date range: {start_date} to {end_date}. ")
            print(f"   Available data range: {data.index.min()} to {data.index.max()}")
            print(f"   Requested start date: {start_date}")
            print(f"   Requested end date: {end_date}")
            print(f"   Total available data points: {len(data)}")
            
            # Check if the issue is with the date range
            if start_date > data.index.max():
                print(f"   ⚠️  Start date {start_date} is after the latest available data {data.index.max()}")
            if end_date < data.index.min():
                print(f"   ⚠️  End date {end_date} is before the earliest available data {data.index.min()}")
            
            return False
        
        # Update the data attribute
        self.data = filtered_data
        
        print(f"✅ Data validation complete:")
        print(f"   Final data shape: {self.data.shape}")
        print(f"   Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"   Trading days: {len(self.data)}")
        
        # Check for gaps in the data (missing trading days)
        expected_business_days = len(pd.bdate_range(start=self.data.index.min(), end=self.data.index.max()))
        actual_trading_days = len(self.data)
        if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
            print(f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}")
        
        return True

    def _add_position(self, position: Position):
        """
        Add a position to the positions list.
        """

        if self.capital < position.entry_price * position.quantity * 100:
            raise ValueError("Not enough capital to add position")
        
        self.capital -= position.entry_price * position.quantity * 100
        self.positions.append(position)

    def _remove_position(self, position: Position, exit_price: float):
        """
        Remove a position from the positions list and update capital.
        
        Args:
            position: Position to remove
            exit_price: Price at which the position is being closed
        """
        if position not in self.positions:
            print(f"⚠️  Warning: Position {position} not found in positions list")
            return
            
        # Calculate the return for this position
        position_return = position.get_return_dollars(exit_price)
        
        # Update capital
        self.capital += position_return
        
        # Remove the position
        self.positions.remove(position)
        
        # Log the position closure
        print(f"   Position closed: {position.__str__()}")
        print(f"     Entry: ${position.entry_price:.2f} | Exit: ${exit_price:.2f}")
        print(f"     Return: ${position_return:+.2f} | Capital: ${self.capital:.2f}")

if __name__ == "__main__":
    # Test with a smaller date range to verify the fix
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    print("🧪 Testing backtest with fixed date handling...")
    
    data_retriever = DataRetriever(symbol='SPY', hmm_start_date=start_date, lstm_start_date=start_date, use_free_tier=False, quiet_mode=True)

    # Load model directory from environment variable
    model_save_base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    model_dir = os.path.join(model_save_base_path, 'lstm_poc', 'SPY', 'latest')

    try:
        hmm_model = load_hmm_model(model_dir)
        lstm_model, scaler = load_lstm_model(model_dir, return_lstm_instance=True)

        # Then prepare the data for LSTM
        data, options_data = data_retriever.prepare_data_for_lstm(state_classifier=hmm_model)

        strategy = CreditSpreadStrategy(
            lstm_model=lstm_model, 
            lstm_scaler=scaler
        )
        strategy.set_data(data, options_data)

        backtester = BacktestEngine(
            data=data, 
            strategy=strategy,
            initial_capital=10000,
            start_date=start_date,
            end_date=end_date
        )
        
        success = backtester.run()
        if success:
            print("✅ Backtest completed successfully!")
        else:
            print("❌ Backtest failed!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
