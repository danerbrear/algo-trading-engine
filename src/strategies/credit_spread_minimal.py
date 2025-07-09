from datetime import datetime
from typing import Callable
import numpy as np
import pandas as pd

from src.backtest.models import Position, Strategy, OptionType


class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    holding_period = 10

    def __init__(self, lstm_model, lstm_scaler):
        super().__init__(stop_loss=0.6)
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler
        self.symbol = 'SPY'  # Default symbol, can be overridden

        self.error_count = 0

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, determine if a new position should be opened. We should not open a position if we already have one.
        """

        if len(positions) == 0:
            # Determine if we should open a new position
            print(f"No positions, opening new position")
            prediction = self._make_prediction(date)
            if prediction is not None:
                if prediction['strategy'] == 1:
                    stock_price = self.data.loc[date]['Close']
                    # Call Credit Spread - sell ATM call, buy OTM call
                    expiration_date = (date + pd.Timedelta(days=30)).to_pydatetime()  # 30 days to expiration
                    atm_strike = round(stock_price)  # ATM strike
                    # Estimate option premium (rough approximation for ATM call)
                    option_premium = max(0.5, stock_price * 0.02)  # At least $0.50, or 2% of stock price
                    position = Position(
                        symbol=self.symbol, 
                        quantity=1, 
                        expiration_date=expiration_date,
                        option_type=OptionType.CALL.value,
                        strike_price=atm_strike,
                        entry_date=date, 
                        entry_price=option_premium
                    )
                    print(f"Adding position: {position.__str__()}")
                    add_position(position)
                elif prediction['strategy'] == 2:
                    stock_price = self.data.loc[date]['Close']
                    # Put Credit Spread - sell ATM put, buy OTM put
                    expiration_date = (date + pd.Timedelta(days=30)).to_pydatetime()  # 30 days to expiration
                    atm_strike = round(stock_price)  # ATM strike
                    # Estimate option premium (rough approximation for ATM put)
                    option_premium = max(0.5, stock_price * 0.02)  # At least $0.50, or 2% of stock price
                    position = Position(
                        symbol=self.symbol, 
                        quantity=1, 
                        expiration_date=expiration_date,
                        option_type=OptionType.PUT.value,
                        strike_price=atm_strike,
                        entry_date=date, 
                        entry_price=option_premium
                    )
                    print(f"Adding position: {position.__str__()}")
                    add_position(position)
            else:
                self.error_count += 1
            
        else:
            for position in positions:
                # Determine if we should close a position
                if (self._profit_target_hit(position, self.data.loc[date]['Close']) or self._stop_loss_hit(position, self.data.loc[date]['Close'])):
                    print(f"Profit target or stop loss hit for {position.__str__()}")
                    remove_position(position, self.data.loc[date]['Close'])
                elif position.get_days_to_expiration(date) < self.holding_period:
                    print(f"Position {position.__str__()} expired or near expiration")
                    remove_position(position, self.data.loc[date]['Close'])

    def on_end(self, positions: tuple['Position', ...]):
        """
        On end, execute strategy.
        """
        print(f"Total error count: {self.error_count}")
    
    def _make_prediction(self, date: datetime):
        """
        Make a prediction from the LSTM model for a given date.
        
        Args:
            date: The date to make a prediction for
            
        Returns:
            dict: Dictionary containing:
                - strategy: int (0=Hold, 1=Call Credit Spread, 2=Put Credit Spread)
                - confidence: float (probability of the predicted strategy)
                - probabilities: dict (probabilities for all strategies)
        """
        if self.data is None:
            raise ValueError("Data not set for strategy. Call set_data() first.")
        
        if self.lstm_model is None:
            raise ValueError("LSTM model not set for strategy.")
        
        # Ensure the date is a pandas Timestamp and matches the index type
        date_key = pd.Timestamp(date)
        
        # First try exact match
        if date_key in self.data.index:
            date_idx = self.data.index.get_loc(date_key)
        else:
            # Try to normalize both index and date (remove time component)
            date_key = pd.Timestamp(date).normalize()
            index_normalized = self.data.index.normalize()
            
            if date_key in index_normalized:
                date_idx = index_normalized.get_loc(date_key)
            else:
                # Provide detailed error information
                print(f"❌ Date {date} not found in data index.")
                print(f"   Available date range: {self.data.index.min()} to {self.data.index.max()}")
                print(f"   Total data points: {len(self.data)}")
                print(f"   Normalized date: {date_key}")
                print(f"   First 5 available dates: {list(self.data.index[:5])}")
                print(f"   Last 5 available dates: {list(self.data.index[-5:])}")
                
                # Check if it's a weekend or holiday
                if date_key.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    print(f"   Note: {date_key.date()} is a weekend (market closed)")
                else:
                    print(f"   Note: {date_key.date()} might be a market holiday or missing data")
                
                return None

        print(f"Date index: {date_idx}") # TODO: Remove this
        
        # Check if we have enough data for the sequence
        sequence_length = self.lstm_model.sequence_length
        if date_idx < sequence_length:
            print(f"Warning: Not enough data for prediction at {date}. Need at least {sequence_length} days of history.")
            return {
                'strategy': 0,  # Default to Hold
                'confidence': 0.0,
                'probabilities': {0: 1.0, 1: 0.0, 2: 0.0}
            }
        
        # Get the feature columns that the LSTM model expects
        feature_columns = self.lstm_model.feature_columns
        if feature_columns is None:
            # Fallback to the original features if not set
            feature_columns = [
                'High_Low_Range',
                'SMA20_to_SMA50', 'SMA20_to_SMA50_Lag1', 'SMA20_to_SMA50_Lag5', 
                'SMA20_to_SMA50_MA5', 'SMA20_to_SMA50_MA10', 'SMA20_to_SMA50_Std5', 'SMA20_to_SMA50_Momentum',
                'RSI', 'RSI_Lag1', 'RSI_Lag5', 'RSI_MA5', 'RSI_MA10', 'RSI_Std5', 'RSI_Momentum', 
                'RSI_Overbought', 'RSI_Oversold',
                'MACD_Hist',
                'Volume_Ratio', 'OBV',
                'Put_Call_Ratio', 'Option_Volume_Ratio',
                'Market_State',
                'Days_Until_Next_CPI', 'Days_Since_Last_CPI',
                'Days_Until_Next_CC', 'Days_Since_Last_CC',
                'Days_Until_Next_FFR', 'Days_Since_Last_FFR'
            ]
        
        # Extract the sequence of features for the prediction
        start_idx = date_idx - sequence_length + 1
        end_idx = date_idx + 1
        
        # Get the feature data for the sequence
        sequence_data = self.data.iloc[start_idx:end_idx][feature_columns].copy()
        
        # Check if all required features are available in the sequence data
        missing_features = [col for col in feature_columns if col not in sequence_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features for LSTM prediction: {missing_features}. Available columns: {list(sequence_data.columns)}")
        
        # Check for missing values
        if sequence_data.isnull().any().any():
            print(f"Warning: Missing values in feature data for prediction at {date}")
            # Fill missing values with forward fill, then backward fill
            sequence_data = sequence_data.fillna(method='ffill').fillna(method='bfill')
            
            # If still have missing values, fill with 0
            sequence_data = sequence_data.fillna(0)
        
        # Scale the features using the model's scaler
        if self.lstm_scaler is None:
            raise ValueError("LSTM scaler not set for strategy.")
        
        scaled_features = self.lstm_scaler.transform(sequence_data)
        
        # Reshape for LSTM input: (1, sequence_length, n_features)
        X = scaled_features.reshape(1, sequence_length, len(feature_columns))
        
        # Make prediction
        try:
            prediction = self.lstm_model.predict(X)[0]
            probabilities = self.lstm_model.predict_proba(X)[0]
            
            # Strategy labels
            strategy_labels = ['Hold', 'Call Credit Spread', 'Put Credit Spread']
            predicted_strategy = strategy_labels[prediction]
            confidence = probabilities[prediction]
            
            print(f"LSTM Prediction for {date}: {predicted_strategy} (confidence: {confidence:.2%})")
            
            return {
                'strategy': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    0: float(probabilities[0]),  # Hold
                    1: float(probabilities[1]),  # Call Credit Spread
                    2: float(probabilities[2])   # Put Credit Spread
                }
            }
            
        except Exception as e:
            print(f"Error making LSTM prediction: {e}")
            raise ValueError(f"Error making LSTM prediction: {e}")
