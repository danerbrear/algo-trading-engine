from datetime import datetime
from typing import Callable
import numpy as np
import pandas as pd

from src.backtest.models import Position, Strategy, OptionType, StrategyType
from src.common.models import Option, OptionChain


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
        self.symbol = 'SPY'  # Default symbol

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
                    # Call Credit Spread using real options data
                    position = self._create_call_credit_spread_from_chain(date)
                    if position:
                        print(f"Adding position: {position.__str__()}")
                        add_position(position)
                elif prediction['strategy'] == 2:
                    # Put Credit Spread using real options data
                    position = self._create_put_credit_spread_from_chain(date)
                    if position:
                        print(f"Adding position: {position.__str__()}")
                        add_position(position)
            else:
                self.error_count += 1
            
        else:
            for position in positions:
                # Calculate exit price for the position
                date_key = date.strftime('%Y-%m-%d')
                if date_key in self.options_data:
                    exit_price = position.calculate_exit_price(self.options_data[date_key])
                else:
                    exit_price = None
                
                if exit_price is None:
                    print(f"Error calculating exit price for {position.__str__()}")
                    self.error_count += 1
                    continue
                
                # Determine if we should close a position
                if (self._profit_target_hit(position, exit_price) or self._stop_loss_hit(position, exit_price)):
                    print(f"Profit target or stop loss hit for {position.__str__()}")
                    remove_position(position, exit_price)
                elif position.get_days_to_expiration(date) < self.holding_period:
                    print(f"Position {position.__str__()} expired or near expiration")
                    remove_position(position, exit_price)

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

    def _create_call_credit_spread_from_chain(self, date: datetime) -> Position:
        """Create a call credit spread using the options chain data"""
        if not self.options_data:
            print("⚠️  No options data available")
            return None
            
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            print(f"⚠️  No options data for {date_key}")
            return None
            
        option_chain = self.options_data[date_key]
        current_price = self.data.loc[date]['Close']
        
        # Find ATM call option
        atm_call = self._find_atm_option(option_chain.calls, current_price)
        if not atm_call:
            print(f"⚠️  No ATM call option found for price ${current_price:.2f}")
            return None
            
        # Find OTM call option (higher strike)
        otm_call = self._find_otm_call(option_chain.calls, current_price, atm_call.strike)
        if not otm_call:
            print(f"⚠️  No OTM call option found for call credit spread")
            return None
            
        # Calculate net credit (sell ATM, buy OTM)
        net_credit = atm_call.last_price - otm_call.last_price
        
        if net_credit <= 0:
            print(f"⚠️  Invalid net credit for call spread: ${net_credit:.2f}")
            return None
            
        print(f"📊 Call Credit Spread:")
        print(f"   Sell ATM Call: ${atm_call.strike:.0f} @ ${atm_call.last_price:.2f}")
        print(f"   Buy OTM Call: ${otm_call.strike:.0f} @ ${otm_call.last_price:.2f}")
        print(f"   Net Credit: ${net_credit:.2f}")
        
        # Create position using the ATM call as the primary option
        position = Position(
            symbol=self.symbol,
            quantity=1,
            expiration_date=datetime.strptime(atm_call.expiration, '%Y-%m-%d'),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=atm_call.strike,
            entry_date=date,
            entry_price=net_credit,  # Use net credit as entry price
            spread_options=[atm_call, otm_call]  # Store the specific spread options
        )
        
        return position

    def _create_put_credit_spread_from_chain(self, date: datetime) -> Position:
        """Create a put credit spread using the options chain data"""
        if not self.options_data:
            print("⚠️  No options data available")
            return None
            
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            print(f"⚠️  No options data for {date_key}")
            return None
            
        option_chain = self.options_data[date_key]
        current_price = self.data.loc[date]['Close']
        
        # Find ATM put option
        atm_put = self._find_atm_option(option_chain.puts, current_price)
        if not atm_put:
            print(f"⚠️  No ATM put option found for price ${current_price:.2f}")
            return None
            
        # Find OTM put option (lower strike)
        otm_put = self._find_otm_put(option_chain.puts, current_price, atm_put.strike)
        if not otm_put:
            print(f"⚠️  No OTM put option found for put credit spread")
            return None
            
        # Calculate net credit (sell ATM, buy OTM)
        net_credit = atm_put.last_price - otm_put.last_price
        
        if net_credit <= 0:
            print(f"⚠️  Invalid net credit for put spread: ${net_credit:.2f}")
            return None
            
        print(f"📊 Put Credit Spread:")
        print(f"   Sell ATM Put: ${atm_put.strike:.0f} @ ${atm_put.last_price:.2f}")
        print(f"   Buy OTM Put: ${otm_put.strike:.0f} @ ${otm_put.last_price:.2f}")
        print(f"   Net Credit: ${net_credit:.2f}")
        
        # Create position using the ATM put as the primary option
        position = Position(
            symbol=self.symbol,
            quantity=1,
            expiration_date=datetime.strptime(atm_put.expiration, '%Y-%m-%d'),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=atm_put.strike,
            entry_date=date,
            entry_price=net_credit,  # Use net credit as entry price
            spread_options=[atm_put, otm_put]  # Store the specific spread options
        )
        
        return position

    def _find_atm_option(self, options: list, current_price: float):
        """Find the ATM option closest to the current price"""
        if not options:
            return None
            
        # Find option closest to current price
        atm_option = min(options, key=lambda opt: abs(opt.strike - current_price))
        
        # Check if it's reasonably close to ATM (within 5% of current price)
        if abs(atm_option.strike - current_price) / current_price <= 0.05:
            return atm_option
            
        return None

    def _find_otm_call(self, calls: list, current_price: float, atm_strike: float):
        """Find an OTM call option (higher strike than ATM)"""
        # Find calls with higher strikes than ATM
        otm_calls = [call for call in calls if call.strike > atm_strike]
        
        if not otm_calls:
            return None
            
        # Return the closest OTM call (lowest strike above ATM)
        return min(otm_calls, key=lambda opt: opt.strike)

    def _find_otm_put(self, puts: list, current_price: float, atm_strike: float):
        """Find an OTM put option (lower strike than ATM)"""
        # Find puts with lower strikes than ATM
        otm_puts = [put for put in puts if put.strike < atm_strike]
        
        if not otm_puts:
            return None
            
        # Return the closest OTM put (highest strike below ATM)
        return max(otm_puts, key=lambda opt: opt.strike)


