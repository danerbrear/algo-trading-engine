from datetime import datetime
from typing import Callable
import pandas as pd

from src.backtest.models import Position, Strategy, StrategyType
from src.common.models import Option, OptionType, OptionChain
from src.model.options_handler import OptionsHandler


class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    holding_period = 15

    def __init__(self, lstm_model, lstm_scaler, options_handler: OptionsHandler = None, start_date_offset: int = 0):
        super().__init__(stop_loss=0.6, profit_target=0.8, start_date_offset=start_date_offset)
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler
        self.options_handler = options_handler

        self.error_count = 0

    def on_new_date(self, date: datetime, positions: tuple['Position', ...],
                    add_position: Callable[['Position'], None], 
                    remove_position: Callable[['Position'], None]):
        """
        On new date, determine if a new position should be opened. 
        We should not open a position if we already have one.
        """
        if date.date() < self.data.index[self.start_date_offset].date():
            return
        
        super().on_new_date(date, positions, add_position, remove_position)

        if len(positions) == 0:
            # Determine if we should open a new position
            print("No positions, opening new position")
            prediction = self._make_prediction(date)
            if prediction is not None:
                if prediction['strategy'] == 1:
                    try:
                        # Call Credit Spread using real options data
                        position = self._create_call_credit_spread_from_chain(date, prediction)
                        if position:
                            print(f"Adding position: {position.__str__()}")
                            current_price = self.data.loc[date]['Close']
                            print(f"    Current price: {round(current_price, 2)}")
                            add_position(position)
                    except Exception as e:
                        print(f"Error creating call credit spread: {e}")
                        self.error_count += 1
                elif prediction['strategy'] == 2:
                    try:
                        # Put Credit Spread using real options data
                        position = self._create_put_credit_spread_from_chain(date, prediction)
                        if position:
                            print(f"Adding position: {position.__str__()}")
                            current_price = self.data.loc[date]['Close']
                            print(f"    Current price: {round(current_price, 2)}")
                            add_position(position)
                    except Exception as e:
                        print(f"Error creating put credit spread: {e}")
                        self.error_count += 1
            else:
                self.error_count += 1

        else:
            for position in positions:
                # Calculate exit price for the position
                date_key = date.strftime('%Y-%m-%d')
                if date_key in self.options_data:
                    option_chain = self.options_data[date_key]
                    exit_price = position.calculate_exit_price(option_chain)
                else:
                    exit_price = None

                # Fetch missing contract
                if exit_price is None:
                    for option in position.spread_options:
                        contract = self.options_handler.get_specific_option_contract(option.strike, option.expiration, option.option_type.value, date)
                        if contract is None:
                            print(f"Error: No contract found for {option.strike} {option.expiration} {option.option_type.value}")
                            self.error_count += 1
                            continue
                        
                        if (contract.option_type == OptionType.CALL):
                            option_chain.calls.append(contract)
                        elif (contract.option_type == OptionType.PUT):
                            option_chain.puts.append(contract)
                        else:
                            print(f"Error: Invalid option type: {contract.option_type}")
                            self.error_count += 1
                            continue
                    exit_price = position.calculate_exit_price(option_chain)

                if exit_price is None:
                    print(f"Error calculating exit price for {position.__str__()}")
                    self.error_count += 1
                    continue
                else:
                    exit_price = round(max(exit_price, 0), 2)

                # Determine if we should close a position
                if (self._profit_target_hit(position, exit_price) or self._stop_loss_hit(position, exit_price)):
                    print(f"Profit target or stop loss hit for {position.__str__()}")
                    print(f"    Exit price: {exit_price} for {position.__str__()}")
                    remove_position(position, exit_price)
                elif position.get_days_held(date) >= self.holding_period:
                    print(f"Position {position.__str__()} past holding period")
                    print(f"    Exit price: {exit_price} for {position.__str__()}")
                    remove_position(position, exit_price)
                elif position.get_days_to_expiration(date) < 1:
                    print(f"Position {position.__str__()} expired or near expiration")
                    print(f"    Exit price: {exit_price} for {position.__str__()}")
                    remove_position(position, exit_price)

    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        """
        On end, execute strategy.
        """
        super().on_end(positions, remove_position, date)
        for position in positions:
            try:
                # Calculate the return for this position
                exit_price = position.calculate_exit_price(self.options_data[date.strftime('%Y-%m-%d')])

                # Remove the position and update capital
                remove_position(position, exit_price)

            except Exception as e:
                print(f"   Error closing position {position}: {e}")
        print(f"Total error count: {self.error_count}")

    def _find_best_spread(self, current_price: float, strategy_type: StrategyType, confidence: float, 
                          date: datetime, min_days: int = 20, max_days: int = 40):
        """Find the best spread (expiry and width) minimizing risk/reward and maximizing probability of profit"""
        candidates = []
        total_evaluated = 0
        total_rejected = 0
        
        print(f"   🔍 Evaluating spreads for {strategy_type.value} strategy...")
        
        # Get available expirations from options handler
        if not self.options_handler:
            print("   ⚠️  No options handler available")
            raise Exception("No options handler available")
            
        # Get the option chain for the current date to extract available expirations
        try:
            chain_data = self.options_handler._get_option_chain_with_cache(date, current_price)
            if not chain_data or (not chain_data.calls and not chain_data.puts):
                print("   ⚠️  No option chain data available")
                raise Exception("No option chain data available")
                
            # Extract unique expiration dates from the chain data
            expirations = set()
            for option in chain_data.calls:
                expirations.add(option.expiration)
            for option in chain_data.puts:
                expirations.add(option.expiration)
                        
            if not expirations:
                print("   ⚠️  No expiration dates found in option chain")
                return None
                
            print(f"   📅 Available expirations: {len(expirations)} total")
            
        except Exception as e:
            print(f"   ⚠️  Error getting option chain: {e}")
            raise e
        
        for expiry_str in expirations:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            days_to_expiry = (expiry_date - date.date()).days
            if not (min_days <= days_to_expiry <= max_days):
                continue
                
            print(f"   📊 Evaluating {expiry_str} ({days_to_expiry} days)...")
            
            # Filter options for this expiration
            calls = [opt for opt in chain_data.calls if opt.expiration == expiry_str]
            puts = [opt for opt in chain_data.puts if opt.expiration == expiry_str]
            
            for width in [5, 7, 8, 10, 12, 15, 25]:
                total_evaluated += 1

                if strategy_type == StrategyType.CALL_CREDIT_SPREAD:
                    atm_strike = round(current_price)
                    otm_strike = atm_strike + width
                    
                    # Find ATM and OTM call options
                    atm_call = self._find_option_by_strike(calls, atm_strike)
                    otm_call = self._find_option_by_strike(calls, otm_strike)
                    
                    if not atm_call or not otm_call:
                        total_rejected += 1
                        continue
                        
                    credit = atm_call.last_price - otm_call.last_price
                    max_risk = width - credit
                    direction = 'bearish'
                else:  # PUT_CREDIT_SPREAD
                    atm_strike = round(current_price)
                    otm_strike = atm_strike - width
                    
                    # Find ATM and OTM put options
                    atm_put = self._find_option_by_strike(puts, atm_strike)
                    otm_put = self._find_option_by_strike(puts, otm_strike)
                    
                    if not atm_put or not otm_put:
                        total_rejected += 1
                        continue
                        
                    credit = atm_put.last_price - otm_put.last_price
                    max_risk = width - credit
                    direction = 'bullish'
                    
                if max_risk <= 0 or credit <= 0:
                    total_rejected += 1
                    continue
                    
                risk_reward = credit / max_risk
                prob_profit = self._estimate_probability_of_profit(confidence, direction, width, atm_strike, otm_strike, current_price, days_to_expiry)
                
                # Calculate minimum required risk/reward ratio
                min_risk_reward = (1 - prob_profit) / prob_profit if prob_profit > 0 else float('inf')
                
                # Only include spreads that meet the minimum risk/reward requirement
                if risk_reward >= min_risk_reward:
                    candidates.append({
                        'expiry': expiry_date,
                        'width': width,
                        'atm_strike': atm_strike,
                        'otm_strike': otm_strike,
                        'atm_option': atm_call if strategy_type == StrategyType.CALL_CREDIT_SPREAD else atm_put,
                        'otm_option': otm_call if strategy_type == StrategyType.CALL_CREDIT_SPREAD else otm_put,
                        'credit': credit,
                        'max_risk': max_risk,
                        'risk_reward': risk_reward,
                        'prob_profit': prob_profit,
                        'min_risk_reward': min_risk_reward,
                        'days': days_to_expiry
                    })
                    print(f"      ✅ {width}pt spread: R/R={risk_reward:.2f}, Prob={prob_profit:.1%}, Min={min_risk_reward:.2f}")
                else:
                    total_rejected += 1
                    print(f"      ❌ {width}pt spread: R/R={risk_reward:.2f} < Min={min_risk_reward:.2f} (Prob={prob_profit:.1%})")
        
        print(f"   📈 Evaluation Summary:")
        print(f"      • Total spreads evaluated: {total_evaluated}")
        print(f"      • Spreads rejected: {total_rejected}")
        print(f"      • Spreads meeting criteria: {len(candidates)}")
        
        # Sort by risk/reward ascending (minimize), then probability of profit descending (maximize)
        if candidates:
            candidates.sort(key=lambda x: (x['risk_reward'], -x['prob_profit']))
            best = candidates[0]
            print(f"   🏆 Best spread selected:")
            print(f"      • {best['width']}pt spread expiring {best['expiry'].strftime('%Y-%m-%d')}")
            print(f"      • Risk/Reward: 1:{best['risk_reward']:.2f}, Probability: {best['prob_profit']:.1%}")
            return best
        else:
            print(f"   ❌ No spreads meet the minimum criteria")
            return None

    def _find_option_by_strike(self, options: list[Option], strike: float):
        """Find an option by strike price"""
        for option in options:
            if option.strike == strike:
                return option
        return None

    def _estimate_probability_of_profit(self, confidence: float, direction: str, width: int = None, 
                                       atm_strike: float = None, otm_strike: float = None, 
                                       current_price: float = None, days_to_expiry: int = None) -> float:
        """Estimate probability of profit based on spread characteristics and model confidence"""
        base_prob = confidence
        
        # Base adjustment from model confidence
        if confidence > 0.6:
            base_prob += 0.1
        elif confidence < 0.4:
            base_prob -= 0.1
        
        # Adjust based on spread width (wider spreads = higher probability)
        if width is not None:
            width_bonus = min(0.15, (width - 3) * 0.02)  # +2% per point width, max 15%
            base_prob += width_bonus
        
        # Adjust based on distance from current price (more OTM = higher probability)
        if atm_strike is not None and otm_strike is not None and current_price is not None:
            if direction == 'bullish':  # Put credit spread
                # For put spreads, we want the stock to stay above the short put
                distance_otm = (current_price - atm_strike) / current_price
                distance_bonus = min(0.10, distance_otm * 100)  # +1% per 1% OTM, max 10%
                base_prob += distance_bonus
            else:  # Call credit spread
                # For call spreads, we want the stock to stay below the short call
                distance_otm = (atm_strike - current_price) / current_price
                distance_bonus = min(0.10, distance_otm * 100)  # +1% per 1% OTM, max 10%
                base_prob += distance_bonus
        
        # Adjust based on days to expiration (shorter = higher probability)
        if days_to_expiry is not None:
            if days_to_expiry <= 30:
                time_bonus = 0.05  # +5% for short-term
            elif days_to_expiry <= 45:
                time_bonus = 0.02  # +2% for medium-term
            else:
                time_bonus = -0.02  # -2% for long-term
            base_prob += time_bonus
        
        # Ensure probability is within reasonable bounds
        return max(0.35, min(0.85, base_prob))
    
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

    def _create_call_credit_spread_from_chain(self, date: datetime, prediction: dict) -> Position:
        """Create a call credit spread using the options chain data"""
        if not self.options_data:
            print("⚠️  No options data available")
            return None
            
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            print(f"⚠️  No options data for {date_key}")
            return None
            
        current_price = self.data.loc[date]['Close']
        
        confidence = prediction['confidence'] if prediction else 0.5
        
        # Find best spread using the new method
        best_spread = self._find_best_spread(current_price, StrategyType.CALL_CREDIT_SPREAD, confidence, date)
        
        if not best_spread:
            print("⚠️  No suitable call credit spread found")
            return None
            
        # Convert dictionary options to Option objects
        atm_option = best_spread['atm_option']
        otm_option = best_spread['otm_option']
        
        # Create position using the best spread
        position = Position(
            symbol=self.options_handler.symbol,
            quantity=1,
            expiration_date=datetime.strptime(atm_option.expiration, '%Y-%m-%d'),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=best_spread['atm_strike'],
            entry_date=date,
            entry_price=best_spread['credit'],  # Use net credit as entry price
            spread_options=[atm_option, otm_option]  # Store the specific spread options
        )
        
        print(f"📊 Call Credit Spread:")
        print(f"   Sell ATM Call: ${best_spread['atm_strike']:.0f} @ ${atm_option.last_price:.2f}")
        print(f"   Buy OTM Call: ${best_spread['otm_strike']:.0f} @ ${otm_option.last_price:.2f}")
        print(f"   Net Credit: ${best_spread['credit']:.2f}")
        print(f"   Risk/Reward: 1:{best_spread['risk_reward']:.2f}")
        print(f"   Probability: {best_spread['prob_profit']:.1%}")
        
        return position

    def _create_put_credit_spread_from_chain(self, date: datetime, prediction: dict) -> Position:
        """Create a put credit spread using the options chain data"""
        if not self.options_data:
            print("⚠️  No options data available")
            return None
            
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            print(f"⚠️  No options data for {date_key}")
            return None
            
        current_price = self.data.loc[date]['Close']
        
        if prediction['confidence'] is None:
            print("⚠️  No prediction available")
            raise Exception("No prediction available")
        
        confidence = prediction['confidence']
        
        # Find best spread using the new method
        best_spread = self._find_best_spread(current_price, StrategyType.PUT_CREDIT_SPREAD, confidence, date)
        
        if not best_spread:
            print("⚠️  No suitable put credit spread found")
            return None

        # Convert dictionary options to Option objects
        atm_option = best_spread['atm_option']
        otm_option = best_spread['otm_option']
        
        # Create position using the best spread
        position = Position(
            symbol=self.options_handler.symbol,
            quantity=1,
            expiration_date=datetime.strptime(atm_option.expiration, '%Y-%m-%d'),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=best_spread['atm_strike'],
            entry_date=date,
            entry_price=best_spread['credit'],  # Use net credit as entry price
            spread_options=[atm_option, otm_option]  # Store the specific spread options
        )
        
        print(f"📊 Put Credit Spread:")
        print(f"   Sell ATM Put: ${best_spread['atm_strike']:.0f} @ ${atm_option.last_price:.2f}")
        print(f"   Buy OTM Put: ${best_spread['otm_strike']:.0f} @ ${otm_option.last_price:.2f}")
        print(f"   Net Credit: ${best_spread['credit']:.2f}")
        print(f"   Risk/Reward: 1:{best_spread['risk_reward']:.2f}")
        print(f"   Probability: {best_spread['prob_profit']:.1%}")
        
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

    def _find_otm_call(self, calls: list, atm_strike: float, current_date: datetime, expiration: str):
        """Find an OTM call option (higher strike than ATM)"""
        strike = atm_strike + 10

        # Find calls with higher strikes than ATM
        otm_calls = [call for call in calls if call.strike > strike]
        
        if not otm_calls:
            contract = self.options_handler.get_specific_option_contract(strike, expiration, OptionType.CALL.value, current_date)
            if contract is None:
                return None
            otm_calls = [contract]
            
        # Return the closest OTM call (lowest strike above ATM)
        return min(otm_calls, key=lambda opt: opt.strike)

    def _find_otm_put(self, puts: list, atm_strike: float, current_date: datetime, expiration: str):
        """Find an OTM put option (lower strike than ATM)"""
        strike = atm_strike - 10

        # Find puts with lower strikes than ATM
        otm_puts = [put for put in puts if put.strike < strike]
        
        if not otm_puts:
            contract = self.options_handler.get_specific_option_contract(strike, expiration, OptionType.PUT.value, current_date)
            if contract is None:
                return None
            otm_puts = [contract]
            
        # Return the closest OTM put (highest strike below ATM)
        return max(otm_puts, key=lambda opt: opt.strike)
