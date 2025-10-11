from datetime import datetime
from typing import Callable, Optional
import pandas as pd

from src.backtest.models import Position, Strategy, StrategyType
from src.common.models import Option, OptionType
from src.model.options_handler import OptionsHandler
from src.common.progress_tracker import progress_print


class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    holding_period = 25

    def __init__(self, lstm_model, lstm_scaler, options_handler: OptionsHandler = None, start_date_offset: int = 0):
        super().__init__(stop_loss=0.6, start_date_offset=start_date_offset)
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
        
        super().on_new_date(date, positions)

        has_error = False

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
                            current_price = self.data.loc[date]['Close']
                            print(f"    Current price: {round(current_price, 2)}")
                            add_position(position)
                    except Exception as e:
                        print(f"Error creating call credit spread: {e}")
                        has_error = True
                elif prediction['strategy'] == 2:
                    try:
                        # Put Credit Spread using real options data
                        position = self._create_put_credit_spread_from_chain(date, prediction)
                        if position:
                            current_price = self.data.loc[date]['Close']
                            print(f"    Current price: {round(current_price, 2)}")
                            add_position(position)
                    except Exception as e:
                        print(f"Error creating put credit spread: {e}")
                        has_error = True
            else:
                has_error = True

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
                            has_error = True
                        
                        if contract is not None:
                            if (contract.option_type == OptionType.CALL):
                                option_chain = option_chain.add_option(contract)
                            elif (contract.option_type == OptionType.PUT):
                                option_chain = option_chain.add_option(contract)
                            else:
                                print(f"Error: Invalid option type: {contract.option_type}")
                                has_error = True

                    if not has_error:
                        exit_price = position.calculate_exit_price(option_chain)

                if exit_price is None or has_error:
                    print(f"Error calculating exit price for {position.__str__()}")
                    has_error = True
                else:
                    exit_price = round(max(exit_price, 0), 2)

                # Determine if we should close a position
                try:
                    if position.get_days_to_expiration(date) < 1:
                        print(f"Position {position.__str__()} expired or near expiration")
                        underlying_price = self._get_current_underlying_price(date)
                        if underlying_price is not None:
                            print(f"    Underlying price: {underlying_price}")
                            remove_position(date, position, exit_price, underlying_price)
                        else:
                            print("    Failed to get underlying price")
                    elif (self._profit_target_hit(position, exit_price) or self._stop_loss_hit(position, exit_price)):
                        print(f"Profit target or stop loss hit for {position.__str__()}")
                        print(f"    Exit price: {exit_price} for {position.__str__()}")
                        remove_position(date, position, exit_price)
                    elif position.get_days_held(date) >= self.holding_period:
                        print(f"Position {position.__str__()} past holding period")
                        if exit_price:
                            print(f"    Exit price: {exit_price} for {position.__str__()}")
                            remove_position(date, position, exit_price)
                        else:
                            print(f"    No exit price available for {position.__str__()} on {date}. Skipping.")
                except Exception as e:
                    print(f"Error closing position {position}: {e}")
                    import traceback
                    traceback.print_exc()
                    has_error = True

        if has_error:
            self.error_count += 1

    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        """
        On end, execute strategy with enhanced current date volume validation.
        """
        for position in positions:
            try:
                # Calculate the return for this position
                exit_price = position.calculate_exit_price(self.options_data[date.strftime('%Y-%m-%d')])

                # Fetch current date volume data for enhanced validation
                current_volumes = self.get_current_volumes_for_position(position, date)

                # Remove the position and update capital with current date volume validation
                remove_position(date, position, exit_price, current_volumes=current_volumes)
            except Exception as e:
                print(f"   Error closing position {position}: {e}")
                import traceback
                traceback.print_exc()
                raise e
            
        print(f"Total error count: {self.error_count}")

    def _find_best_spread(self, current_price: float, strategy_type: StrategyType, confidence: float, 
                          date: datetime, min_days: int = 20, max_days: int = 40):
        """Find the best spread (expiry and width) minimizing risk/reward and maximizing probability of profit"""
        candidates = []
        total_evaluated = 0
        total_rejected = 0
        
        print(f"   🔍 Evaluating spreads for {strategy_type.value} strategy...")
        
        # Get available expirations from options handler (lightweight approach)
        if not self.options_handler:
            print("   ⚠️  No options handler available")
            raise Exception("No options handler available")
            
        # Get available expiration dates without fetching full option chains
        try:
            expirations = self.options_handler.get_available_expirations(date, min_days, max_days)
            
            if not expirations:
                print("   ⚠️  No expiration dates found in target range")
                return None
                
            print(f"   📅 Available expirations: {len(expirations)} total")
            
        except Exception as e:
            print(f"   ⚠️  Error getting expiration dates: {e}")
            raise e
        
        for expiry_str in expirations:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            days_to_expiry = (expiry_date - date.date()).days
            if not (min_days <= days_to_expiry <= max_days):
                continue
                
            print(f"   📊 Evaluating {expiry_str} ({days_to_expiry} days)...")

            for width in [5, 7, 8, 10, 12, 15]:
                total_evaluated += 1

                if strategy_type == StrategyType.CALL_CREDIT_SPREAD:
                    atm_strike = round(current_price)
                    otm_strike = atm_strike + width
                    
                    # Find ATM and OTM call options
                    atm_call = self.options_handler.get_specific_option_contract(atm_strike, expiry_date, OptionType.CALL.value, date)
                    otm_call = self.options_handler.get_specific_option_contract(otm_strike, expiry_date, OptionType.CALL.value, date)
                    
                    if not atm_call or not otm_call:
                        progress_print(f"      ❌ No ATM or OTM call options found for {width}pt spread")
                        total_rejected += 1
                        continue
                        
                    credit = atm_call.last_price - otm_call.last_price
                    max_risk = width - credit
                    direction = 'bearish'
                else:  # PUT_CREDIT_SPREAD
                    atm_strike = round(current_price)
                    otm_strike = atm_strike - width
                    
                    # Find ATM and OTM put options
                    atm_put = self.options_handler.get_specific_option_contract(atm_strike, expiry_date, OptionType.PUT.value, date)
                    otm_put = self.options_handler.get_specific_option_contract(otm_strike, expiry_date, OptionType.PUT.value, date)
                    
                    if not atm_put or not otm_put:
                        print(f"      ❌ No ATM or OTM put options found for {width}pt spread")
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
                
                # Calculate expected value and minimum required risk/reward ratio
                expected_value = (credit * prob_profit) - (max_risk * (1 - prob_profit))
                
                # Minimum R/R ratio based on probability of profit
                # For credit spreads, we want at least 1:1 R/R for 50% probability
                # Higher probability trades can accept lower R/R ratios
                min_risk_reward = 1.0 / prob_profit if prob_profit > 0 else float('inf')
                
                # Only include spreads that are profitable (positive expected value)
                # and meet minimum risk/reward requirements
                if expected_value > 0 and risk_reward >= min_risk_reward:
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
                        'expected_value': expected_value,
                        'days': days_to_expiry
                    })
                    print(f"      ✅ {width}pt spread: R/R={risk_reward:.2f}, Prob={prob_profit:.1%}, EV=${expected_value:.2f}, Min={min_risk_reward:.2f}")
                else:
                    total_rejected += 1
                    print(f"      ❌ {width}pt spread: R/R={risk_reward:.2f} < Min={min_risk_reward:.2f} or EV=${expected_value:.2f} <= 0 (Prob={prob_profit:.1%})")
        
        print(f"   📈 Evaluation Summary:")
        print(f"      • Total spreads evaluated: {total_evaluated}")
        print(f"      • Spreads rejected: {total_rejected}")
        print(f"      • Spreads meeting criteria: {len(candidates)}")
        
        # Sort by expected value descending (maximize), then risk/reward descending (maximize)
        if candidates:
            candidates.sort(key=lambda x: (-x.get('expected_value', 0), -x['risk_reward'], -x['prob_profit']))
            best = candidates[0]
            print(f"   🏆 Best spread selected:")
            print(f"      • {best['width']}pt spread expiring {best['expiry'].strftime('%Y-%m-%d')}")
            print(f"      • Risk/Reward: 1:{best['risk_reward']:.2f}, Probability: {best['prob_profit']:.1%}, Expected Value: ${best['expected_value']:.2f}")
            return best
        else:
            print(f"   ❌ No spreads meet the minimum criteria")
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

    def map_prediction_to_strategy_type(self, prediction: dict) -> Optional[StrategyType]:
        """
        Map prediction integer to StrategyType enum.
        
        Args:
            prediction: Dictionary containing 'strategy' key with integer value
            
        Returns:
            StrategyType or None: Mapped strategy type, None for hold (strategy=0)
        """
        strategy_int = prediction.get('strategy')
        
        if strategy_int == 1:
            return StrategyType.CALL_CREDIT_SPREAD
        elif strategy_int == 2:
            return StrategyType.PUT_CREDIT_SPREAD
        else:
            # strategy_int == 0 or invalid values return None (no position to open)
            return None

    def _create_call_credit_spread_from_chain(self, date: datetime, prediction: dict) -> Position:
        """Create a call credit spread using the options chain data"""
        if not self.options_data:
            print("⚠️  No options data available")
            return None
            
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            print(f"⚠️  No options data for {date_key}")
            return None
            
        current_price = self._get_current_underlying_price(date)
        if current_price is None:
            print("⚠️  Failed to get current price")
            return None
        
        confidence = prediction['confidence'] if prediction else 0.5
        
        # Find best spread using the new method
        best_spread = self._find_best_spread(current_price, StrategyType.CALL_CREDIT_SPREAD, confidence, date)
        
        if not best_spread:
            print("⚠️  No suitable call credit spread found")
            return None
            
        # Convert dictionary options to Option objects
        atm_option = best_spread['atm_option']
        otm_option = best_spread['otm_option']
        
        # Ensure volume data is available for both options
        atm_option = self._ensure_volume_data(atm_option, date)
        otm_option = self._ensure_volume_data(otm_option, date)
        
        if atm_option is None or otm_option is None:
            print(f"⚠️  Could not fetch volume data for options")
            return None
        
        # Create position using the best spread
        position = Position(
            symbol=self.options_handler.symbol,
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
        print(f"   Max Risk: ${position.get_max_risk():.2f}")
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
            
        current_price = self._get_current_underlying_price(date)
        if current_price is None:
            print("⚠️  Failed to get current price")
            return None
        
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
        
        # Ensure volume data is available for both options
        atm_option = self._ensure_volume_data(atm_option, date)
        otm_option = self._ensure_volume_data(otm_option, date)
        
        if atm_option is None or otm_option is None:
            print(f"⚠️  Could not fetch volume data for options")
            return None
        
        # Create position using the best spread
        position = Position(
            symbol=self.options_handler.symbol,
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
        print(f"   Max Risk: ${position.get_max_risk():.2f}")
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

    def _ensure_volume_data(self, option: Option, date: datetime) -> Option:
        """
        Ensure option has volume data, fetch if missing.
        
        Args:
            option: The option to check/fetch volume data for
            date: The date for the data
            
        Returns:
            Option: The option with volume data, or None if unable to fetch
        """
        if option.volume is not None:
            return option
        
        # Fetch fresh data from API
        fresh_option = self.options_handler.get_specific_option_contract(
            option.strike, 
            option.expiration, 
            option.option_type.value, 
            date
        )
        
        if fresh_option and fresh_option.volume is not None:
            print(f"📡 Fetched volume data for {option.symbol}: {fresh_option.volume}")
            return fresh_option
        else:
            print(f"⚠️  No volume data available for {option.symbol}")
            return None

    def recommend_open_position(self, date: datetime, current_price: float) -> Optional[dict]:
        """
        Recommend opening a position for the given date and current price.
        
        Args:
            date: Current date
            current_price: Current underlying price
            
        Returns:
            dict or None: Position recommendation with required keys, None if no position should be opened
        """
        # Get model prediction
        prediction = self._make_prediction(date)
        if prediction is None:
            return None
            
        # Map prediction to strategy type
        strategy_type = self.map_prediction_to_strategy_type(prediction)
        if strategy_type is None:
            return None
        print(f"****Strategy type: {strategy_type.value}****")
            
        confidence = float(prediction.get("confidence", 0.5))
        
        # Find best spread
        best = self._find_best_spread(current_price, strategy_type, confidence, date)
        if not best:
            print("No suitable spread found")
            return None

        # Ensure volume for both legs
        atm_option = best["atm_option"]
        otm_option = best["otm_option"]
        atm_option = self._ensure_volume_data(atm_option, date)
        otm_option = self._ensure_volume_data(otm_option, date)
        if atm_option is None or otm_option is None:
            print("Could not fetch volume data for options")
            return None

        # Return standardized recommendation dict
        return {
            "strategy_type": strategy_type,
            "legs": (atm_option, otm_option),
            "credit": float(best["credit"]),
            "width": float(best["width"]),
            "probability_of_profit": float(best.get("prob_profit", confidence)),
            "confidence": confidence,
            "expiration_date": best["expiry"].strftime("%Y-%m-%d"),
        }

    def _get_current_underlying_price(self, date: datetime) -> Optional[float]:
        """Get current underlying price, using live price if date is current date."""
        # Check if the specified date is the current date
        current_date = datetime.now().date()
        if date.date() == current_date:
            # Use live price from DataRetriever if available
            if hasattr(self, 'data_retriever') and self.data_retriever:
                live_price = self.data_retriever.get_live_price()
                if live_price is not None:
                    return live_price
            elif hasattr(self.options_handler, 'symbol'):
                # Fallback: create a temporary DataRetriever for live price
                from src.common.data_retriever import DataRetriever
                temp_retriever = DataRetriever(symbol=self.options_handler.symbol, use_free_tier=True, quiet_mode=True)
                temp_retriever.options_handler = self.options_handler
                live_price = temp_retriever.get_live_price()
                if live_price is not None:
                    return live_price
        
        # Fallback to cached data if live price failed or date is not current
        if self.data is None or self.data.empty or date not in self.data.index:
            return None
        try:
            return float(self.data.loc[date]['Close'])
        except Exception:
            return None

    def get_current_volumes_for_position(self, position: Position, date: datetime) -> list[int]:
        """
        Fetch current date volume data for all options in a position.
        
        Args:
            position: The position containing options to check
            date: The current date for volume validation
            
        Returns:
            list[int]: List of current volume values for each option in position.spread_options
        """
        current_volumes = []
        
        for option in position.spread_options:
            try:
                # Fetch fresh data from API for the current date
                fresh_option = self.options_handler.get_specific_option_contract(
                    option.strike, 
                    option.expiration, 
                    option.option_type.value, 
                    date  # Use the current date for closure validation
                )
                
                if fresh_option and fresh_option.volume is not None:
                    current_volumes.append(fresh_option.volume)
                    print(f"📡 Fetched volume data for {option.symbol} on {date.date()}: {fresh_option.volume}")
                else:
                    current_volumes.append(None)
                    print(f"⚠️  No volume data available for {option.symbol} on {date.date()}")
                    
            except Exception as e:
                print(f"⚠️  Error fetching volume data for {option.symbol}: {e}")
                current_volumes.append(None)
        
        return current_volumes

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data for the Credit Spread Strategy.
        
        This strategy requires:
        - Basic OHLCV data
        - LSTM model features (technical indicators, market state, calendar features)
        - Options data for spread creation
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid for this strategy, False otherwise
        """
        progress_print(f"\n🔍 Validating data for Credit Spread Strategy...")
        progress_print(f"   Data shape: {data.shape}")
        
        # Check if the data has the required columns for credit spread strategy
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
            progress_print(f"⚠️  Warning: Missing required columns: {missing_columns}")
            progress_print(f"   Available columns: {list(data.columns)}")
            return False
        else:
            progress_print(f"✅ All required columns present")
        
        # Check if data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            progress_print("❌ Error: Data must have a datetime index for backtesting")
            return False
        
        # Check if we have enough data for LSTM model (need at least sequence_length days)
        if self.lstm_model and hasattr(self.lstm_model, 'sequence_length'):
            min_required_days = self.lstm_model.sequence_length
            if len(data) < min_required_days:
                progress_print(f"⚠️  Warning: Not enough data for LSTM model. Need at least {min_required_days} days, got {len(data)}")
                return False
        else:
            # Fallback: need at least 50 days for technical indicators
            if len(data) < 50:
                progress_print(f"⚠️  Warning: Not enough data for technical analysis. Need at least 50 days, got {len(data)}")
                return False
        
        # Check for gaps in the data (missing trading days)
        if len(data) > 1:
            date_range = pd.bdate_range(start=data.index.min(), end=data.index.max())
            expected_business_days = len(date_range)
            actual_trading_days = len(data)
            if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
                progress_print(f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}")
        
        # Check for missing values in critical columns
        critical_columns = ['Close', 'Volume', 'Market_State']
        for col in critical_columns:
            if col in data.columns and data[col].isnull().any():
                null_count = data[col].isnull().sum()
                progress_print(f"⚠️  Warning: {null_count} missing values found in {col}")
        
        progress_print(f"✅ Data validation complete for Credit Spread Strategy")
        progress_print(f"   Final data shape: {data.shape}")
        progress_print(f"   Date range: {data.index.min()} to {data.index.max()}")
        progress_print(f"   Trading days: {len(data)}")
        
        return True
