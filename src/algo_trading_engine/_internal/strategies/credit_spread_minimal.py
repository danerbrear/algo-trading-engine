from __future__ import annotations

import traceback
from datetime import datetime, date
from typing import Callable, Optional, Sequence
import pandas as pd

from algo_trading_engine._internal.util.market_clock import label_position, market_index, session_dates
from algo_trading_engine._internal.util.date_formats import DateFormats
from algo_trading_engine._internal.common.ml_pipeline import prepare_credit_spread_backtest_data
from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo.universe import PostLoadCallback, UniverseSet
from algo_trading_engine.vo import Position, create_position
from algo_trading_engine.enums import OptionType, StrategyType
from algo_trading_engine.vo import Option
from algo_trading_engine._internal.common.progress_tracker import progress_print
from algo_trading_engine.dto import ExpirationRangeDTO, StrikeRangeDTO
from algo_trading_engine.vo import StrikePrice, ExpirationDate
from algo_trading_engine._internal.ml_models.lstm_model import LSTMModel
from decimal import Decimal


class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    holding_period = 25
    universe_set: UniverseSet = ["SPY"]

    def __init__(
        self,
        get_contract_list_for_date: Optional[Callable],
        get_option_bar: Optional[Callable],
        get_options_chain: Optional[Callable],
        lstm_model,
        lstm_scaler,
        options_handler=None,
    ):
        super().__init__(stop_loss=0.6)
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler

        if get_contract_list_for_date is not None:
            self.get_contract_list_for_date = get_contract_list_for_date
        if get_option_bar is not None:
            self.get_option_bar = get_option_bar
        if get_options_chain is not None:
            self.get_options_chain = get_options_chain
        # Store options_handler for LSTMModel compatibility (temporary bridge)
        self._options_handler = options_handler

        self.error_count = 0

    def on_universe_load_complete(self) -> PostLoadCallback:
        def post_load(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
            enriched = prepare_credit_spread_backtest_data(data, symbol)
            if self._options_handler is None:
                raise ValueError("options_handler is required for LSTMModel compatibility")
            with_features = LSTMModel.calculate_option_features(enriched, self._options_handler, symbol)
            return LSTMModel.calculate_option_signals(with_features)

        return post_load

    def on_new_date(
        self,
        date: datetime,
        positions: tuple["Position", ...],
        add_position: Callable[["Position"], None],
        remove_position: Strategy.RemovePositionCallback,
    ):
        """
        On new date, determine if a new position should be opened.
        We should not open a position if we already have one.
        """
        super().on_new_date(date, positions, add_position, remove_position)

        if self.universe is None:
            return

        has_error = False
        for symbol in self.universe.symbols:
            if not self.universe.has_bar(symbol, date):
                continue
            symbol_positions = tuple(p for p in positions if p.symbol == symbol)
            if len(symbol_positions) == 0:
                if self._open_predicted_position(date, add_position, symbol):
                    has_error = True
            elif self._close_positions_for_date(date, symbol_positions, remove_position, symbol):
                has_error = True

        if has_error:
            self.error_count += 1

    def on_end(self, positions: Sequence["Position"], remove_position: Strategy.RemovePositionCallback, date: datetime):
        """
        On end, execute strategy with enhanced current date volume validation.
        """
        for position in positions:
            try:
                underlying_price = self.get_current_underlying_price(date, position.symbol)
                exit_price = position.exit_price_from_leg_quotes(underlying_price)
                if exit_price is None:
                    raise TypeError("exit_price is required")

                # Fetch current date volume data for enhanced validation
                current_volumes = self.get_current_volumes_for_position(position, date)

                # Remove the position and update capital with current date volume validation
                remove_position(date, position, exit_price, current_volumes=current_volumes)
            except Exception as e:
                print(f"   Error closing position {position}: {e}")
                traceback.print_exc()
                raise e

        print(f"Total error count: {self.error_count}")

    def map_prediction_to_strategy_type(self, prediction: dict) -> Optional[StrategyType]:
        """
        Map prediction integer to StrategyType enum.

        Args:
            prediction: Dictionary containing 'strategy' key with integer value

        Returns:
            StrategyType or None: Mapped strategy type, None for hold (strategy=0)
        """
        strategy_int = prediction.get("strategy")

        if strategy_int == 1:
            return StrategyType.CALL_CREDIT_SPREAD
        elif strategy_int == 2:
            return StrategyType.PUT_CREDIT_SPREAD
        else:
            # strategy_int == 0 or invalid values return None (no position to open)
            return None

    def get_current_volumes_for_position(self, position: Position, date: datetime | None) -> list[int | None]:
        """
        Return the volume already stored on each leg.

        ``Strategy._update_positions`` refreshes those volumes for ``date`` before close checks.
        """
        _ = date
        return [option.volume for option in position.legs]

    def _open_predicted_position(self, date: datetime, add_position: Callable[["Position"], None], symbol: str) -> bool:
        """Open the predicted credit spread. Returns True when opening fails."""
        print("No positions, opening new position")
        prediction = self._make_prediction(date, symbol)
        if prediction is None:
            return True
        if prediction["strategy"] == 1:
            spread_name = "call"
            create_spread = self._create_call_credit_spread_from_chain
        elif prediction["strategy"] == 2:
            spread_name = "put"
            create_spread = self._create_put_credit_spread_from_chain
        else:
            return False

        try:
            position = create_spread(date, prediction, symbol)
            if position:
                market_data = self.market_frame(symbol)
                current_price = market_data.loc[date]["Close"]
                print(f"    Current price: {round(current_price, 2)}")
                add_position(position)
        except Exception as error:
            print(f"Error creating {spread_name} credit spread: {error}")
            return True
        return False

    def _close_positions_for_date(
        self,
        date: datetime,
        positions: tuple["Position", ...],
        remove_position: Strategy.RemovePositionCallback,
        symbol: str,
    ) -> bool:
        """Apply close rules to open positions. Returns True when a close check fails."""
        has_error = False
        for position in positions:
            exit_price, quote_failed = self._quoted_exit_price(position, date, symbol)
            if quote_failed:
                has_error = True
            if exit_price is None or has_error:
                print(f"Error calculating exit price for {position.__str__()}")
                has_error = True
            else:
                exit_price = round(max(exit_price, 0), 2)
            if self._close_position_if_due(date, position, exit_price, remove_position, symbol):
                has_error = True
        return has_error

    def _quoted_exit_price(self, position: "Position", date: datetime, symbol: str):
        """Return the leg-quote exit price and whether quoting failed."""
        underlying_price = self.get_current_underlying_price(date, symbol)
        try:
            return position.exit_price_from_leg_quotes(underlying_price), False
        except Exception as error:
            print(f"Error calculating exit price for {position.__str__()}: {error}")
            return None, True

    def _close_position_if_due(
        self,
        date: datetime,
        position: "Position",
        exit_price,
        remove_position: Strategy.RemovePositionCallback,
        symbol: str,
    ) -> bool:
        """Close the position when a rule is met. Returns True when closing raises."""
        try:
            if position.get_days_to_expiration(date) < 1:
                print(f"Position {position.__str__()} expired or near expiration")
                underlying_price = self.get_current_underlying_price(date, symbol)
                if underlying_price is not None:
                    print(f"    Underlying price: {underlying_price}")
                    if exit_price is None:
                        raise TypeError("exit_price is required")
                    remove_position(date, position, exit_price, underlying_price)
                else:
                    print("    Failed to get underlying price")
            elif self._profit_target_hit(position, exit_price) or self._stop_loss_hit(position, exit_price):
                print(f"Profit target or stop loss hit for {position.__str__()}")
                print(f"    Exit price: {exit_price} for {position.__str__()}")
                if exit_price is None:
                    raise TypeError("exit_price is required")
                remove_position(date, position, exit_price)
            elif position.get_days_held(date) >= self.holding_period:
                print(f"Position {position.__str__()} past holding period")
                if exit_price:
                    print(f"    Exit price: {exit_price} for {position.__str__()}")
                    remove_position(date, position, exit_price)
                else:
                    print(f"    No exit price available for {position.__str__()} on {date}. Skipping.")
        except Exception as error:
            print(f"Error closing position {position}: {error}")
            traceback.print_exc()
            return True
        return False

    def _find_best_spread(
        self,
        current_price: float,
        strategy_type: StrategyType,
        confidence: float,
        date: datetime,
        symbol: str,
        min_days: int = 20,
        max_days: int = 40,
    ):
        """Find the best spread (expiry and width) minimizing risk/reward and maximizing probability of profit"""
        candidates = []
        total_evaluated = 0
        total_rejected = 0

        print(f"   🔍 Evaluating spreads for {strategy_type.value} strategy...")

        # Get available expirations from options handler using new API
        if not self._options_handler:
            print("   ⚠️  No options handler available")
            raise ValueError("No options handler available")

        # Get available expiration dates using get_contract_list_for_date with ExpirationRangeDTO
        try:
            expiration_range = ExpirationRangeDTO(min_days=min_days, max_days=max_days)
            contracts = self.get_contract_list_for_date(symbol, date, expiration_range=expiration_range)
            expirations = sorted(set(str(contract.expiration_date) for contract in contracts))

            if not expirations:
                print("   ⚠️  No expiration dates found in target range")
                return None

            print(f"   📅 Available expirations: {len(expirations)} total")

        except Exception as e:
            print(f"   ⚠️  Error getting expiration dates: {e}")
            raise e

        for expiry_str in expirations:
            expiry_date = datetime.strptime(expiry_str, DateFormats.EXPIRATION).date()
            days_to_expiry = (expiry_date - date.date()).days
            if not (min_days <= days_to_expiry <= max_days):
                continue

            print(f"   📊 Evaluating {expiry_str} ({days_to_expiry} days)...")

            for width in [5, 7, 8, 10, 12, 15]:
                total_evaluated += 1

                if strategy_type == StrategyType.CALL_CREDIT_SPREAD:
                    atm_strike = round(current_price)
                    otm_strike = atm_strike + width

                    # Find ATM and OTM call options using new API
                    atm_call = self._get_option_with_bar(atm_strike, expiry_date, OptionType.CALL, date, symbol)
                    otm_call = self._get_option_with_bar(otm_strike, expiry_date, OptionType.CALL, date, symbol)

                    if not atm_call or not otm_call:
                        progress_print(f"      ❌ No ATM or OTM call options found for {width}pt spread")
                        total_rejected += 1
                        continue

                    credit = atm_call.last_price - otm_call.last_price
                    max_risk = width - credit
                    direction = "bearish"
                    atm_option = atm_call
                    otm_option = otm_call
                else:  # PUT_CREDIT_SPREAD
                    atm_strike = round(current_price)
                    otm_strike = atm_strike - width

                    # Find ATM and OTM put options using new API
                    atm_put = self._get_option_with_bar(atm_strike, expiry_date, OptionType.PUT, date, symbol)
                    otm_put = self._get_option_with_bar(otm_strike, expiry_date, OptionType.PUT, date, symbol)

                    if not atm_put or not otm_put:
                        print(f"      ❌ No ATM or OTM put options found for {width}pt spread")
                        total_rejected += 1
                        continue

                    credit = atm_put.last_price - otm_put.last_price
                    max_risk = width - credit
                    direction = "bullish"
                    atm_option = atm_put
                    otm_option = otm_put

                if max_risk <= 0 or credit <= 0:
                    total_rejected += 1
                    continue

                risk_reward = credit / max_risk
                prob_profit = self._estimate_probability_of_profit(
                    confidence, direction, width, atm_strike, otm_strike, current_price, days_to_expiry
                )

                # Calculate expected value and minimum required risk/reward ratio
                expected_value = (credit * prob_profit) - (max_risk * (1 - prob_profit))

                # Minimum R/R ratio based on probability of profit
                # For credit spreads, we want at least 1:1 R/R for 50% probability
                # Higher probability trades can accept lower R/R ratios
                min_risk_reward = 1.0 / prob_profit if prob_profit > 0 else float("inf")

                # Only include spreads that are profitable (positive expected value)
                # and meet minimum risk/reward requirements
                if expected_value > 0 and risk_reward >= min_risk_reward:
                    candidates.append(
                        {
                            "expiry": expiry_date,
                            "width": width,
                            "atm_strike": atm_strike,
                            "otm_strike": otm_strike,
                            "atm_option": atm_option,
                            "otm_option": otm_option,
                            "credit": credit,
                            "max_risk": max_risk,
                            "risk_reward": risk_reward,
                            "prob_profit": prob_profit,
                            "min_risk_reward": min_risk_reward,
                            "expected_value": expected_value,
                            "days": days_to_expiry,
                        }
                    )
                    print(
                        f"      ✅ {width}pt spread: R/R={risk_reward:.2f}, Prob={prob_profit:.1%}, EV=${expected_value:.2f}, Min={min_risk_reward:.2f}"
                    )
                else:
                    total_rejected += 1
                    print(
                        f"      ❌ {width}pt spread: R/R={risk_reward:.2f} < Min={min_risk_reward:.2f} or EV=${expected_value:.2f} <= 0 (Prob={prob_profit:.1%})"
                    )

        print("   📈 Evaluation Summary:")
        print(f"      • Total spreads evaluated: {total_evaluated}")
        print(f"      • Spreads rejected: {total_rejected}")
        print(f"      • Spreads meeting criteria: {len(candidates)}")

        # Sort by expected value descending (maximize), then risk/reward descending (maximize)
        if candidates:
            candidates.sort(key=lambda x: (-x.get("expected_value", 0), -x["risk_reward"], -x["prob_profit"]))
            best = candidates[0]
            print("   🏆 Best spread selected:")
            print(f"      • {best['width']}pt spread expiring {best['expiry'].strftime(DateFormats.EXPIRATION)}")
            print(
                f"      • Risk/Reward: 1:{best['risk_reward']:.2f}, Probability: {best['prob_profit']:.1%}, Expected Value: ${best['expected_value']:.2f}"
            )
            return best
        else:
            print("   ❌ No spreads meet the minimum criteria")
            return None

    def _get_option_with_bar(
        self, strike: float, expiry_date: date, option_type: OptionType, date: datetime, symbol: str
    ) -> Optional[Option]:
        """
        Get an Option object with price/volume data using new API.

        Args:
            strike: Strike price
            expiry_date: Expiration date (date object)
            option_type: OptionType enum
            date: Current date for data lookup
            symbol: Underlying symbol

        Returns:
            Option object if found, None otherwise
        """
        try:
            # Get contract DTO (metadata only)
            strike_price = StrikePrice(Decimal(str(strike)))
            expiration_date = ExpirationDate(expiry_date)
            strike_range = StrikeRangeDTO(min_strike=strike_price, max_strike=strike_price)
            expiration_range = ExpirationRangeDTO(target_date=expiration_date, current_date=date.date())

            contracts = self.get_contract_list_for_date(
                symbol, date, strike_range=strike_range, expiration_range=expiration_range
            )

            # Find matching contract by type
            contract_dto = next((c for c in contracts if c.contract_type == option_type), None)

            if contract_dto is None:
                return None

            # Get price/volume data (real-time snapshot when live, else historical /aggs)
            bar = self.get_current_option_bar(symbol, contract_dto.ticker, date)
            if bar is None:
                return None

            # Combine contract + bar into Option object
            return Option.from_contract_and_bar(contract_dto, bar)

        except Exception as e:
            print(f"⚠️  Error getting option with bar for strike {strike}, expiry {expiry_date}: {e}")
            return None

    def _estimate_probability_of_profit(
        self,
        confidence: float,
        direction: str,
        width: Optional[int] = None,
        atm_strike: Optional[float] = None,
        otm_strike: Optional[float] = None,
        current_price: Optional[float] = None,
        days_to_expiry: Optional[int] = None,
    ) -> float:
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
            if direction == "bullish":  # Put credit spread
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

    def _make_prediction(self, date: datetime, symbol: str):
        """
        Make a prediction from the LSTM model for a given date.

        Args:
            date: The date to make a prediction for
            symbol: Underlying symbol

        Returns:
            dict: Dictionary containing:
                - strategy: int (0=Hold, 1=Call Credit Spread, 2=Put Credit Spread)
                - confidence: float (probability of the predicted strategy)
                - probabilities: dict (probabilities for all strategies)
        """
        data = self.market_frame(symbol)

        if self.lstm_model is None:
            raise ValueError("LSTM model not set for strategy.")

        # Ensure the date is a pandas Timestamp and matches the index type
        date_key = pd.Timestamp(date)

        # First try exact match
        if date_key in data.index:
            date_idx = label_position(data.index, date_key)
        else:
            # Try to normalize both index and date (remove time component)
            date_key = datetime(date.year, date.month, date.day)
            index_normalized = session_dates(market_index(data))

            if date_key in index_normalized:
                date_idx = label_position(index_normalized, date_key)
            else:
                # Provide detailed error information
                print(f"❌ Date {date} not found in data index.")
                print(f"   Available date range: {data.index.min()} to {data.index.max()}")
                print(f"   Total data points: {len(data)}")
                print(f"   Normalized date: {date_key}")
                print(f"   First 5 available dates: {list(data.index[:5])}")
                print(f"   Last 5 available dates: {list(data.index[-5:])}")

                # Check if it's a weekend or holiday
                if date_key.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    print(f"   Note: {date_key.date()} is a weekend (market closed)")
                else:
                    print(f"   Note: {date_key.date()} might be a market holiday or missing data")

                return None

        print(f"Date index: {date_idx}")  # TODO: Remove this

        # Check if we have enough data for the sequence
        sequence_length = self.lstm_model.sequence_length
        if date_idx < sequence_length:
            print(
                f"Warning: Not enough data for prediction at {date}. Need at least {sequence_length} days of history."
            )
            return {"strategy": 0, "confidence": 0.0, "probabilities": {0: 1.0, 1: 0.0, 2: 0.0}}  # Default to Hold

        # Get the feature columns that the LSTM model expects
        feature_columns = self.lstm_model.feature_columns
        if feature_columns is None:
            # Fallback to the original features if not set
            feature_columns = [
                "High_Low_Range",
                "SMA20_to_SMA50",
                "SMA20_to_SMA50_Lag1",
                "SMA20_to_SMA50_Lag5",
                "SMA20_to_SMA50_MA5",
                "SMA20_to_SMA50_MA10",
                "SMA20_to_SMA50_Std5",
                "SMA20_to_SMA50_Momentum",
                "RSI",
                "RSI_Lag1",
                "RSI_Lag5",
                "RSI_MA5",
                "RSI_MA10",
                "RSI_Std5",
                "RSI_Momentum",
                "RSI_Overbought",
                "RSI_Oversold",
                "MACD_Hist",
                "Volume_Ratio",
                "OBV",
                "Put_Call_Ratio",
                "Option_Volume_Ratio",
                "Market_State",
                "Days_Until_Next_CPI",
                "Days_Since_Last_CPI",
                "Days_Until_Next_CC",
                "Days_Since_Last_CC",
                "Days_Until_Next_FFR",
                "Days_Since_Last_FFR",
            ]

        # Extract the sequence of features for the prediction
        start_idx = date_idx - sequence_length + 1
        end_idx = date_idx + 1

        # Get the feature data for the sequence
        sequence_data = data.iloc[start_idx:end_idx][feature_columns].copy()

        # Check if all required features are available in the sequence data
        missing_features = [col for col in feature_columns if col not in sequence_data.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features for LSTM prediction: {missing_features}. Available columns: {list(sequence_data.columns)}"
            )

        # Check for missing values in the sequence - these indicate missing options data
        # Skip prediction if any values are missing in the sequence
        if sequence_data.isnull().any().any():
            # Identify which features have missing values for debugging
            missing_cols = sequence_data.columns[sequence_data.isnull().any()].tolist()
            missing_count = sequence_data.isnull().sum().sum()
            missing_dates = sequence_data[sequence_data.isnull().any(axis=1)].index.tolist()

            # Log but don't fail - just skip this prediction
            print(
                f"⚠️  Skipping prediction for {date} - sequence has {missing_count} missing values in {len(missing_cols)} features"
            )
            print(
                f"   Dates in sequence with missing data: {[d.strftime(DateFormats.ISO_DATE) for d in missing_dates[:5]]}{'...' if len(missing_dates) > 5 else ''}"
            )
            return None

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
            strategy_labels = ["Hold", "Call Credit Spread", "Put Credit Spread"]
            predicted_strategy = strategy_labels[prediction]
            confidence = probabilities[prediction]

            print(f"LSTM Prediction for {date}: {predicted_strategy} (confidence: {confidence:.2%})")

            return {
                "strategy": int(prediction),
                "confidence": float(confidence),
                "probabilities": {
                    0: float(probabilities[0]),  # Hold
                    1: float(probabilities[1]),  # Call Credit Spread
                    2: float(probabilities[2]),  # Put Credit Spread
                },
            }

        except Exception as e:
            print(f"Error making LSTM prediction: {e}")
            raise ValueError(f"Error making LSTM prediction: {e}") from e

    def _create_call_credit_spread_from_chain(
        self, date: datetime, prediction: dict, symbol: str
    ) -> Optional[Position]:
        """Create a call credit spread using the options chain data"""
        current_price = self.get_current_underlying_price(date, symbol)
        if current_price is None:
            print("⚠️  Failed to get current price")
            return None

        confidence = prediction["confidence"] if prediction else 0.5

        # Find best spread using the new method
        best_spread = self._find_best_spread(current_price, StrategyType.CALL_CREDIT_SPREAD, confidence, date, symbol)

        if not best_spread:
            print("⚠️  No suitable call credit spread found")
            return None

        # Convert dictionary options to Option objects
        atm_option = best_spread["atm_option"]
        otm_option = best_spread["otm_option"]

        # Ensure volume data is available for both options
        atm_option = self._ensure_volume_data(atm_option, date, symbol)
        otm_option = self._ensure_volume_data(otm_option, date, symbol)

        if atm_option is None or otm_option is None:
            print("⚠️  Could not fetch volume data for options")
            return None

        # Create position using the best spread
        position = create_position(
            symbol=symbol,
            expiration_date=datetime.strptime(atm_option.expiration, DateFormats.EXPIRATION),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=best_spread["atm_strike"],
            entry_date=date,
            entry_price=best_spread["credit"],  # Use net credit as entry price
            legs=[atm_option, otm_option],  # Store the specific spread options
        )

        print("📊 Call Credit Spread:")
        print(f"   Sell ATM Call: ${best_spread['atm_strike']:.0f} @ ${atm_option.last_price:.2f}")
        print(f"   Buy OTM Call: ${best_spread['otm_strike']:.0f} @ ${otm_option.last_price:.2f}")
        print(f"   Net Credit: ${best_spread['credit']:.2f}")
        print(f"   Max Risk: ${position.max_risk_dollars_per_contract():.2f}")
        print(f"   Risk/Reward: 1:{best_spread['risk_reward']:.2f}")
        print(f"   Probability: {best_spread['prob_profit']:.1%}")

        return position

    def _create_put_credit_spread_from_chain(self, date: datetime, prediction: dict, symbol: str) -> Optional[Position]:
        """Create a put credit spread using the options chain data"""

        current_price = self.get_current_underlying_price(date, symbol)
        if current_price is None:
            print("⚠️  Failed to get current price")
            return None

        if prediction["confidence"] is None:
            print("⚠️  No prediction available")
            raise ValueError("No prediction available")

        confidence = prediction["confidence"]

        # Find best spread using the new method
        best_spread = self._find_best_spread(current_price, StrategyType.PUT_CREDIT_SPREAD, confidence, date, symbol)

        if not best_spread:
            print("⚠️  No suitable put credit spread found")
            return None

        # Convert dictionary options to Option objects
        atm_option = best_spread["atm_option"]
        otm_option = best_spread["otm_option"]

        # Ensure volume data is available for both options
        atm_option = self._ensure_volume_data(atm_option, date, symbol)
        otm_option = self._ensure_volume_data(otm_option, date, symbol)

        if atm_option is None or otm_option is None:
            print("⚠️  Could not fetch volume data for options")
            return None

        # Create position using the best spread
        position = create_position(
            symbol=symbol,
            expiration_date=datetime.strptime(atm_option.expiration, DateFormats.EXPIRATION),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=best_spread["atm_strike"],
            entry_date=date,
            entry_price=best_spread["credit"],  # Use net credit as entry price
            legs=[atm_option, otm_option],  # Store the specific spread options
        )

        print("📊 Put Credit Spread:")
        print(f"   Sell ATM Put: ${best_spread['atm_strike']:.0f} @ ${atm_option.last_price:.2f}")
        print(f"   Buy OTM Put: ${best_spread['otm_strike']:.0f} @ ${otm_option.last_price:.2f}")
        print(f"   Net Credit: ${best_spread['credit']:.2f}")
        print(f"   Max Risk: ${position.max_risk_dollars_per_contract():.2f}")
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

    def _find_otm_call(self, calls: list, atm_strike: float, current_date: datetime, expiration: str, symbol: str):
        """Find an OTM call option (higher strike than ATM)"""
        strike = atm_strike + 10

        # Find calls with higher strikes than ATM
        otm_calls = [call for call in calls if call.strike > strike]

        if not otm_calls:
            contract = self._get_option_with_bar(
                strike,
                datetime.strptime(expiration, DateFormats.EXPIRATION).date(),
                OptionType.CALL,
                current_date,
                symbol,
            )
            if contract is None:
                return None
            otm_calls = [contract]

        # Return the closest OTM call (lowest strike above ATM)
        return min(otm_calls, key=lambda opt: opt.strike)

    def _find_otm_put(self, puts: list, atm_strike: float, current_date: datetime, expiration: str, symbol: str):
        """Find an OTM put option (lower strike than ATM)"""
        strike = atm_strike - 10

        # Find puts with lower strikes than ATM
        otm_puts = [put for put in puts if put.strike < strike]

        if not otm_puts:
            contract = self._get_option_with_bar(
                strike,
                datetime.strptime(expiration, DateFormats.EXPIRATION).date(),
                OptionType.PUT,
                current_date,
                symbol,
            )
            if contract is None:
                return None
            otm_puts = [contract]

        # Return the closest OTM put (highest strike below ATM)
        return max(otm_puts, key=lambda opt: opt.strike)

    def _ensure_volume_data(self, option: Option, date: datetime, symbol: str) -> Optional[Option]:
        """
        Ensure option has volume data, fetch if missing.

        Args:
            option: The option to check/fetch volume data for
            date: The date for the data
            symbol: Underlying symbol

        Returns:
            Option: The option with volume data, or None if unable to fetch
        """
        if option.volume is not None:
            return option

        # Fetch fresh data from API using new API
        fresh_option = self._get_option_with_bar(
            option.strike,
            datetime.strptime(option.expiration, DateFormats.EXPIRATION).date(),
            option.option_type,
            date,
            symbol,
        )

        if fresh_option and fresh_option.volume is not None:
            print(f"📡 Fetched volume data for {option.symbol}: {fresh_option.volume}")
            return fresh_option
        else:
            print(f"⚠️  No volume data available for {option.symbol}")
            return None
