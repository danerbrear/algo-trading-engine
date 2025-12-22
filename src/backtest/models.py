from __future__ import annotations

from typing import Callable, Dict, Optional, List
from datetime import datetime
import pandas as pd
from enum import Enum
from src.common.models import OptionChain, Option, TreasuryRates
from src.common.options_dtos import OptionBarDTO
from abc import ABC, abstractmethod

class Benchmark():
    """
    Benchmark to compare returns against
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.start_price = 0.0
        self.end_price = None

    def set_start_price(self, start_price: float):
        """
        Set the start price for the benchmark.
        """
        self.start_price = start_price

    def set_end_price(self, end_price: float):
        """
        Set the end price for the benchmark.
        """
        self.end_price = end_price

    def get_return_dollars(self) -> float:
        """
        Get the return for the benchmark.
        """
        shares = self.initial_capital / self.start_price
        return (self.end_price - self.start_price) * shares

    def get_return_percentage(self) -> float:
        """
        Get the return for the benchmark.
        """
        if self.end_price is None:
            return None
        return (self.end_price - self.start_price) / self.start_price * 100

class OptionType(Enum):
    """
    Enum for option types.
    """
    PUT = "P"
    CALL = "C"

class Strategy(ABC):
    """
    Strategy is a class that represents a trading strategy.
    """

    def __init__(self, profit_target: float = None, stop_loss: float = None, start_date_offset: int = 0):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.data = None
        self.start_date_offset = start_date_offset

    @abstractmethod
    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, execute strategy.
        """

    def on_new_date(self, date: datetime, positions: tuple['Position', ...]):
        """
        On new date, execute strategy.
        """
    
        if len(positions) > 0:
            print(f"\n{date}: Open positions:")
            for position in positions:
                print(f"  {position.__str__()}")

    @abstractmethod
    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        """
        On end, execute strategy.
        """

    def recommend_open_position(self, date: datetime, current_price: float) -> Optional[Dict]:
        """
        Recommend opening a position for the given date and current price.
        
        This method uses the strategy's on_new_date logic to determine if a position
        should be opened, then captures the position creation to return as a recommendation.
        
        Args:
            date: Current date
            current_price: Current underlying price
            
        Returns:
            dict or None: Position recommendation with keys:
                - strategy_type: StrategyType enum value
                - legs: tuple of (atm_option, otm_option) Option objects
                - credit: float, net credit received
                - width: float, strike width
                - probability_of_profit: float, estimated probability of profit
                - confidence: float, model confidence (0.0-1.0)
                - expiration_date: str, expiration in 'YYYY-MM-DD' format
            Returns None if no position should be opened.
        """
        # Store the recommended position to return
        recommended_position = None
        
        def capture_add_position(position: 'Position'):
            """Capture the position created by the strategy's on_new_date logic"""
            nonlocal recommended_position
            recommended_position = position
        
        def dummy_remove_position(date: datetime, position: 'Position', exit_price: float, 
                                 underlying_price: float = None, current_volumes: list[int] = None):
            """Dummy remove_position function - not used in recommendation"""
            pass
        
        # Use the strategy's on_new_date logic with no existing positions
        # This will trigger the strategy to potentially create a new position
        try:
            self.on_new_date(date, (), capture_add_position, dummy_remove_position)
        except Exception as e:
            raise e
        
        # If no position was created, return None
        if recommended_position is None:
            return None
        
        # Extract the recommendation details from the created position
        if not recommended_position.spread_options or len(recommended_position.spread_options) != 2:
            return None
        
        atm_option, otm_option = recommended_position.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        
        return {
            "strategy_type": recommended_position.strategy_type,
            "legs": (atm_option, otm_option),
            "credit": recommended_position.entry_price,
            "width": width,
            "probability_of_profit": 0.7,  # Default confidence for rule-based strategies
            "confidence": 0.7,  # Default confidence for rule-based strategies
            "expiration_date": recommended_position.expiration_date.strftime('%Y-%m-%d'),
        }

    def recommend_close_positions(self, date: datetime, positions: List['Position']) -> List[Dict]:
        """
        Recommend closing positions for the given date and current positions.
        
        This method uses the strategy's on_new_date logic to determine which positions
        should be closed, then captures the position closures to return as recommendations.
        
        Args:
            date: Current date
            positions: List of current open positions
            
        Returns:
            List[Dict]: List of position closure recommendations with keys:
                - position: Position object to close
                - exit_price: float, exit price for the position
                - rationale: str, reason for closing
            Returns empty list if no positions should be closed.
        """
        if not positions:
            return []
        
        # Store the positions that should be closed
        positions_to_close = []
        
        def capture_remove_position(date: datetime, position: 'Position', exit_price: float, 
                                  underlying_price: float = None, current_volumes: list[int] = None):
            """Capture the position closure decision from the strategy's on_new_date logic"""
            positions_to_close.append({
                "position": position,
                "exit_price": exit_price,
                "underlying_price": underlying_price,
                "current_volumes": current_volumes,
                "rationale": "strategy_decision"
            })
        
        def dummy_add_position(position: 'Position'):
            """Dummy add_position function - not used for closing"""
            pass
        
        # Use the strategy's on_new_date logic with existing positions
        # This will trigger the strategy to potentially close positions
        try:
            self.on_new_date(date, tuple(positions), dummy_add_position, capture_remove_position)
        except Exception as e:
            # If on_new_date fails, return empty list
            return []
        
        return positions_to_close

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data for this specific strategy.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid for this strategy, False otherwise
        """

    def set_profit_target(self, profit_target: float):
        """
        Set the profit target for the strategy.
        """
        self.profit_target = profit_target

    def set_stop_loss(self, stop_loss: float):
        """
        Set the stop loss for the strategy.
        """
        self.stop_loss = stop_loss

    def set_data(self, data: pd.DataFrame, treasury_data: Optional[TreasuryRates] = None):
        """
        Set the data for the strategy.
        """
        self.data = data
        self.treasury_data = treasury_data

    def _profit_target_hit(self, position: 'Position', exit_price: float) -> bool:
        """
        Check if the profit target has been hit for a position.
        """
        if self.profit_target is None:
            return False
        return position.profit_target_hit(self.profit_target, exit_price)

    def _stop_loss_hit(self, position: 'Position', exit_price: float) -> bool:
        """
        Check if the stop loss has been hit for a position.
        """
        if self.stop_loss is None:
            return False
        return position.stop_loss_hit(self.stop_loss, exit_price)

class StrategyType(Enum):
    """
    Enum for strategy types.
    """
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    LONG_STOCK = "long_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

class Position:
    """
    Position is a class that represents a position in a stock.
    
    Args:
        spread_options (list[Option]): List of Option objects that make up the spread (e.g., [atm_option, otm_option])
    """

    def __init__(self, symbol: str, expiration_date: datetime, strategy_type: StrategyType, strike_price: float, entry_date: datetime, entry_price: float, exit_price: float = None, spread_options: list[Option] = None):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.quantity = None
        self.strategy_type = strategy_type
        self.strike_price = strike_price
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.spread_options: list[Option] = spread_options if spread_options is not None else []
        # Runtime type check
        if self.spread_options and not all(isinstance(opt, Option) for opt in self.spread_options):
            raise TypeError("All elements of spread_options must be of type Option")
        
    def set_quantity(self, quantity: int):
        """
        Set the quantity for a position.
        """
        self.quantity = quantity
    
    def profit_target_hit(self, profit_target: float, exit_price: float) -> bool:
        """
        Check if the profit target has been hit for a position.
        """
        return self._get_return(exit_price) >= profit_target if exit_price is not None else False
    
    def stop_loss_hit(self, stop_loss: float, exit_price: float) -> bool:
        """
        Check if the stop loss has been hit for a position.
        """
        return self._get_return(exit_price) <= -stop_loss if exit_price is not None else False
    
    def get_days_to_expiration(self, current_date: datetime) -> int:
        """
        Get the number of days to expiration for a position from the given current_date.
        """
        if self.expiration_date is not None:
            # Ensure both datetimes are timezone-naive for consistent comparison
            if current_date.tzinfo is not None:
                current_date = current_date.replace(tzinfo=None)
            if self.expiration_date.tzinfo is not None:
                expiration_date = self.expiration_date.replace(tzinfo=None)
            else:
                expiration_date = self.expiration_date
            return (expiration_date - current_date).days
        else:
            raise ValueError("Expiration date is not set")

    def get_days_held(self, current_date: datetime) -> int:
        """
        Get the number of days held for a position from the given current_date.
        Uses date-only comparison for trading day calculation.
        """
        if self.entry_date is not None:
            # Convert to date-only for trading day calculation
            if current_date.tzinfo is not None:
                current_date = current_date.replace(tzinfo=None)
            if self.entry_date.tzinfo is not None:
                entry_date = self.entry_date.replace(tzinfo=None)
            else:
                entry_date = self.entry_date
            
            # Use date-only comparison for trading days
            current_date_only = current_date.date()
            entry_date_only = entry_date.date()
            return (current_date_only - entry_date_only).days
        else:
            raise ValueError("Entry date is not set")
        
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """
        Get the return from assignment for a credit spread position at expiration.
        
        For credit spreads:
        - Initial credit received is stored in entry_price
        - Maximum risk is the width of the spread minus the credit received
        - At expiration, we calculate the intrinsic value of our short and long legs
        """
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread options are not set")
            
        atm_option, otm_option = self.spread_options
        
        if self.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            # Short ATM call, Long OTM call
            short_strike = atm_option.strike  # ATM strike
            long_strike = otm_option.strike   # OTM strike (higher)
            
            # Calculate intrinsic values at expiration
            short_intrinsic = max(0, underlying_price - short_strike)
            long_intrinsic = max(0, underlying_price - long_strike)
            
            # Net P&L = Initial credit - Short leg cost + Long leg value
            # Since we sold the short leg and bought the long leg
            net_pnl = self.entry_price - short_intrinsic + long_intrinsic
            
        elif self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            # Short ATM put, Long OTM put
            short_strike = atm_option.strike  # ATM strike
            long_strike = otm_option.strike   # OTM strike (lower)
            
            # Calculate intrinsic values at expiration
            short_intrinsic = max(0, short_strike - underlying_price)
            long_intrinsic = max(0, long_strike - underlying_price)
            
            # Net P&L = Initial credit - Short leg cost + Long leg value
            net_pnl = self.entry_price - short_intrinsic + long_intrinsic
            
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
        
        return net_pnl * self.quantity * 100
        
    def get_return_dollars(self, exit_price: float) -> float:
        """
        Get the return in dollars for a position.
        """
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        
        # For credit spreads, the return is: Initial Credit - Cost to Close
        if self.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
            # entry_price = net credit received when opening
            # exit_price = cost to close the position
            # Return = Credit received - Cost to close
            return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
        else:
            # For other position types, use the standard calculation
            return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """
        Get the percentage return for a position.
        """
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        
        # For credit spreads, calculate percentage return based on max risk
        if self.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
            return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
        else:
            # For other position types, use the standard calculation
            return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """
        Calculate the current exit price for a credit spread position based on current market prices.
        
        Args:
            current_option_chain: Current option chain data for the date
            current_date: Current date for API fetching if needed
            current_price: Current stock price for API fetching if needed
            
        Returns:
            float: Current net credit/debit for the spread, or None if option contract data is not available
        """
        
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread options are not set")
            
        atm_option, otm_option = self.spread_options

        # Find current prices for our specific options
        current_atm_option = current_option_chain.get_option_data_for_option(atm_option)
        current_otm_option = current_option_chain.get_option_data_for_option(otm_option)
        
        if current_atm_option is None or current_otm_option is None:
            print(f"No current option data found for {atm_option.__str__()} or {otm_option.__str__()}")
            return None

        current_atm_price = current_atm_option.last_price
        current_otm_price = current_otm_option.last_price

        print(f"Current ATM price: {current_atm_price}, Current OTM price: {current_otm_price}")

        # Calculate current net credit/debit based on strategy type
        if self.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            # For call credit spread: sell ATM call, buy OTM call
            current_net_credit = current_atm_price - current_otm_price
            return current_net_credit
        elif self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            # For put credit spread: sell ATM put, buy OTM put
            current_net_credit = current_atm_price - current_otm_price
            return current_net_credit
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """
        Calculate the current exit price for a credit spread position using OptionBarDTO data.
        
        Args:
            atm_bar: OptionBarDTO for the ATM option
            otm_bar: OptionBarDTO for the OTM option
            
        Returns:
            float: Current net credit/debit for the spread
            
        Raises:
            ValueError: If the input options don't match the position's spread_options
        """
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread options are not set")
            
        atm_option, otm_option = self.spread_options
        
        # Verify the input bars match our spread options
        # Check ATM option match - either ticker match OR (expiration, strike, type) match
        if not self._options_match(atm_bar, atm_option):
            raise ValueError(f"ATM bar doesn't match position ATM option. Expected: {atm_option.ticker} (strike: {atm_option.strike}, exp: {atm_option.expiration}, type: {atm_option.option_type.value}), Got: {atm_bar.ticker}")
            
        # Check OTM option match - either ticker match OR (expiration, strike, type) match
        if not self._options_match(otm_bar, otm_option):
            raise ValueError(f"OTM bar doesn't match position OTM option. Expected: {otm_option.ticker} (strike: {otm_option.strike}, exp: {otm_option.expiration}, type: {otm_option.option_type.value}), Got: {otm_bar.ticker}")

        # Extract current prices from bar data
        current_atm_price = float(atm_bar.close_price)
        current_otm_price = float(otm_bar.close_price)

        print(f"Current ATM price: {current_atm_price}, Current OTM price: {current_otm_price}")

        # Calculate current net credit/debit based on strategy type
        if self.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            # For call credit spread: sell ATM call, buy OTM call
            current_net_credit = current_atm_price - current_otm_price
            return current_net_credit
        elif self.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            # For put credit spread: sell ATM put, buy OTM put
            current_net_credit = current_atm_price - current_otm_price
            return current_net_credit
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
    
    def _options_match(self, bar: OptionBarDTO, option: Option) -> bool:
        """
        Check if an OptionBarDTO matches an Option by either ticker match OR 
        (expiration, strike, type) match.
        
        Args:
            bar: OptionBarDTO to check
            option: Option to match against
            
        Returns:
            bool: True if options match, False otherwise
        """
        # Direct ticker match
        if bar.ticker == option.ticker:
            return True
            
        # Check expiration date match
        bar_expiration = bar.timestamp.date()
        option_expiration = datetime.strptime(option.expiration, '%Y-%m-%d').date()
        if bar_expiration != option_expiration:
            return False
            
        # Extract strike and type from bar ticker
        # Format: O:SYMBOLyymmdd[C/P]strikeprice
        try:
            # Find the option type in the ticker
            option_type_in_ticker = None
            if 'C' in bar.ticker:
                option_type_in_ticker = 'C'
            elif 'P' in bar.ticker:
                option_type_in_ticker = 'P'
            
            # Check option type match
            if option_type_in_ticker != option.option_type.value:
                return False
            
            # Extract strike price from ticker
            # Find the position after the option type
            type_pos = bar.ticker.find(option_type_in_ticker)
            if type_pos == -1:
                return False
                
            strike_str = bar.ticker[type_pos + 1:]
            if len(strike_str) >= 8:
                strike_from_ticker = float(strike_str[:8]) / 1000
                return abs(strike_from_ticker - option.strike) < 0.001  # Allow small floating point differences
                
        except (ValueError, IndexError):
            return False
            
        return False
    
    def get_max_risk(self):
        """
        Determine the max loss for a position.
        """
        atm_option, otm_option = self.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        net_credit = atm_option.last_price - otm_option.last_price
        return (width - net_credit) * 100
    
    def __eq__(self, other) -> bool:
        """
        Check if two positions are equal based on key attributes.
        
        Two positions are considered equal if they have:
        - Same symbol
        - Same strategy type  
        - Same strike price
        - Same expiration date
        - Same entry date
        - Same exit date (if both have one)
        - Same spread options (if both have them)
        
        Args:
            other: Another Position object to compare against
            
        Returns:
            bool: True if positions are equal, False otherwise
        """
        if not isinstance(other, Position):
            return False
            
        # Compare basic attributes
        if (self.symbol != other.symbol or
            self.strategy_type != other.strategy_type or
            self.strike_price != other.strike_price or
            self.expiration_date != other.expiration_date or
            self.entry_date != other.entry_date):
            return False
            
        # Compare exit dates if both have them
        if hasattr(self, 'exit_date') and hasattr(other, 'exit_date'):
            if self.exit_date != other.exit_date:
                return False
        elif hasattr(self, 'exit_date') or hasattr(other, 'exit_date'):
            # One has exit date, other doesn't
            return False
            
        # Compare spread options if both have them
        if self.spread_options and other.spread_options:
            if len(self.spread_options) != len(other.spread_options):
                return False
            # Compare each spread option using Option's __eq__ method
            for self_opt, other_opt in zip(self.spread_options, other.spread_options):
                if self_opt != other_opt:
                    return False
        elif self.spread_options or other.spread_options:
            # One has spread options, other doesn't
            return False
            
        return True

    def __str__(self) -> str:
        return f"{self.strategy_type.value} {self.symbol} {self.strike_price} @ {self.entry_price:.2f} x{self.quantity} (Open, expires {self.expiration_date.strftime('%Y-%m-%d')})"
    