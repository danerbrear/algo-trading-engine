from typing import Callable, Dict
from datetime import datetime
import pandas as pd
from enum import Enum
from src.common.models import OptionChain, Option
import os
from src.model.options_handler import OptionsHandler

class OptionType(Enum):
    """
    Enum for option types.
    """
    PUT = "P"
    CALL = "C"

class Strategy:
    """
    Strategy is a class that represents a trading strategy.
    """

    def __init__(self, profit_target: float = None, stop_loss: float = None):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.data = None
        self.options_data = None

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

    def set_data(self, data: pd.DataFrame, options_data: Dict[str, OptionChain]):
        """
        Set the data for the strategy.
        """
        self.data = data
        self.options_data = options_data

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, execute strategy.
        """
        pass

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
    
    def on_end(self, positions: tuple['Position', ...]):
        """
        On end, execute strategy.
        """
        pass

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

    def __init__(self, symbol: str, quantity: int, expiration_date: datetime, strategy_type: StrategyType, strike_price: float, entry_date: datetime, entry_price: float, exit_price: float = None, spread_options: list[Option] = None):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.quantity = quantity
        self.strategy_type = strategy_type
        self.strike_price = strike_price
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.spread_options: list[Option] = spread_options if spread_options is not None else []
        # Runtime type check
        if self.spread_options and not all(isinstance(opt, Option) for opt in self.spread_options):
            raise TypeError("All elements of spread_options must be of type Option")
    
    def profit_target_hit(self, profit_target: float, exit_price: float) -> bool:
        """
        Check if the profit target has been hit for a position.
        """
        return self._get_return(exit_price) >= profit_target
    
    def stop_loss_hit(self, stop_loss: float, exit_price: float) -> bool:
        """
        Check if the stop loss has been hit for a position.
        """
        return self._get_return(exit_price) <= -stop_loss
    
    def get_days_to_expiration(self, current_date: datetime) -> int:
        """
        Get the number of days to expiration for a position from the given current_date.
        """
        if self.expiration_date is not None:
            return (self.expiration_date - current_date).days
        else:
            raise ValueError("Expiration date is not set")
        
    def get_return_dollars(self, exit_price: float) -> float:
        """
        Get the return in dollars for a position.
        """
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """
        Get the percentage return for a position.
        """
        if exit_price is None:
            return None
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

        if current_option_chain is None:
            raise ValueError("Current option chain is None")

        # Find current prices for our specific options
        current_atm_option = current_option_chain.get_option_data_for_option(atm_option)
        current_otm_option = current_option_chain.get_option_data_for_option(otm_option)
        
        if current_atm_option is None or current_otm_option is None:
            print(f"No current option data found for {atm_option.__str__()} or {otm_option.__str__()}")
            return None

        current_atm_price = current_atm_option.last_price
        current_otm_price = current_otm_option.last_price
            
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

    def __str__(self) -> str:
        if self.exit_price is not None:
            return_pct = self._get_return(self.exit_price) * 100
            return_dollars = self.get_return_dollars(self.exit_price)
            return f"{self.symbol} {self.strike_price} @ {self.entry_price:.2f} -> {self.exit_price:.2f} ({return_pct:+.2f}%, ${return_dollars:+.2f})"
        else:
            return f"{self.symbol} {self.strike_price} @ {self.entry_price:.2f} (Open, expires {self.expiration_date.strftime('%Y-%m-%d')})"
    