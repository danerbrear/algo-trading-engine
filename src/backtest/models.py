from typing import Callable, Dict
from datetime import datetime
import pandas as pd
from enum import Enum
from src.common.models import OptionChain

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

class Position:
    """
    Position is a class that represents a position in a stock.
    """

    def __init__(self, symbol: str, quantity: int, expiration_date: datetime, option_type: str, strike_price: float, entry_date: datetime, entry_price: float, exit_price: float = None):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.quantity = quantity
        self.option_type = option_type
        self.strike_price = strike_price
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_price = exit_price
    
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
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def __str__(self) -> str:
        """
        String representation of the position.
        """
        if self.exit_price is not None:
            return_pct = self._get_return(self.exit_price) * 100
            return_dollars = self.get_return_dollars(self.exit_price)
            return f"{self.symbol} {self.option_type} {self.strike_price} @ {self.entry_price:.2f} -> {self.exit_price:.2f} ({return_pct:+.2f}%, ${return_dollars:+.2f})"
        else:
            return f"{self.symbol} {self.option_type} {self.strike_price} @ {self.entry_price:.2f} (Open, expires {self.expiration_date.strftime('%Y-%m-%d')})"
    