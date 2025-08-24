from typing import Callable, Dict, Optional
from datetime import datetime
import pandas as pd
from enum import Enum
from src.common.models import OptionChain, Option, TreasuryRates
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
        self.options_data = None
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

    def set_data(self, data: pd.DataFrame, options_data: Dict[str, OptionChain], treasury_data: Optional[TreasuryRates] = None):
        """
        Set the data for the strategy.
        """
        self.data = data
        self.options_data = options_data
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
            return (self.expiration_date - current_date).days
        else:
            raise ValueError("Expiration date is not set")

    def get_days_held(self, current_date: datetime) -> int:
        """
        Get the number of days held for a position from the given current_date.
        """
        if self.entry_date is not None:
            return (current_date - self.entry_date).days
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
    
    def get_max_risk(self):
        """
        Determine the max loss for a position.
        """
        atm_option, otm_option = self.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        net_credit = atm_option.last_price - otm_option.last_price
        return (width - net_credit) * 100
    
    def __str__(self) -> str:
        return f"{self.strategy_type.value} {self.symbol} {self.strike_price} @ {self.entry_price:.2f} x{self.quantity} (Open, expires {self.expiration_date.strftime('%Y-%m-%d')})"
    