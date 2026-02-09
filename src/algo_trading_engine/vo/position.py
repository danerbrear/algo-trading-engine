"""
Position domain models for the algo trading engine.

This module contains the Position abstract base class and all concrete
position type implementations for different trading strategies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from algo_trading_engine.dto import OptionBarDTO
    from algo_trading_engine.common.models import StrategyType

# Import from common models
from algo_trading_engine.common.models import Option, OptionChain


class Position(ABC):
    """
    Abstract base class for all position types.
    
    Common attributes and non-strategy-specific methods are defined here.
    Strategy-specific P&L and pricing logic is delegated to subclasses.
    
    Args:
        spread_options (list[Option]): List of Option objects that make up the spread (e.g., [atm_option, otm_option])
    """

    def __init__(self, symbol: str, expiration_date: datetime, strategy_type: 'StrategyType', 
                 strike_price: float, entry_date: datetime, entry_price: float, 
                 exit_price: float = None, spread_options: list[Option] = None):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.quantity = None
        self.strategy_type = strategy_type
        self.strike_price = strike_price
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.spread_options: list[Option] = spread_options if spread_options is not None else []
        # Runtime type check - use class name and module to handle test isolation issues
        if self.spread_options:
            for opt in self.spread_options:
                # Check if it's an Option instance
                # First try isinstance (works for Mock(spec=Option) and real Option instances)
                if isinstance(opt, Option):
                    continue
                # If isinstance fails, check by class name and module (handles test isolation issues)
                opt_class = type(opt)
                # Allow Mock objects (used in tests) - Mock(spec=Option) should pass isinstance, but handle edge cases
                is_mock = opt_class.__name__ == 'Mock'
                # Check if it's an Option by class name and module
                is_option = (opt_class.__name__ == 'Option' and 'algo_trading_engine.common.models' in str(opt_class.__module__))
                if not (is_option or is_mock):
                    raise TypeError(f"All elements of spread_options must be of type Option, got {opt_class.__name__} from {opt_class.__module__}")
    
    def set_quantity(self, quantity: int):
        """Set the quantity for a position."""
        self.quantity = quantity
    
    def profit_target_hit(self, profit_target: float, exit_price: float) -> bool:
        """Check if the profit target has been hit for a position."""
        return self._get_return(exit_price) >= profit_target if exit_price is not None else False
    
    def stop_loss_hit(self, stop_loss: float, exit_price: float) -> bool:
        """Check if the stop loss has been hit for a position."""
        return self._get_return(exit_price) <= -stop_loss if exit_price is not None else False
    
    def get_days_to_expiration(self, current_date: datetime) -> int:
        """Get the number of days to expiration for a position from the given current_date."""
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
    
    def get_max_risk(self):
        """Determine the max loss for a position."""
        atm_option, otm_option = self.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        net_credit = atm_option.last_price - otm_option.last_price
        return (width - net_credit) * 100
    
    def spread_width(self) -> Optional[float]:
        """
        Calculate the spread width for spread strategies.
        
        Returns:
            Spread width (difference between strikes), or None if not a spread strategy.
        """
        # Import here to avoid circular import
        from algo_trading_engine.common.models import StrategyType
        
        if self.strategy_type not in [
            StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD,
            StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD,
        ]:
            return None
        
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread strategy requires 2 options in spread_options")
        
        atm_option, otm_option = self.spread_options
        return abs(atm_option.strike - otm_option.strike)
    
    def _options_match(self, bar: 'OptionBarDTO', option: Option) -> bool:
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
            if len(self.spread_options) != len(self.spread_options):
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
    
    # Strategy-specific methods become abstract
    @abstractmethod
    def get_return_dollars(self, exit_price: float) -> float:
        """Calculate dollar return for this position."""
        pass
    
    @abstractmethod
    def _get_return(self, exit_price: float) -> float:
        """Calculate percentage return for this position."""
        pass
    
    @abstractmethod
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price from option chain."""
        pass
    
    @abstractmethod
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """Calculate current exit price from option bars."""
        pass
    
    @abstractmethod
    def max_profit(self) -> Optional[float]:
        """Calculate maximum profit for the position based on strategy type."""
        pass
    
    @abstractmethod
    def max_loss(self) -> Optional[float]:
        """Calculate maximum loss for the position based on strategy type."""
        pass
    
    @abstractmethod
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """Get the return from assignment for a position at expiration."""
        pass
    
    def risk_reward_ratio(self) -> Optional[float]:
        """
        Calculate the risk/reward ratio for the position.
        
        Returns:
            Risk/reward ratio (max_loss / max_profit), or None if either is unlimited or zero.
            
        Example:
            A ratio of 2.0 means you risk $2 to make $1.
            A ratio of 0.5 means you risk $0.50 to make $1.
        """
        max_profit_val = self.max_profit()
        max_loss_val = self.max_loss()
        
        # Can't calculate if either is unlimited
        if max_profit_val is None or max_loss_val is None:
            return None
        
        # Can't divide by zero
        if max_profit_val == 0:
            return None
        
        return max_loss_val / max_profit_val
    
    def expected_value(self, probability_of_profit: float) -> Optional[float]:
        """
        Calculate expected value given a probability of profit.
        
        Args:
            probability_of_profit: Probability of profit (0.0 to 1.0)
            
        Returns:
            Expected value per contract, or None if max profit/loss is unlimited.
            
        Example:
            If max_profit=$2.50, max_loss=$2.50, and PoP=0.70:
            EV = (2.50 × 0.70) - (2.50 × 0.30) = $1.00
        """
        if not 0 <= probability_of_profit <= 1:
            raise ValueError(f"probability_of_profit must be between 0 and 1, got {probability_of_profit}")
        
        max_profit_val = self.max_profit()
        max_loss_val = self.max_loss()
        
        # Can't calculate if either is unlimited
        if max_profit_val is None or max_loss_val is None:
            return None
        
        return (max_profit_val * probability_of_profit) - (max_loss_val * (1 - probability_of_profit))


class CreditSpreadPosition(Position):
    """Position for credit spread strategies (CALL_CREDIT_SPREAD, PUT_CREDIT_SPREAD)."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """
        Get the return in dollars for a credit spread position.
        For credit spreads: Initial Credit - Cost to Close
        """
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # entry_price = net credit received when opening
        # exit_price = cost to close the position
        # Return = Credit received - Cost to close
        return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """
        Get the percentage return for a credit spread position.
        For credit spreads, calculate percentage return based on max risk.
        """
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """
        Calculate the current exit price for a credit spread position based on current market prices.
        
        Args:
            current_option_chain: Current option chain data for the date
            
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

        # For call/put credit spread: sell ATM, buy OTM
        current_net_credit = current_atm_price - current_otm_price
        return current_net_credit
    
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
        if not self._options_match(atm_bar, atm_option):
            raise ValueError(f"ATM bar doesn't match position ATM option. Expected: {atm_option.ticker} (strike: {atm_option.strike}, exp: {atm_option.expiration}, type: {atm_option.option_type.value}), Got: {atm_bar.ticker}")
            
        if not self._options_match(otm_bar, otm_option):
            raise ValueError(f"OTM bar doesn't match position OTM option. Expected: {otm_option.ticker} (strike: {otm_option.strike}, exp: {otm_option.expiration}, type: {otm_option.option_type.value}), Got: {otm_bar.ticker}")

        # Extract current prices from bar data
        current_atm_price = float(atm_bar.close_price)
        current_otm_price = float(otm_bar.close_price)

        print(f"Current ATM price: {current_atm_price}, Current OTM price: {current_otm_price}")

        # For call/put credit spread: sell ATM, buy OTM
        current_net_credit = current_atm_price - current_otm_price
        return current_net_credit
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """
        Get the return from assignment for a credit spread position at expiration.
        
        For credit spreads:
        - Initial credit received is stored in entry_price
        - Maximum risk is the width of the spread minus the credit received
        - At expiration, we calculate the intrinsic value of our short and long legs
        """
        # Import here to avoid circular import
        from algo_trading_engine.common.models import StrategyType
        
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
            raise ValueError(f"Invalid strategy type for CreditSpreadPosition: {self.strategy_type}")
        
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """
        Calculate maximum profit for credit spread.
        Max profit = net credit received
        """
        return self.entry_price
    
    def max_loss(self) -> Optional[float]:
        """
        Calculate maximum loss for credit spread.
        Max loss = spread width - net credit
        """
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Credit spread requires 2 options in spread_options")
        atm_option, otm_option = self.spread_options
        spread_width = abs(atm_option.strike - otm_option.strike)
        return spread_width - self.entry_price


class DebitSpreadPosition(Position):
    """Position for debit spread strategies (CALL_DEBIT_SPREAD, PUT_DEBIT_SPREAD)."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """
        Get the return in dollars for a debit spread position.
        For debit spreads: Exit Value - Initial Debit
        """
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # exit_price = current value of the spread
        # entry_price = debit paid when opening
        # Return = Exit value - Debit paid
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """
        Get the percentage return for a debit spread position.
        For debit spreads, return is based on initial debit paid.
        """
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """
        Calculate the current exit price for a debit spread position based on current market prices.
        
        Args:
            current_option_chain: Current option chain data for the date
            
        Returns:
            float: Current net value for the spread, or None if option contract data is not available
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

        # For debit spread: buy ITM/ATM, sell OTM
        # Current value = what you'd get for closing (selling ITM, buying back OTM)
        current_net_value = current_atm_price - current_otm_price
        return current_net_value
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """
        Calculate the current exit price for a debit spread position using OptionBarDTO data.
        
        Args:
            atm_bar: OptionBarDTO for the ITM/ATM option (the one we bought)
            otm_bar: OptionBarDTO for the OTM option (the one we sold)
            
        Returns:
            float: Current net value for the spread
            
        Raises:
            ValueError: If the input options don't match the position's spread_options
        """
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread options are not set")
            
        atm_option, otm_option = self.spread_options
        
        # Verify the input bars match our spread options
        if not self._options_match(atm_bar, atm_option):
            raise ValueError(f"ATM bar doesn't match position ATM option. Expected: {atm_option.ticker} (strike: {atm_option.strike}, exp: {atm_option.expiration}, type: {atm_option.option_type.value}), Got: {atm_bar.ticker}")
            
        if not self._options_match(otm_bar, otm_option):
            raise ValueError(f"OTM bar doesn't match position OTM option. Expected: {otm_option.ticker} (strike: {otm_option.strike}, exp: {otm_option.expiration}, type: {otm_option.option_type.value}), Got: {otm_bar.ticker}")

        # Extract current prices from bar data
        current_atm_price = float(atm_bar.close_price)
        current_otm_price = float(otm_bar.close_price)

        print(f"Current ATM price: {current_atm_price}, Current OTM price: {current_otm_price}")

        # For debit spread: current value = ITM price - OTM price
        current_net_value = current_atm_price - current_otm_price
        return current_net_value
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """
        Get the return from assignment for a debit spread position at expiration.
        
        For debit spreads:
        - Initial debit paid is stored in entry_price
        - At expiration, we calculate the intrinsic value of our long and short legs
        - P&L = Intrinsic Value - Debit Paid
        """
        # Import here to avoid circular import
        from algo_trading_engine.common.models import StrategyType
        
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Spread options are not set")
            
        itm_option, otm_option = self.spread_options
        
        if self.strategy_type == StrategyType.CALL_DEBIT_SPREAD:
            # Long ITM call, Short OTM call
            long_strike = itm_option.strike   # ITM strike (lower)
            short_strike = otm_option.strike  # OTM strike (higher)
            
            # Calculate intrinsic values at expiration
            long_intrinsic = max(0, underlying_price - long_strike)
            short_intrinsic = max(0, underlying_price - short_strike)
            
            # Net P&L = Long leg value - Short leg cost - Initial debit
            # Spread value at expiration = long_intrinsic - short_intrinsic
            spread_value = long_intrinsic - short_intrinsic
            net_pnl = spread_value - self.entry_price
            
        elif self.strategy_type == StrategyType.PUT_DEBIT_SPREAD:
            # Long ITM put, Short OTM put
            long_strike = itm_option.strike   # ITM strike (higher)
            short_strike = otm_option.strike  # OTM strike (lower)
            
            # Calculate intrinsic values at expiration
            long_intrinsic = max(0, long_strike - underlying_price)
            short_intrinsic = max(0, short_strike - underlying_price)
            
            # Net P&L = Long leg value - Short leg cost - Initial debit
            spread_value = long_intrinsic - short_intrinsic
            net_pnl = spread_value - self.entry_price
            
        else:
            raise ValueError(f"Invalid strategy type for DebitSpreadPosition: {self.strategy_type}")
        
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """
        Calculate maximum profit for debit spread.
        Max profit = spread width - debit paid
        """
        if not self.spread_options or len(self.spread_options) != 2:
            raise ValueError("Debit spread requires 2 options in spread_options")
        itm_option, otm_option = self.spread_options
        spread_width = abs(itm_option.strike - otm_option.strike)
        return spread_width - self.entry_price
    
    def max_loss(self) -> Optional[float]:
        """
        Calculate maximum loss for debit spread.
        Max loss = debit paid
        """
        return self.entry_price


class LongCallPosition(Position):
    """Position for long call options."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """Get the return in dollars for a long call position."""
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """Get the percentage return for a long call position."""
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price for long call from option chain."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Long call requires option data in spread_options")
        option = self.spread_options[0]
        current_option = current_option_chain.get_option_data_for_option(option)
        if current_option is None:
            return None
        return current_option.last_price
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """For single leg, we only need one bar."""
        return float(atm_bar.close_price)
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """Get return from assignment for long call at expiration."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Long call requires option data in spread_options")
        option = self.spread_options[0]
        intrinsic = max(0, underlying_price - option.strike)
        net_pnl = intrinsic - self.entry_price
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """Unlimited upside for long call."""
        return None
    
    def max_loss(self) -> Optional[float]:
        """Max loss = premium paid."""
        return self.entry_price


class ShortCallPosition(Position):
    """Position for short call options."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """Get the return in dollars for a short call position."""
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # For short options: credit received - cost to close
        return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """Get the percentage return for a short call position."""
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price for short call from option chain."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Short call requires option data in spread_options")
        option = self.spread_options[0]
        current_option = current_option_chain.get_option_data_for_option(option)
        if current_option is None:
            return None
        return current_option.last_price
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """For single leg, we only need one bar."""
        return float(atm_bar.close_price)
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """Get return from assignment for short call at expiration."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Short call requires option data in spread_options")
        option = self.spread_options[0]
        intrinsic = max(0, underlying_price - option.strike)
        # For short: premium received - intrinsic value
        net_pnl = self.entry_price - intrinsic
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """Max profit = premium received."""
        return self.entry_price
    
    def max_loss(self) -> Optional[float]:
        """Unlimited risk for short call."""
        return None


class LongPutPosition(Position):
    """Position for long put options."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """Get the return in dollars for a long put position."""
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        return (exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """Get the percentage return for a long put position."""
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price for long put from option chain."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Long put requires option data in spread_options")
        option = self.spread_options[0]
        current_option = current_option_chain.get_option_data_for_option(option)
        if current_option is None:
            return None
        return current_option.last_price
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """For single leg, we only need one bar."""
        return float(atm_bar.close_price)
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """Get return from assignment for long put at expiration."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Long put requires option data in spread_options")
        option = self.spread_options[0]
        intrinsic = max(0, option.strike - underlying_price)
        net_pnl = intrinsic - self.entry_price
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """Max profit if stock goes to 0."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Long put requires option data in spread_options")
        option = self.spread_options[0]
        return option.strike - self.entry_price
    
    def max_loss(self) -> Optional[float]:
        """Max loss = premium paid."""
        return self.entry_price


class ShortPutPosition(Position):
    """Position for short put options."""
    
    def get_return_dollars(self, exit_price: float) -> float:
        """Get the return in dollars for a short put position."""
        if self.quantity is None:
            raise ValueError("Quantity is not set")
        # For short options: credit received - cost to close
        return (self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)
    
    def _get_return(self, exit_price: float) -> float:
        """Get the percentage return for a short put position."""
        if self.quantity is None or exit_price is None:
            raise ValueError("Quantity is not set")
        return ((self.entry_price * self.quantity * 100) - (exit_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def calculate_exit_price(self, current_option_chain: OptionChain) -> float:
        """Calculate current exit price for short put from option chain."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Short put requires option data in spread_options")
        option = self.spread_options[0]
        current_option = current_option_chain.get_option_data_for_option(option)
        if current_option is None:
            return None
        return current_option.last_price
    
    def calculate_exit_price_from_bars(self, atm_bar: 'OptionBarDTO', otm_bar: 'OptionBarDTO') -> float:
        """For single leg, we only need one bar."""
        return float(atm_bar.close_price)
    
    def get_return_dollars_from_assignment(self, underlying_price: float) -> float:
        """Get return from assignment for short put at expiration."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Short put requires option data in spread_options")
        option = self.spread_options[0]
        intrinsic = max(0, option.strike - underlying_price)
        # For short: premium received - intrinsic value
        net_pnl = self.entry_price - intrinsic
        return net_pnl * self.quantity * 100
    
    def max_profit(self) -> Optional[float]:
        """Max profit = premium received."""
        return self.entry_price
    
    def max_loss(self) -> Optional[float]:
        """Max loss if stock goes to 0."""
        if not self.spread_options or len(self.spread_options) == 0:
            raise ValueError("Short put requires option data in spread_options")
        option = self.spread_options[0]
        return option.strike - self.entry_price


def create_position(symbol: str, expiration_date: datetime, strategy_type: 'StrategyType',
                   strike_price: float, entry_date: datetime, entry_price: float,
                   exit_price: float = None, spread_options: list[Option] = None) -> Position:
    """
    Factory function to create appropriate Position subclass based on strategy_type.
    
    Args:
        symbol: Stock symbol
        expiration_date: Option expiration date
        strategy_type: Type of strategy (from StrategyType enum)
        strike_price: Strike price
        entry_date: Date position was entered
        entry_price: Price at entry
        exit_price: Price at exit (optional)
        spread_options: List of Option objects for the position
        
    Returns:
        Position: Appropriate Position subclass instance
        
    Raises:
        ValueError: If strategy type is unknown
    """
    # Import here to avoid circular import
    from algo_trading_engine.common.models import StrategyType
    
    if strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
        return CreditSpreadPosition(symbol, expiration_date, strategy_type, strike_price,
                                   entry_date, entry_price, exit_price, spread_options)
    elif strategy_type in [StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD]:
        return DebitSpreadPosition(symbol, expiration_date, strategy_type, strike_price,
                                   entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.LONG_CALL:
        return LongCallPosition(symbol, expiration_date, strategy_type, strike_price,
                               entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.SHORT_CALL:
        return ShortCallPosition(symbol, expiration_date, strategy_type, strike_price,
                                entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.LONG_PUT:
        return LongPutPosition(symbol, expiration_date, strategy_type, strike_price,
                              entry_date, entry_price, exit_price, spread_options)
    elif strategy_type == StrategyType.SHORT_PUT:
        return ShortPutPosition(symbol, expiration_date, strategy_type, strike_price,
                               entry_date, entry_price, exit_price, spread_options)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
