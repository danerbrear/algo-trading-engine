"""
Strategy interface for trading strategies.

This module provides the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional, Dict, List, TYPE_CHECKING
import pandas as pd

from algo_trading_engine.common.models import TreasuryRates
from algo_trading_engine.core.indicators.indicator import Indicator

if TYPE_CHECKING:
    from algo_trading_engine.vo import Position
    from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO, OptionsChainDTO, ExpirationRangeDTO, StrikeRangeDTO

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies must inherit from this class and implement
    the required abstract methods.
    
    Options Trading Callables:
        Strategies that trade options can expect the following callables to be available
        as instance attributes (set via strategy builder):
        
        - get_contract_list_for_date: Get list of option contracts for a specific date and symbol
        - get_option_bar: Get bar data for a specific option contract on a specific date
        - get_options_chain: Get the full options chain for a symbol on a specific date
        - get_current_volumes_for_position: Get current volumes for an open position
        - compute_exit_price: Compute the exit price for a position on a specific date
    """
    
    # Optional callables for options trading strategies
    # These are set via the strategy builder and available for use in concrete strategies
    get_contract_list_for_date: Optional[Callable[[datetime, str], List['OptionContractDTO']]] = None
    get_option_bar: Optional[Callable[['OptionContractDTO', datetime], Optional['OptionBarDTO']]] = None
    get_options_chain: Optional[Callable[[str, datetime, Optional['ExpirationRangeDTO'], Optional['StrikeRangeDTO']], 'OptionsChainDTO']] = None
    get_current_volumes_for_position: Optional[Callable[['Position'], Optional[List[int]]]] = None
    compute_exit_price: Optional[Callable[['Position', datetime], Optional[float]]] = None

    def __init__(self, profit_target: float = None, stop_loss: float = None, start_date_offset: int = 0):
        """
        Initialize the strategy.
        
        Args:
            profit_target: Optional profit target percentage (e.g., 0.5 for 50%)
            stop_loss: Optional stop loss percentage (e.g., 0.6 for 60%)
            start_date_offset: Number of days to skip at the beginning for warm-up
        """
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.data: Optional[pd.DataFrame] = None
        self.start_date_offset = start_date_offset
        self.treasury_data: Optional[TreasuryRates] = None
        self.indicators: List[Indicator] = []

    @abstractmethod
    def on_new_date(
        self,
        date: datetime,
        positions: tuple['Position', ...],
        add_position: Callable[['Position', Optional[Callable[[], None]]], None],
        remove_position: Callable[[datetime, 'Position', float, Optional[float], Optional[list[int]], Optional[Callable[[], None]]], None]
    ) -> None:
        """
        Called for each trading day to execute strategy logic.
        
        Args:
            date: Current trading date
            positions: Tuple of currently open positions
            add_position: Callback to add a new position. Signature: (position, on_add_callback=None)
            remove_position: Callback to remove/close a position.
                Signature: (date, position, exit_price, underlying_price=None, current_volumes=None, on_remove_callback=None)
        """
        if not self._update_indicators(date):
            print(f"Error updating indicators for date {date}, skipping execution")
            return

    @abstractmethod
    def on_end(
        self,
        positions: tuple['Position', ...],
        remove_position: Callable[[datetime, 'Position', float, Optional[float], Optional[list[int]], Optional[Callable[[], None]]], None],
        date: datetime
    ) -> None:
        """
        Called at the end of backtest/paper trading to close remaining positions.
        
        Args:
            positions: Tuple of currently open positions
            remove_position: Callback to remove/close a position.
                Signature: (date, position, exit_price, underlying_price=None, current_volumes=None, on_remove_callback=None)
            date: Final date
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the provided data meets the strategy's requirements.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            True if data is valid, False otherwise
        """
        pass

    def add_indicator(self, indicator: Indicator) -> None:
        """
        Add an indicator to the strategy.
        
        This allows indicators to be added after strategy initialization,
        which is useful for dynamic indicator configuration.
        
        Args:
            indicator: Indicator instance to add
            
        Example:
            strategy = MyStrategy()
            atr = ATRIndicator(period=14)
            strategy.add_indicator(atr)
        """
        if not isinstance(indicator, Indicator):
            raise TypeError(f"Expected Indicator instance, got {type(indicator).__name__}")
        self.indicators.append(indicator)

    def get_indicator(self, indicator_class: type) -> Optional[Indicator]:
        """
        Get an indicator by class type.
        
        Args:
            indicator_class: The indicator class to search for (e.g., ATRIndicator)
            
        Returns:
            Indicator instance if found, None otherwise
            
        Example:
            atr = self.get_indicator(ATRIndicator)
            if atr:
                current_atr = atr.value
        """
        for indicator in self.indicators:
            if isinstance(indicator, indicator_class):
                return indicator
        return None

    def get_current_underlying_price(self, date: datetime, symbol: str) -> Optional[float]:
        """
        Get the current underlying price for a given date and symbol. Assigned via the TradingEngine.
        """
        raise NotImplementedError("get_current_underlying_price is not implemented in the base Strategy class.")

    def set_data(self, data: pd.DataFrame, treasury_data: Optional[TreasuryRates] = None):
        """
        Set the market data for the strategy.
        
        Args:
            data: DataFrame with market data
            treasury_data: Optional treasury rates data
        """
        self.data = data
        self.treasury_data = treasury_data

    def set_profit_target(self, profit_target: float):
        """Set the profit target for the strategy."""
        self.profit_target = profit_target

    def set_stop_loss(self, stop_loss: float):
        """Set the stop loss for the strategy."""
        self.stop_loss = stop_loss

    def _profit_target_hit(self, position: 'Position', exit_price: float) -> bool:
        """
        Check if the profit target has been hit for a position.
        
        Args:
            position: Position to check
            exit_price: Current exit price
            
        Returns:
            True if profit target hit, False otherwise
        """
        if self.profit_target is None:
            return False
        return position.profit_target_hit(self.profit_target, exit_price)

    def _stop_loss_hit(self, position: 'Position', exit_price: float) -> bool:
        """
        Check if the stop loss has been hit for a position.
        
        Args:
            position: Position to check
            exit_price: Current exit price
            
        Returns:
            True if stop loss hit, False otherwise
        """
        if self.stop_loss is None:
            return False
        return position.stop_loss_hit(self.stop_loss, exit_price)

    def _update_indicators(self, date: datetime) -> bool:
        """
        Update the indicators for the strategy.
        
        Args:
            date: Current date

        Returns:
            True if indicators were updated, False otherwise
        """
        for indicator in self.indicators:
            try:
                indicator.update(date, self.data)
            except Exception as e:
                print(f"Error updating indicator {indicator.name}: {e}")
                return False
        print(f"Indicators updated successfully for date {date}")
        return True
