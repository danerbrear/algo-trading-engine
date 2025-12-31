"""
Strategy interface for trading strategies.

This module provides the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional
import pandas as pd

from algo_trading_engine.common.models import TreasuryRates


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies must inherit from this class and implement
    the required abstract methods.
    """

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

    @abstractmethod
    def on_new_date(
        self,
        date: datetime,
        positions: tuple['Position', ...],
        add_position: Callable[['Position'], None],
        remove_position: Callable[[datetime, 'Position', float, Optional[float], Optional[list[int]]], None]
    ) -> None:
        """
        Called for each trading day to execute strategy logic.
        
        Args:
            date: Current trading date
            positions: Tuple of currently open positions
            add_position: Callback function to add a new position
            remove_position: Callback function to remove/close a position
                Signature: (date, position, exit_price, underlying_price=None, current_volumes=None)
        """
        pass

    @abstractmethod
    def on_end(
        self,
        positions: tuple['Position', ...],
        remove_position: Callable[[datetime, 'Position', float, Optional[float], Optional[list[int]]], None],
        date: datetime
    ) -> None:
        """
        Called at the end of backtest/paper trading to close remaining positions.
        
        Args:
            positions: Tuple of currently open positions
            remove_position: Callback function to remove/close a position
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

