"""
Strategy interface for trading strategies.

This module provides the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional, Dict, List, TYPE_CHECKING
import pandas as pd

from algo_trading_engine.common.models import TreasuryRates

if TYPE_CHECKING:
    from algo_trading_engine.backtest.models import Position

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

    def get_current_underlying_price(self, date: datetime, symbol: str) -> Optional[float]:
        """
        Get the current underlying price for a given date and symbol. Assigned via the TradingEngine.
        """
        raise NotImplementedError("get_current_underlying_price is not implemented in the base Strategy class.")

    def recommend_open_position(self, date: datetime, current_price: float) -> Optional[Dict]:
        """
        Recommend opening a position for the given date and current price.
        
        This method uses the strategy's on_new_date logic to determine if a position
        should be opened, then captures the position creation to return as a recommendation.
        
        This is a default implementation that works for any strategy by using on_new_date.
        Strategies can override this method to provide custom recommendation logic.
        
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
        # Import here to avoid circular dependency
        from algo_trading_engine.backtest.models import Position
        
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
            "expiration_date": recommended_position.expiration_date.strftime("%Y-%m-%d"),
        }

    def recommend_close_positions(self, date: datetime, positions: List['Position']) -> List[Dict]:
        """
        Recommend closing positions for the given date and current positions.
        
        This method uses the strategy's on_new_date logic to determine which positions
        should be closed, then captures the position closures to return as recommendations.
        
        This is a default implementation that works for any strategy by using on_new_date.
        Strategies can override this method to provide custom recommendation logic.
        
        Args:
            date: Current date
            positions: List of current open positions
            
        Returns:
            List[Dict]: List of position closure recommendations with keys:
                - position: Position object to close
                - exit_price: float, exit price for the position
                - underlying_price: Optional[float], underlying price at closure
                - current_volumes: Optional[List[int]], current volumes for options
                - rationale: str, reason for closing
            Returns empty list if no positions should be closed.
        """
        # Import here to avoid circular dependency
        from algo_trading_engine.backtest.models import Position
        
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

