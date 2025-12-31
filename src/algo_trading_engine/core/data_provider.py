"""
Data provider protocol for dependency injection.

This module defines the protocol that data providers must implement,
allowing for flexible data source implementations.
"""

from typing import Protocol, Optional
from datetime import datetime
import pandas as pd

from algo_trading_engine.common.models import OptionChain


class DataProvider(Protocol):
    """
    Protocol for data providers.
    
    This allows dependency injection of different data sources
    (historical data, live data, mock data for testing, etc.)
    """
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        ...
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current/live price for a symbol.
        
        Used for paper trading to get real-time prices.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price as float
        """
        ...
    
    def get_option_chain(
        self,
        symbol: str,
        date: datetime
    ) -> Optional[OptionChain]:
        """
        Get options chain data for a symbol and date.
        
        Args:
            symbol: Stock symbol
            date: Date for options data
            
        Returns:
            OptionChain object or None if unavailable
        """
        ...
    
    def load_treasury_rates(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> None:
        """
        Load treasury rates for a date range.
        
        Args:
            start_date: Start date
            end_date: End date (optional, defaults to start_date)
        """
        ...

