"""
Refactored OptionsHandler with clean caching architecture.

This module implements the new OptionsHandler following the requirements
in features/improved_data_fetching.md Phase 2.
"""

import os
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from dotenv import load_dotenv

from .cache.cache_manager import CacheManager
from .options_dtos import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    OptionsChainDTO, StrikePrice, ExpirationDate
)
from .models import OptionType

# Load environment variables
load_dotenv()


class OptionsCacheManager:
    """
    Manages the new caching structure for options data.
    
    Target Structure:
    data_cache/options/{symbol}/
    ├── {date}/
    │   ├── contracts.pkl          # All contracts for the date
    │   └── bars/
    │       └── {ticker}.pkl       # Individual option bar data
    """
    
    def __init__(self, base_dir: str = 'data_cache'):
        self.cache_manager = CacheManager(base_dir)
        self.base_dir = Path(base_dir)
    
    def get_contracts_cache_path(self, symbol: str, date: date) -> Path:
        """Get the path for contracts cache file."""
        return self.base_dir / 'options' / symbol / date.strftime('%Y-%m-%d') / 'contracts.pkl'
    
    def get_bars_cache_path(self, symbol: str, date: date, ticker: str) -> Path:
        """Get the path for individual option bar cache file."""
        return self.base_dir / 'options' / symbol / date.strftime('%Y-%m-%d') / 'bars' / f'{ticker}.pkl'
    
    def get_date_cache_dir(self, symbol: str, date: date) -> Path:
        """Get the cache directory for a specific date."""
        return self.base_dir / 'options' / symbol / date.strftime('%Y-%m-%d')
    
    def get_bars_cache_dir(self, symbol: str, date: date) -> Path:
        """Get the bars cache directory for a specific date."""
        return self.get_date_cache_dir(symbol, date) / 'bars'
    
    def load_contracts(self, symbol: str, date: date) -> Optional[List[OptionContractDTO]]:
        """Load contracts for a specific date."""
        cache_path = self.get_contracts_cache_path(symbol, date)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading contracts cache: {e}")
                return None
        return None
    
    def save_contracts(self, symbol: str, date: date, contracts: List[OptionContractDTO]) -> None:
        """Save contracts for a specific date."""
        cache_path = self.get_contracts_cache_path(symbol, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(contracts, f)
        except Exception as e:
            print(f"Error saving contracts cache: {e}")
    
    def load_bar(self, symbol: str, date: date, ticker: str) -> Optional[OptionBarDTO]:
        """Load bar data for a specific option ticker."""
        cache_path = self.get_bars_cache_path(symbol, date, ticker)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading bar cache: {e}")
                return None
        return None
    
    def save_bar(self, symbol: str, date: date, ticker: str, bar: OptionBarDTO) -> None:
        """Save bar data for a specific option ticker."""
        cache_path = self.get_bars_cache_path(symbol, date, ticker)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(bar, f)
        except Exception as e:
            print(f"Error saving bar cache: {e}")
    
    def load_bars_for_date(self, symbol: str, date: date) -> Dict[str, OptionBarDTO]:
        """Load all bar data for a specific date."""
        bars_dir = self.get_bars_cache_dir(symbol, date)
        bars = {}
        
        if bars_dir.exists():
            for bar_file in bars_dir.glob('*.pkl'):
                ticker = bar_file.stem
                bar = self.load_bar(symbol, date, ticker)
                if bar:
                    bars[ticker] = bar
        
        return bars
    
    def contract_exists(self, symbol: str, date: date, ticker: str) -> bool:
        """Check if a contract exists in cache."""
        contracts = self.load_contracts(symbol, date)
        if contracts:
            return any(contract.ticker == ticker for contract in contracts)
        return False
    
    def bar_exists(self, symbol: str, date: date, ticker: str) -> bool:
        """Check if bar data exists in cache."""
        return self.get_bars_cache_path(symbol, date, ticker).exists()
    
    def get_cached_contracts_count(self, symbol: str, date: date) -> int:
        """Get the number of cached contracts for a date."""
        contracts = self.load_contracts(symbol, date)
        return len(contracts) if contracts else 0
    
    def get_cached_bars_count(self, symbol: str, date: date) -> int:
        """Get the number of cached bars for a date."""
        bars_dir = self.get_bars_cache_dir(symbol, date)
        if bars_dir.exists():
            return len(list(bars_dir.glob('*.pkl')))
        return 0
    
    def list_available_dates(self, symbol: str) -> List[date]:
        """List all available dates for a symbol."""
        symbol_dir = self.base_dir / 'options' / symbol
        dates = []
        
        if symbol_dir.exists():
            for date_dir in symbol_dir.iterdir():
                if date_dir.is_dir():
                    try:
                        date_obj = datetime.strptime(date_dir.name, '%Y-%m-%d').date()
                        dates.append(date_obj)
                    except ValueError:
                        continue
        
        return sorted(dates)

class OptionsHandler:
    """
    Refactored OptionsHandler with clean API and efficient caching.
    
    This implements the new architecture specified in features/improved_data_fetching.md
    with simplified caching, clean public API, and proper DTOs.
    
    Private Method Enforcement:
    - Methods prefixed with '_' are private and should not be accessed externally
    - Private methods are enforced via __getattribute__ to prevent external access
    - Access is allowed during testing to enable proper test setup
    - Use public API methods instead of private cache manipulation methods
    """
    
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache'):
        """Initialize the OptionsHandler."""
        self.symbol = symbol.upper()
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        
        if not self.api_key:
            raise ValueError("Polygon.io API key is required")
        
        self.cache_manager = OptionsCacheManager(cache_dir)
        
        # TODO: Initialize API client when implementing Phase 3
        # self.client = RESTClient(self.api_key)
        # self.retry_handler = APIRetryHandler()
    
    def __getattribute__(self, name):
        """
        Enforce private method access restrictions.
        
        Prevents external access to private cache manipulation methods.
        """
        # List of private methods that should not be accessed externally
        private_methods = {
            '_cache_contracts', '_cache_bar', '_get_cache_stats'
        }
        
        # Check if the requested attribute is a private method
        if name in private_methods:
            # Allow access during testing (when called from test files)
            import inspect
            frame = inspect.currentframe()
            try:
                # Check the call stack to see if we're in a test
                current_frame = frame
                while current_frame:
                    filename = current_frame.f_code.co_filename
                    if 'test_' in filename or 'pytest' in filename:
                        # We're in a test, allow access
                        return super().__getattribute__(name)
                    current_frame = current_frame.f_back
                
                # Not in a test, block external access
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                    f"This is a private method and should not be accessed externally. "
                    f"Use the public API methods instead."
                )
            finally:
                del frame
        
        # For all other attributes, use the default behavior
        return super().__getattribute__(name)
    
    def get_contract_list_for_date(
        self, 
        date: datetime, 
        strike_range: Optional[StrikeRangeDTO] = None,
        expiration_range: Optional[ExpirationRangeDTO] = None
    ) -> List[OptionContractDTO]:
        """
        Get all option contracts for a specific date with optional filtering.
        
        Args:
            date: The date to get contracts for
            strike_range: Optional strike price filtering
            expiration_range: Optional expiration date filtering
            
        Returns:
            List of OptionContractDTO objects
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        
        # Load contracts from cache
        contracts = self.cache_manager.load_contracts(self.symbol, date_obj)
        
        if contracts:
            # Apply filters if provided
            if strike_range:
                contracts = [c for c in contracts if strike_range.contains_strike(c.strike_price)]
            
            if expiration_range:
                contracts = [c for c in contracts if expiration_range.contains_expiration(c.expiration_date)]
        else:
            # TODO: Implement API fetching in Phase 3
            print(f"No cached contracts found for {self.symbol} on {date_obj}")
            return []
        
        return contracts
    
    def get_option_bar(
        self, 
        contract: OptionContractDTO, 
        date: datetime,
        multiplier: int = 1,
        timespan: str = "day"
    ) -> Optional[OptionBarDTO]:
        """
        Get bar data for a specific option contract.
        
        Args:
            contract: The option contract
            date: The date to get bar data for
            multiplier: Bar multiplier (default: 1)
            timespan: Bar timespan (default: "day")
            
        Returns:
            OptionBarDTO if found, None otherwise
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        
        # Load bar from cache
        bar = self.cache_manager.load_bar(self.symbol, date_obj, contract.ticker)
        
        if not bar:
            # TODO: Implement API fetching in Phase 3
            print(f"No cached bar data found for {contract.ticker} on {date_obj}")
            return None
        
        return bar
    
    def get_options_chain(
        self, 
        date: datetime, 
        current_price: float,
        strike_range: Optional[StrikeRangeDTO] = None,
        expiration_range: Optional[ExpirationRangeDTO] = None
    ) -> OptionsChainDTO:
        """
        Get complete option chain for a date.
        
        Args:
            date: The date to get option chain for
            current_price: Current underlying price
            strike_range: Optional strike price filtering
            expiration_range: Optional expiration date filtering
            
        Returns:
            OptionsChainDTO with contracts and bars
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        
        # Get contracts
        contracts = self.get_contract_list_for_date(date, strike_range, expiration_range)
        
        # Get bars for all contracts
        bars = {}
        for contract in contracts:
            bar = self.get_option_bar(contract, date)
            if bar:
                bars[contract.ticker] = bar
        
        return OptionsChainDTO(
            underlying_symbol=self.symbol,
            current_price=current_price,
            date=date_obj,
            contracts=contracts,
            bars=bars
        )
    
    def _cache_contracts(self, date: datetime, contracts: List[OptionContractDTO]) -> None:
        """
        Cache contracts for a specific date.
        
        This is a private method for internal use only.
        External callers should not directly manipulate the cache.
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        self.cache_manager.save_contracts(self.symbol, date_obj, contracts)
    
    def _cache_bar(self, date: datetime, ticker: str, bar: OptionBarDTO) -> None:
        """
        Cache bar data for a specific option.
        
        This is a private method for internal use only.
        External callers should not directly manipulate the cache.
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        self.cache_manager.save_bar(self.symbol, date_obj, ticker, bar)
    
    def _get_cache_stats(self, date: datetime) -> Dict[str, int]:
        """
        Get cache statistics for a specific date.
        
        This is a private method for internal use only.
        External callers should not directly access cache statistics.
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        
        return {
            'contracts_count': self.cache_manager.get_cached_contracts_count(self.symbol, date_obj),
            'bars_count': self.cache_manager.get_cached_bars_count(self.symbol, date_obj)
        }
    
    def list_available_dates(self) -> List[date]:
        """List all available cached dates for this symbol."""
        return self.cache_manager.list_available_dates(self.symbol)
