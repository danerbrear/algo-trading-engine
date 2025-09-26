"""
Options Cache Manager for the refactored OptionsHandler.

This module provides the OptionsCacheManager class for managing the new
caching structure with contracts.pkl and bars/ subdirectory.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
from datetime import date

from .cache_manager import CacheManager
from ..options_dtos import OptionContractDTO, OptionBarDTO


class OptionsCacheManager(CacheManager):
    """
    Manages caching operations for the refactored OptionsHandler.
    
    This extends CacheManager and implements the new caching structure:
    data_cache/options/{symbol}/
    ├── {date}/
    │   ├── contracts.pkl          # All contracts for the date
    │   └── bars/
    │       └── {ticker}.pkl       # Individual option bar data
    """
    
    def __init__(self, base_dir: str = 'data_cache'):
        super().__init__(base_dir)
    
    def get_contracts_cache_path(self, symbol: str, date: date) -> Path:
        """Get the path for contracts cache file."""
        return self.get_cache_path('contracts.pkl', 'options', symbol, date.strftime('%Y-%m-%d'))
    
    def get_bars_cache_path(self, symbol: str, date: date, ticker: str) -> Path:
        """Get the path for individual option bar cache file."""
        return self.get_cache_path(f'{ticker}.pkl', 'options', symbol, date.strftime('%Y-%m-%d'), 'bars')
    
    def get_date_cache_dir(self, symbol: str, date: date) -> Path:
        """Get the cache directory for a specific date."""
        return self.get_cache_dir('options', symbol, date.strftime('%Y-%m-%d'))
    
    def get_bars_cache_dir(self, symbol: str, date: date) -> Path:
        """Get the bars cache directory for a specific date."""
        return self.get_cache_dir('options', symbol, date.strftime('%Y-%m-%d'), 'bars')
    
    def load_contracts(self, symbol: str, date: date) -> Optional[List[OptionContractDTO]]:
        """Load contracts from cache."""
        cache_path = self.get_contracts_cache_path(symbol, date)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading contracts cache: {e}")
        return None
    
    def save_contracts(self, symbol: str, date: date, contracts: List[OptionContractDTO]) -> None:
        """Save contracts to cache."""
        cache_path = self.get_contracts_cache_path(symbol, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(contracts, f)
    
    def load_bar(self, symbol: str, date: date, ticker: str) -> Optional[OptionBarDTO]:
        """Load individual option bar from cache."""
        cache_path = self.get_bars_cache_path(symbol, date, ticker)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading bar cache: {e}")
        return None
    
    def save_bar(self, symbol: str, date: date, ticker: str, bar: OptionBarDTO) -> None:
        """Save individual option bar to cache."""
        cache_path = self.get_bars_cache_path(symbol, date, ticker)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(bar, f)
    
    def load_bars_for_date(self, symbol: str, date: date) -> Dict[str, OptionBarDTO]:
        """Load all bars for a specific date."""
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
        """Check if contract exists in cache."""
        contracts = self.load_contracts(symbol, date)
        if contracts:
            return any(contract.ticker == ticker for contract in contracts)
        return False
    
    def bar_exists(self, symbol: str, date: date, ticker: str) -> bool:
        """Check if bar data exists in cache."""
        return self.get_bars_cache_path(symbol, date, ticker).exists()
    
    def get_cached_contracts_count(self, symbol: str, date: date) -> int:
        """Get count of cached contracts for a date."""
        contracts = self.load_contracts(symbol, date)
        return len(contracts) if contracts else 0
    
    def get_cached_bars_count(self, symbol: str, date: date) -> int:
        """Get count of cached bars for a date."""
        bars_dir = self.get_bars_cache_dir(symbol, date)
        if bars_dir.exists():
            return len(list(bars_dir.glob('*.pkl')))
        return 0
    
    def list_available_dates(self, symbol: str) -> List[date]:
        """List all available cached dates for a symbol."""
        symbol_dir = self.base_dir / 'options' / symbol
        dates = []
        
        if symbol_dir.exists():
            for date_dir in symbol_dir.iterdir():
                if date_dir.is_dir():
                    try:
                        date_obj = date.fromisoformat(date_dir.name)
                        dates.append(date_obj)
                    except ValueError:
                        continue
        
        return sorted(dates)
