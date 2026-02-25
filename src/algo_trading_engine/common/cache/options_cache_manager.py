"""
Options Cache Manager for the refactored OptionsHandler.

This module provides the OptionsCacheManager class for managing the new
caching structure with contracts.pkl and bars/ subdirectory. Bar interval
directory names match the stocks cache: daily, hourly, minute.

- Daily: use the date from the datetime to get the daily bar.
- Hourly: use only the :30 half-hour marker. The cache key is the last :30 that has
  already happened (e.g. 11:02 -> 10:30, 10:37 -> 10:30; we never use :00).
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import date, datetime, time, timedelta

from .cache_manager import CacheManager
from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO

# Map API/enum timespan ("day"|"hour"|"minute") to cache dir name, matching stocks cache layout
_TIMESPAN_TO_INTERVAL_DIR: Dict[str, str] = {
    "day": "daily",
    "hour": "hourly",
    "minute": "minute",
}


def _interval_dir(timespan: str) -> str:
    """Return cache subdirectory name for timespan (matches data_retriever stocks layout)."""
    return _TIMESPAN_TO_INTERVAL_DIR.get(timespan, timespan)


def _floor_to_previous_half_hour(dt: datetime) -> Tuple[date, time]:
    """Return (date, time) for the last :30 half-hour that has already happened. E.g. 11:02 -> 10:30, 10:37 -> 10:30. We only use :30 markers, not :00."""
    if dt.minute < 30:
        if dt.hour == 0:
            return (dt.date() - timedelta(days=1), time(23, 30))
        return (dt.date(), time(dt.hour - 1, 30))
    return (dt.date(), time(dt.hour, 30))


def _bar_time_suffix(bar_time: Union[datetime, time]) -> str:
    """Return HHMM suffix for cache filename (e.g. 0930, 1030)."""
    if isinstance(bar_time, datetime):
        return bar_time.strftime("%H%M")
    return bar_time.strftime("%H%M")


class OptionsCacheManager(CacheManager):
    """
    Manages caching operations for the refactored OptionsHandler.
    
    Bar cache layout mirrors stocks cache (data_cache/stocks/{symbol}/{daily|hourly|minute}/):
    data_cache/options/{symbol}/
    ├── {date}/
    │   ├── contracts.pkl          # All contracts for the date
    │   └── bars/
    │       ├── daily/             # One bar per contract per date (use date from datetime)
    │       │   └── {ticker}.pkl
    │       ├── hourly/            # Only :30 markers; key is last :30 that has happened (e.g. 11:02 -> 10:30)
    │       │   └── {ticker}_0930.pkl, {ticker}_1030.pkl, ...
    │       └── minute/            # One bar per contract per date per minute
    │           └── {ticker}_0930.pkl, ...
    """
    
    def __init__(self, base_dir: str = 'data_cache'):
        super().__init__(base_dir)
    
    def get_contracts_cache_path(self, symbol: str, date: date) -> Path:
        """Get the path for contracts cache file."""
        return self.get_cache_path('contracts.pkl', 'options', symbol, date.strftime('%Y-%m-%d'))
    
    def get_bars_cache_path(
        self,
        symbol: str,
        date: Union[date, datetime],
        ticker: str,
        timespan: str = "day",
        bar_time: Optional[Union[datetime, time]] = None,
    ) -> Path:
        """Get the path for an option bar cache file. Daily: use date only. Hourly: use last :30 that has happened (e.g. 11:02 -> 10:30)."""
        if timespan == "hour" and bar_time is None and isinstance(date, datetime):
            date_obj, bar_time = _floor_to_previous_half_hour(date)
        else:
            date_obj = date.date() if isinstance(date, datetime) else date
            if timespan == "minute" and bar_time is None and isinstance(date, datetime):
                bar_time = date.time()
        clean_ticker = ticker[2:] if ticker.startswith('O:') else ticker
        interval_dir = _interval_dir(timespan)
        if timespan in ("hour", "minute"):
            if bar_time is None:
                raise ValueError("bar_time is required for hourly and minute bars (or pass a datetime as date)")
            suffix = _bar_time_suffix(bar_time)
            filename = f"{clean_ticker}_{suffix}.pkl"
        else:
            filename = f"{clean_ticker}.pkl"
        return self.get_cache_path(filename, 'options', symbol, date_obj.strftime('%Y-%m-%d'), 'bars', interval_dir)
    
    def get_date_cache_dir(self, symbol: str, date: date) -> Path:
        """Get the cache directory for a specific date."""
        return self.get_cache_dir('options', symbol, date.strftime('%Y-%m-%d'))
    
    def get_bars_cache_dir(self, symbol: str, date: date, timespan: str = "day") -> Path:
        """Get the bars cache directory for a specific date and interval (daily|hourly|minute)."""
        interval_dir = _interval_dir(timespan)
        return self.get_cache_dir('options', symbol, date.strftime('%Y-%m-%d'), 'bars', interval_dir)
    
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
    
    def load_bar(
        self,
        symbol: str,
        date: Union[date, datetime],
        ticker: str,
        timespan: str = "day",
    ) -> Optional[OptionBarDTO]:
        """Load individual option bar from cache. Daily: date part only. Hourly: last :30 that has happened (e.g. 11:02 -> 10:30)."""
        bar_time = date.time() if timespan == "minute" and isinstance(date, datetime) else None
        cache_path = self.get_bars_cache_path(symbol, date, ticker, timespan, bar_time)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading bar cache: {e}")
        return None
    
    def save_bar(
        self, symbol: str, date: Union[date, datetime], ticker: str, bar: OptionBarDTO, timespan: str = "day"
    ) -> None:
        """Save individual option bar to cache. For hourly, use request date for path (so lookup key matches)."""
        if timespan == "hour" and isinstance(date, datetime):
            date_obj, bar_time = _floor_to_previous_half_hour(date)
            cache_path = self.get_bars_cache_path(symbol, date_obj, ticker, timespan, bar_time)
        elif timespan == "hour" and bar.timestamp:
            date_obj, bar_time = _floor_to_previous_half_hour(bar.timestamp)
            cache_path = self.get_bars_cache_path(symbol, date_obj, ticker, timespan, bar_time)
        else:
            bar_time = bar.timestamp if timespan == "minute" and bar.timestamp else None
            date_obj = date.date() if isinstance(date, datetime) else date
            cache_path = self.get_bars_cache_path(symbol, date_obj, ticker, timespan, bar_time)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(bar, f)
    
    def load_bars_for_date(
        self, symbol: str, date: date, timespan: str = "day"
    ) -> Dict[str, OptionBarDTO]:
        """Load all bars for a date. Daily: one bar per ticker. Hourly/minute: one bar per ticker (last/only bar per ticker in cache)."""
        bars_dir = self.get_bars_cache_dir(symbol, date, timespan)
        bars: Dict[str, OptionBarDTO] = {}
        if not bars_dir.exists():
            return bars
        if timespan == "day":
            for bar_file in bars_dir.glob('*.pkl'):
                if bar_file.stem.count('_') == 0:
                    clean_ticker = bar_file.stem
                    full_ticker = f'O:{clean_ticker}' if not clean_ticker.startswith('O:') else clean_ticker
                    bar = self.load_bar(symbol, date, full_ticker, timespan)
                    if bar:
                        bars[full_ticker] = bar
        else:
            for bar_file in bars_dir.glob('*_*.pkl'):
                stem = bar_file.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and len(parts[1]) == 4:
                    clean_ticker, time_str = parts[0], parts[1]
                    full_ticker = f'O:{clean_ticker}' if not clean_ticker.startswith('O:') else clean_ticker
                    bar_dt = datetime.combine(date, time(int(time_str[:2]), int(time_str[2:])))
                    bar = self.load_bar(symbol, bar_dt, full_ticker, timespan)
                    if bar:
                        bars[f"{full_ticker}_{time_str}"] = bar
        return bars
    
    def contract_exists(self, symbol: str, date: date, ticker: str) -> bool:
        """Check if contract exists in cache."""
        contracts = self.load_contracts(symbol, date)
        if contracts:
            return any(contract.ticker == ticker for contract in contracts)
        return False
    
    def bar_exists(
        self,
        symbol: str,
        date: Union[date, datetime],
        ticker: str,
        timespan: str = "day",
    ) -> bool:
        """Check if bar data exists in cache. Hourly: last :30 that has happened."""
        try:
            bar_time = date.time() if timespan == "minute" and isinstance(date, datetime) else None
            return self.get_bars_cache_path(symbol, date, ticker, timespan, bar_time).exists()
        except ValueError:
            return False
    
    def get_cached_contracts_count(self, symbol: str, date: date) -> int:
        """Get count of cached contracts for a date."""
        contracts = self.load_contracts(symbol, date)
        return len(contracts) if contracts else 0
    
    def get_cached_bars_count(self, symbol: str, date: date, timespan: str = "day") -> int:
        """Get count of cached bars for a date and timespan."""
        bars_dir = self.get_bars_cache_dir(symbol, date, timespan)
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
