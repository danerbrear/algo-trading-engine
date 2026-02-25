"""
Refactored OptionsHandler with clean caching architecture.

This module implements the new OptionsHandler following the requirements
in features/improved_data_fetching.md Phase 2.
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from decimal import Decimal
from dotenv import load_dotenv

from .cache.options_cache_manager import OptionsCacheManager
from algo_trading_engine.dto import (
    OptionContractDTO, OptionBarDTO, StrikeRangeDTO, ExpirationRangeDTO,
    OptionsChainDTO
)
from algo_trading_engine.vo import StrikePrice, ExpirationDate
from algo_trading_engine.enums import BarTimeInterval
from .models import OptionType
from ..ml_models.api_retry_handler import APIRetryHandler
from .progress_tracker import progress_print

# Load environment variables
load_dotenv()


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
    
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache', use_free_tier: bool = False):
        """Initialize the OptionsHandler."""
        if not symbol or not symbol.strip():
            raise ValueError("Symbol is required and cannot be empty")
        
        self.symbol = symbol.upper()
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.use_free_tier = use_free_tier
        
        if not self.api_key:
            raise ValueError("Polygon.io API key is required")
        
        self.cache_manager = OptionsCacheManager(cache_dir)
        
        # Initialize API client and retry handler
        from polygon import RESTClient
        self.client = RESTClient(self.api_key)
        self.api_retry_handler = APIRetryHandler(use_rate_limit=use_free_tier)
    
    def __getattribute__(self, name):
        """
        Enforce private method access restrictions.
        
        Prevents external access to private cache manipulation methods.
        """
        # List of private methods that should not be accessed externally
        private_methods = {
            '_cache_contracts', '_cache_bar', '_get_cache_stats',
            '_cached_contracts_satisfy_criteria', '_merge_contracts',
            '_apply_contract_filters', '_fetch_contracts_from_api',
            '_fetch_bar_from_api', '_convert_api_contract_to_dto',
            '_convert_api_bar_to_dto'
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
                
                # Check if this is an internal call by looking at the calling frame
                # If the calling frame is from the same class, allow access
                if frame and frame.f_back:
                    calling_frame = frame.f_back
                    if hasattr(calling_frame, 'f_code') and hasattr(calling_frame.f_code, 'co_filename'):
                        calling_filename = calling_frame.f_code.co_filename
                        # If the calling frame is from the same file, it's an internal call
                        if 'options_handler.py' in calling_filename:
                            return super().__getattribute__(name)
                
                # Not in a test or internal call, block external access
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
        
        This method implements additive caching:
        1. First tries to load from cache
        2. If cached contracts don't satisfy filtering criteria, fetches additional contracts from API
        3. Merges and caches the combined set
        4. Returns filtered results
        
        Args:
            date: The date to get contracts for
            strike_range: Optional strike price filtering
            expiration_range: Optional expiration date filtering
            
        Returns:
            List of OptionContractDTO objects
            
        Raises:
            TypeError: If date is not a datetime or date object
        """
        from datetime import date as date_class
        if not isinstance(date, (datetime, date_class)):
            raise TypeError(f"date must be a datetime or date object, got {type(date).__name__}")
        
        date_obj = date.date() if isinstance(date, datetime) else date
        
        # Try to load from cache first (with graceful error handling)
        cached_contracts = None
        try:
            cached_contracts = self.cache_manager.load_contracts(self.symbol, date_obj)
        except Exception as e:
            print(f"⚠️  Cache loading failed for {self.symbol} on {date_obj}: {e}. Falling back to API...")
            cached_contracts = None
        
        if cached_contracts:
            progress_print(f"📁 Loaded {len(cached_contracts)} contracts from cache for {self.symbol} on {date_obj}")
            
            # Apply filtering to cached contracts
            filtered_cached = self._apply_contract_filters(cached_contracts, strike_range, expiration_range, date_obj)
            
            # Check if cached contracts satisfy the filtering criteria
            if self._cached_contracts_satisfy_criteria(filtered_cached, strike_range, expiration_range, date_obj):
                progress_print(f"✅ Cached contracts satisfy filtering criteria, returning {len(filtered_cached)} contracts")
                return filtered_cached
            else:
                progress_print(f"⚠️  Cached contracts don't fully satisfy filtering criteria, fetching additional contracts from API...")
        else:
            progress_print(f"🔄 No cached contracts found for {self.symbol} on {date_obj}, fetching from API...")
            filtered_cached = []
        
        # Fetch additional contracts from API
        api_contracts = self._fetch_contracts_from_api(date_obj, strike_range, expiration_range)
        
        if api_contracts:
            # Merge cached and API contracts (remove duplicates by ticker)
            all_contracts = self._merge_contracts(cached_contracts, api_contracts)
            
            # Cache the merged contracts (with graceful error handling)
            try:
                self._cache_contracts(date, all_contracts)
            except Exception as e:
                print(f"⚠️  Cache saving failed for {self.symbol} on {date_obj}: {e}. Continuing without caching...")
            
            # Apply filters to merged contracts before returning
            filtered_all = self._apply_contract_filters(all_contracts, strike_range, expiration_range, date_obj)
            
            cached_count = len(cached_contracts) if cached_contracts else 0
            progress_print(f"✅ Merged {cached_count} cached + {len(api_contracts)} API contracts, returning {len(filtered_all)} contracts after filtering")
            return filtered_all
        
        # If no API contracts, return what we have from cache
        return filtered_cached
    
    def get_option_bar(
        self,
        contract: OptionContractDTO,
        date: datetime,
        multiplier: int = 1,
        timespan: BarTimeInterval = BarTimeInterval.DAY,
    ) -> Optional[OptionBarDTO]:
        """
        Get bar data for a specific option contract.

        Args:
            contract: The option contract
            date: The date (and for hourly/minute, the time) to get bar data for. For daily bars
                only the date part is used. For hourly/minute bars, pass a datetime with the
                desired bar time (e.g. datetime(2025, 2, 10, 9, 30) for the 9:30 bar).
            multiplier: Bar multiplier (default: 1)
            timespan: Bar time interval (default: BarTimeInterval.DAY). Mapped to Polygon API timespan string.

        Returns:
            OptionBarDTO if found, None otherwise
        """
        date_obj = date.date() if isinstance(date, datetime) else date

        # Try to load from cache first (with graceful error handling)
        cached_bar = None
        try:
            # Validate contract has ticker attribute
            if not hasattr(contract, 'ticker'):
                return None
            cached_bar = self.cache_manager.load_bar(
                self.symbol, date, contract.ticker, timespan.value
            )
        except Exception as e:
            ticker_str = getattr(contract, 'ticker', 'unknown')
            print(f"⚠️  Cache loading failed for {ticker_str} on {date_obj}: {e}. Falling back to API...")
            cached_bar = None

        if cached_bar:
            return cached_bar

        # If not in cache, fetch from API (map enum to Polygon timespan string)
        if not hasattr(contract, 'ticker'):
            return None

        progress_print(f"🔄 No cached bar data found for {contract.ticker} on {date_obj}, fetching from API...")
        timespan_str = timespan.value  # Polygon API expects "minute", "hour", "day"
        dt = date if isinstance(date, datetime) else datetime.combine(date, datetime.min.time())
        bar = self._fetch_bar_from_api(contract, dt, multiplier, timespan_str)
        
        if bar:
            # Cache the fetched bar data (with graceful error handling)
            try:
                self._cache_bar(date, contract.ticker, bar, timespan)
            except Exception as e:
                print(f"⚠️  Cache saving failed for {contract.ticker} on {date_obj}: {e}. Continuing without caching...")
            return bar
        
        progress_print(f"⚠️  No bar data received from API for {contract.ticker} on {date_obj}")
        return None
    
    def get_options_chain(
        self, 
        date: datetime, 
        current_price: float,
        strike_range: Optional[StrikeRangeDTO] = None,
        expiration_range: Optional[ExpirationRangeDTO] = None,
        timespan: BarTimeInterval = BarTimeInterval.DAY,
        multiplier: int = 1,
    ) -> OptionsChainDTO:
        """
        Get complete option chain for a date.
        
        Args:
            date: The date to get option chain for
            current_price: Current underlying price
            strike_range: Optional strike price filtering
            expiration_range: Optional expiration date filtering
            timespan: Bar time interval for option bars (default: DAY). Pass HOUR for intraday.
            multiplier: Bar multiplier (default: 1). Used with timespan for bar size.
            
        Returns:
            OptionsChainDTO with contracts and bars
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        
        # Get contracts
        contracts = self.get_contract_list_for_date(date, strike_range, expiration_range)
        
        # Get bars for all contracts
        bars = {}
        for contract in contracts:
            bar = self.get_option_bar(contract, date, multiplier=multiplier, timespan=timespan)
            if bar:
                bars[contract.ticker] = bar
        
        return OptionsChainDTO(
            underlying_symbol=self.symbol,
            current_price=current_price,
            date=date_obj,
            contracts=contracts,
            bars=bars
        )
    
    def _cached_contracts_satisfy_criteria(
        self, 
        filtered_contracts: List[OptionContractDTO],
        strike_range: Optional[StrikeRangeDTO],
        expiration_range: Optional[ExpirationRangeDTO],
        current_date: date
    ) -> bool:
        """
        Check if cached contracts satisfy the filtering criteria.
        
        Returns True if:
        - No filters are specified (return all cached contracts)
        - All specified filters are satisfied by cached contracts
        """
        if not strike_range and not expiration_range:
            # No filters specified, cached contracts are sufficient
            return True
        
        if not filtered_contracts:
            # No contracts match the criteria, need to fetch more
            return False
        
        # Check if we have contracts in the requested ranges
        if strike_range:
            strikes_in_range = [c for c in filtered_contracts if strike_range.contains_strike(c.strike_price)]
            if not strikes_in_range:
                return False
        
        if expiration_range:
            expirations_in_range = [c for c in filtered_contracts if expiration_range.contains_expiration(c.expiration_date, current_date)]
            if not expirations_in_range:
                return False
        
        return True
    
    def _merge_contracts(
        self, 
        cached_contracts: Optional[List[OptionContractDTO]], 
        api_contracts: List[OptionContractDTO]
    ) -> List[OptionContractDTO]:
        """
        Merge cached and API contracts, removing duplicates based on business logic.
        
        Uses the OptionContractDTO.__eq__ method to determine duplicates,
        which considers contracts equal if they have:
        - Same ticker, OR
        - Same expiration date, strike price, and option type
        
        Args:
            cached_contracts: Contracts from cache (can be None if no cache)
            api_contracts: Contracts from API
            
        Returns:
            Merged list of unique contracts
        """
        # Handle case where there are no cached contracts
        if not cached_contracts:
            progress_print(f"🔄 No cached contracts, returning {len(api_contracts)} API contracts")
            return api_contracts
        
        # Start with cached contracts
        merged_contracts = list(cached_contracts)
        
        # Add API contracts that are not duplicates
        for api_contract in api_contracts:
            is_duplicate = any(api_contract == cached_contract for cached_contract in cached_contracts)
            if not is_duplicate:
                merged_contracts.append(api_contract)
        
        progress_print(f"🔄 Merged contracts: {len(cached_contracts)} cached + {len(api_contracts)} API = {len(merged_contracts)} total (removed {len(cached_contracts) + len(api_contracts) - len(merged_contracts)} duplicates)")
        
        return merged_contracts

    def _cache_contracts(self, date: datetime, contracts: List[OptionContractDTO]) -> None:
        """
        Cache contracts for a specific date.
        
        This is a private method for internal use only.
        External callers should not directly manipulate the cache.
        """
        date_obj = date.date() if isinstance(date, datetime) else date
        try:
            self.cache_manager.save_contracts(self.symbol, date_obj, contracts)
        except Exception as e:
            print(f"⚠️  Cache saving failed for {self.symbol} on {date_obj}: {e}. Continuing without caching...")
    
    def _cache_bar(
        self,
        date: datetime,
        ticker: str,
        bar: OptionBarDTO,
        timespan: BarTimeInterval = BarTimeInterval.DAY,
    ) -> None:
        """
        Cache bar data for a specific option under the given timespan.

        This is a private method for internal use only.
        External callers should not directly manipulate the cache.
        """
        try:
            self.cache_manager.save_bar(
                self.symbol, date, ticker, bar, timespan.value
            )
        except Exception as e:
            date_obj = date.date() if isinstance(date, datetime) else date
            print(f"⚠️  Cache saving failed for {ticker} on {date_obj}: {e}. Continuing without caching...")
    
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
    
    def _apply_contract_filters(
        self, 
        contracts: List[OptionContractDTO], 
        strike_range: Optional[StrikeRangeDTO],
        expiration_range: Optional[ExpirationRangeDTO],
        current_date: date
    ) -> List[OptionContractDTO]:
        """Apply filtering to contracts based on provided criteria."""
        filtered_contracts = contracts
        
        if strike_range:
            filtered_contracts = [c for c in filtered_contracts if strike_range.contains_strike(c.strike_price)]
        
        if expiration_range:
            filtered_contracts = [c for c in filtered_contracts if expiration_range.contains_expiration(c.expiration_date, current_date)]
        
        return filtered_contracts
    
    def _fetch_contracts_from_api(self, date_obj: date, strike_range: Optional[StrikeRangeDTO] = None, expiration_range: Optional[ExpirationRangeDTO] = None) -> List[OptionContractDTO]:
        """Fetch option contracts from Polygon.io API."""
        try:
            # Check if date is in the future
            from datetime import date as date_class
            today = date_class.today()
            if date_obj > today:
                progress_print(f"⚠️  Cannot fetch contracts for future date {date_obj} (today is {today})")
                return []
            def fetch_func():
                # Build parameters for API call
                params = {}
                
                # Add strike price filters if provided
                if strike_range:
                    if strike_range.min_strike:
                        params["strike_price.gte"] = float(strike_range.min_strike.value)
                    if strike_range.max_strike:
                        params["strike_price.lte"] = float(strike_range.max_strike.value)
                
                # Add expiration date filters if provided
                if expiration_range:
                    if expiration_range.min_days is not None:
                        min_expiry = date_obj + timedelta(days=expiration_range.min_days)
                        params["expiration_date.gte"] = min_expiry.strftime('%Y-%m-%d')
                    if expiration_range.max_days is not None:
                        max_expiry = date_obj + timedelta(days=expiration_range.max_days)
                        params["expiration_date.lte"] = max_expiry.strftime('%Y-%m-%d')
                
                # Use the list options contracts API with date filtering
                contracts_response = self.client.list_options_contracts(
                    underlying_ticker=self.symbol,
                    as_of=date_obj.strftime('%Y-%m-%d'),
                    expired=False,
                    limit=1000,  # Maximum allowed by API
                    params=params if params else None
                )
                return contracts_response
            
            response = self.api_retry_handler.fetch_with_retry(
                fetch_func, 
                f"Failed to fetch contracts for {self.symbol} on {date_obj}"
            )
            
            if not response:
                progress_print(f"⚠️  No contracts data received from API for {self.symbol} on {date_obj}")
                return []
            
            contracts = []
            for contract_data in response:
                try:
                    # Convert API response to OptionContractDTO
                    contract_dto = self._convert_api_contract_to_dto(contract_data)
                    if contract_dto:
                        contracts.append(contract_dto)
                except Exception as e:
                    progress_print(f"⚠️  Error converting contract {contract_data.get('ticker', 'unknown')}: {e}")
                    continue
            
            progress_print(f"✅ Fetched {len(contracts)} contracts from API for {self.symbol} on {date_obj}")
            return contracts
            
        except Exception as e:
            progress_print(f"❌ Error fetching contracts from API: {e}")
            return []
    
    def _fetch_bar_from_api(
        self,
        contract: OptionContractDTO,
        dt: datetime,
        multiplier: int,
        timespan: str,  # Polygon API: "minute", "hour", "day" (from BarTimeInterval.value)
    ) -> Optional[OptionBarDTO]:
        """Fetch option bar data from Polygon.io API. Uses datetime only; formats from/to by interval."""
        try:
            from datetime import date as date_class
            today = date_class.today()
            if dt.date() > today:
                progress_print(f"⚠️  Cannot fetch bar data for future date {dt.date()} (today is {today})")
                return None

            if timespan == "day":
                from_str = dt.strftime("%Y-%m-%d")
                to_str = from_str
            elif timespan == "hour":
                # Polygon uses ET for US options. Request full trading day (9:30-16:00 ET) to avoid
                # narrow-window gaps; then filter for the bar containing our target time.
                et = ZoneInfo("America/New_York")
                dt_et = dt.replace(tzinfo=et) if dt.tzinfo is None else dt.astimezone(et)
                d = dt_et.date()
                market_open = datetime(d.year, d.month, d.day, 9, 30, 0, tzinfo=et)
                market_close = datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=et)
                from_ts = int(market_open.timestamp() * 1000)
                to_ts = int(market_close.timestamp() * 1000)
                from_str, to_str = from_ts, to_ts
                bar_start = dt_et.replace(minute=0, second=0, microsecond=0)
                target_ts = int(bar_start.timestamp() * 1000)
            elif timespan == "minute":
                bar_start = dt
                bar_end = bar_start + timedelta(minutes=1)
                from_str = bar_start.strftime("%Y-%m-%dT%H:%M:%S")
                to_str = bar_end.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                from_str = dt.strftime("%Y-%m-%d")
                to_str = from_str

            def fetch_func():
                limit_val = 5000 if timespan == "hour" else 1
                bars_response = self.client.get_aggs(
                    ticker=contract.ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_str,
                    to=to_str,
                    limit=limit_val
                )
                bars_list = list(bars_response)
                progress_print(f"✅ Fetched {len(bars_list)} bars from API for {contract.ticker} on {dt.date()}")
                return bars_list

            response = self.api_retry_handler.fetch_with_retry(
                fetch_func,
                f"Error fetching bar data for {contract.ticker} on {dt.date()}"
            )

            if not response or len(response) == 0:
                progress_print(f"⚠️  No bar data received from API for {contract.ticker} on {dt.date()}")
                return None

            if timespan == "hour":
                def _bar_ts(b):
                    return getattr(b, "timestamp", None) or getattr(b, "t", None)
                matching = [b for b in response if _bar_ts(b) == target_ts]
                if not matching:
                    hour_ms = 60 * 60 * 1000
                    matching = [b for b in response if _bar_ts(b) and target_ts <= _bar_ts(b) < target_ts + hour_ms]
                if not matching:
                    matching = [b for b in response if _bar_ts(b) is not None]
                bar_data = matching[0] if matching else response[0]
            else:
                bar_data = response[0]
            bar_dto = self._convert_api_bar_to_dto(contract.ticker, bar_data, dt.date())

            if bar_dto:
                progress_print(f"✅ Fetched bar data from API for {contract.ticker} on {dt.date()}")

            return bar_dto

        except Exception as e:
            progress_print(f"❌ Error fetching bar data from API for {contract.ticker} on {dt.date()}: {e}")
            return None
    
    def _convert_api_contract_to_dto(self, contract_data) -> Optional[OptionContractDTO]:
        """Convert API contract response to OptionContractDTO."""
        try:
            # Handle both OptionsContract objects and dictionaries
            if hasattr(contract_data, 'ticker'):
                # OptionsContract object
                ticker = contract_data.ticker
                underlying_ticker = contract_data.underlying_ticker
                contract_type_str = contract_data.contract_type
                strike_price = contract_data.strike_price
                expiration_date_str = contract_data.expiration_date
                exercise_style = getattr(contract_data, 'exercise_style', 'american')
                shares_per_contract = getattr(contract_data, 'shares_per_contract', 100)
                primary_exchange = getattr(contract_data, 'primary_exchange', None)
                cfi = getattr(contract_data, 'cfi', None)
                additional_underlyings = getattr(contract_data, 'additional_underlyings', None)
            else:
                # Dictionary format
                ticker = contract_data.get('ticker')
                underlying_ticker = contract_data.get('underlying_ticker')
                contract_type_str = contract_data.get('contract_type')
                strike_price = contract_data.get('strike_price')
                expiration_date_str = contract_data.get('expiration_date')
                exercise_style = contract_data.get('exercise_style', 'american')
                shares_per_contract = contract_data.get('shares_per_contract', 100)
                primary_exchange = contract_data.get('primary_exchange')
                cfi = contract_data.get('cfi')
                additional_underlyings = contract_data.get('additional_underlyings')
            
            if not all([ticker, underlying_ticker, contract_type_str, strike_price, expiration_date_str]):
                return None
            
            # Convert contract type
            contract_type = OptionType.CALL if contract_type_str.lower() == 'call' else OptionType.PUT
            
            # Create value objects
            strike = StrikePrice(Decimal(str(strike_price)))
            exp_date = ExpirationDate(datetime.strptime(expiration_date_str, '%Y-%m-%d').date())
            
            return OptionContractDTO(
                ticker=ticker,
                underlying_ticker=underlying_ticker,
                contract_type=contract_type,
                strike_price=strike,
                expiration_date=exp_date,
                exercise_style=exercise_style,
                shares_per_contract=shares_per_contract,
                primary_exchange=primary_exchange,
                cfi=cfi,
                additional_underlyings=additional_underlyings
            )
            
        except Exception as e:
            progress_print(f"⚠️  Error converting contract data: {e}")
            return None
    
    def _convert_api_bar_to_dto(self, ticker: str, bar_data, date_obj: date) -> Optional[OptionBarDTO]:
        """Convert API bar response to OptionBarDTO."""
        try:
            # Handle both dictionary and Agg object formats
            if hasattr(bar_data, 'get'):
                # Dictionary format
                open_price = bar_data.get('o')
                high_price = bar_data.get('h')
                low_price = bar_data.get('l')
                close_price = bar_data.get('c')
                volume = bar_data.get('v', 0)
                vwap = bar_data.get('vw', close_price)
                transactions = bar_data.get('n', 1)
                timestamp_ms = bar_data.get('t')
            else:
                # Agg object format - access attributes directly
                open_price = getattr(bar_data, 'open', None)
                high_price = getattr(bar_data, 'high', None)
                low_price = getattr(bar_data, 'low', None)
                close_price = getattr(bar_data, 'close', None)
                volume = getattr(bar_data, 'volume', 0)
                vwap = getattr(bar_data, 'vwap', close_price)
                transactions = getattr(bar_data, 'transactions', 1)
                timestamp_ms = getattr(bar_data, 'timestamp', None)
            
            if not all([open_price is not None, high_price is not None, low_price is not None, close_price is not None]):
                return None
            
            # Convert timestamp
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000) if timestamp_ms else datetime.combine(date_obj, datetime.min.time())
            
            return OptionBarDTO(
                ticker=ticker,
                timestamp=timestamp,
                open_price=Decimal(str(open_price)),
                high_price=Decimal(str(high_price)),
                low_price=Decimal(str(low_price)),
                close_price=Decimal(str(close_price)),
                volume=int(volume),
                volume_weighted_avg_price=Decimal(str(vwap)),
                number_of_transactions=int(transactions),
                adjusted=True
            )
            
        except Exception as e:
            progress_print(f"⚠️  Error converting bar data: {e}")
            return None
