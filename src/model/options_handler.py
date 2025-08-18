import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
from polygon import RESTClient
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
try:
    from common.cache.cache_manager import CacheManager
except ImportError:
    # Fallback for direct script execution
    from src.common.cache.cache_manager import CacheManager
from .api_retry_handler import APIRetryHandler
import sys
import os
# Add the project root to the path to import progress_tracker
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.common.progress_tracker import ProgressTracker, set_global_progress_tracker, progress_print, is_quiet_mode
from ..common.models import OptionChain, OptionType, Option

# Load environment variables from .env file
load_dotenv()

# Global variable to track API calls
total_api_calls_made = 0

class OptionsHandler:
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache', start_date: str = None, use_free_tier: bool = False, quiet_mode: bool = True):
        """Initialize the OptionsHandler with a symbol and optional Polygon.io API key"""
        self.symbol = symbol
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.quiet_mode = quiet_mode
        
        if not self.api_key:
            raise ValueError("Polygon.io API key is required.")
            
        self.cache_manager = CacheManager(cache_dir)
        self.client = RESTClient(self.api_key)
        self.retry_handler = APIRetryHandler(use_rate_limit=use_free_tier)
        # Convert start_date to naive datetime for consistent comparison
        self.start_date = pd.Timestamp(start_date).tz_localize(None) if start_date else None
        
    def _fetch_historical_contract_data(self, contract, current_date: datetime) -> Optional[Option]:
        """Fetch historical data for a specific contract with retries"""
        def fetch_func():
            # Show detailed messages only when not in quiet mode
            if not is_quiet_mode():
                print(f"Fetching historical data for {contract.__str__()}")
            try:
                aggs_response = self.client.get_aggs(
                    ticker=contract.ticker,
                    multiplier=1,
                    timespan="day",
                    from_=current_date.strftime('%Y-%m-%d'),
                    to=current_date.strftime('%Y-%m-%d'),
                    limit=1
                )
                
                # Safely handle the generator response
                aggs = []
               
                if aggs_response:
                    try:
                        aggs = list(aggs_response)
                    except Exception as e:
                        error_str = str(e)
                        progress_print(f"Error converting generator to list for {contract.__str__()}: {error_str}")
                        
                        # Check for authorization errors that indicate plan limitations
                        if "NOT_AUTHORIZED" in error_str or "doesn't include this data timeframe" in error_str:
                            progress_print(f"⚠️ Authorization error for {current_date.date()}: Plan doesn't cover this timeframe")
                            raise ValueError("SKIP_DATE_UNAUTHORIZED")
                        return None
                        
                if not aggs:
                    if not is_quiet_mode():
                        print(f"No aggregate data available for {contract.ticker}")
                    return None
                    
                day_data = aggs[0]
                
                # Enhanced data structure to support Polygon.io's available fields
                # Note: For historical data, Greeks/IV would need Options Snapshot API
                return Option.from_dict({
                    'ticker': contract.ticker,
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date,
                    'type': 'call' if contract.contract_type.lower() == 'call' else 'put',
                    'symbol': contract.ticker,
                    'volume': day_data.volume if hasattr(day_data, 'volume') else 0,
                    'open_interest': None,  # Available in Options Snapshot API (Starter+ plans)
                    'implied_volatility': None,  # Available in Options Snapshot API (Starter+ plans)
                    'delta': None,
                    'gamma': None,
                    'theta': None,
                    'vega': None,
                    'last_price': day_data.close,
                    'bid': day_data.low,  # Real bid/ask available with Advanced plan ($199/month)
                    'ask': day_data.high,  # Real bid/ask available with Advanced plan ($199/month)
                    'mid_price': (day_data.low + day_data.high) / 2,
                    'moneyness': day_data.close / float(contract.strike_price),
                })
            except ValueError as e:
                if "SKIP_DATE_UNAUTHORIZED" in str(e):
                    raise  # Re-raise to be caught by the outer handler
                progress_print(f"ValueError in fetch_func for {contract.__str__()}: {str(e)}")
                return None
            except Exception as e:
                error_str = str(e)
                progress_print(f"Error in fetch_func for {contract.__str__()}: {error_str}")
                
                # Check for authorization errors in general exceptions too
                if "NOT_AUTHORIZED" in error_str or "doesn't include this data timeframe" in error_str:
                    progress_print(f"⚠️ Authorization error for {current_date.date()}: Plan doesn't cover this timeframe")
                    raise ValueError("SKIP_DATE_UNAUTHORIZED")
                return None
            
        return self.retry_handler.fetch_with_retry(
            fetch_func,
            f"Error fetching historical data for {contract.__str__()}"
        )

    def get_specific_option_contract(self, target_strike: float, expiry: str, option_type: str, current_date: datetime) -> Optional[Option]:
        """
        Get a specific option contract for a given date, expiration, and strike price.
        First checks cache, then fetches from API if not available.
        
        Args:
            target_strike: The strike price of the option
            expiry: The expiration date (YYYY-MM-DD format)
            option_type: The option type ('call' or 'put')
            current_date: The date for which to get the option data
            
        Returns:
            Optional[Option]: Option contract data if found, None otherwise
        """
        progress_print(f"Getting specific option contract for {target_strike} {expiry} {option_type} as of {current_date}")
        # Ensure current_date is naive datetime
        current_date_naive = pd.Timestamp(current_date).tz_localize(None)

        # First, try to get from cache
        cached_chain_data = self.cache_manager.load_date_from_cache(
            current_date_naive,
            '',  # No suffix for main chain data
            'options',
            self.symbol
        )

        if cached_chain_data:
            # Look for the specific option in cached data
            options_list = cached_chain_data.get('calls' if option_type.lower() == 'call' else 'puts', [])
            
            for option_data in options_list:
                if (abs(option_data.get('strike', 0) - target_strike) < 0.01 and 
                    option_data.get('expiration') == expiry and
                    option_data.get('type', '').lower() == option_type.lower()):
                    
                    progress_print(f"✅ Found {option_type} ${target_strike:.0f} exp {expiry} in cache")
                    return Option.from_dict(option_data)

        # If not in cache, fetch from API
        progress_print(f"🔍 Fetching {option_type} ${target_strike:.0f} exp {expiry} from API")

        try:
            # Fetch contracts for this specific option
            contracts = self._fetch_contracts_for_strike(
                target_strike=target_strike,
                expiry=expiry,
                option_type=option_type,
                current_date=current_date_naive
            )

            if contracts:
                # Find the closest contract to our target strike
                closest_contract = min(contracts, 
                                     key=lambda x: abs(float(x.strike_price) - target_strike))
                
                # Check if the closest contract is within acceptable range (within $5)
                if abs(float(closest_contract.strike_price) - target_strike) == 0.0:
                    # Fetch historical data for this contract
                    option_data = self._fetch_historical_contract_data(closest_contract, current_date_naive)
                    
                    if option_data:
                        # Update cache with the new option data
                        self._update_cache_with_specific_option(option_data, current_date_naive)
                        
                        progress_print(f"✅ Successfully fetched {option_type} ${float(closest_contract.strike_price):.0f} @ ${option_data.last_price:.2f}")
                        return option_data
                    else:
                        progress_print(f"❌ No historical data available for {option_type} ${float(closest_contract.strike_price):.0f}")
                else:
                    progress_print(f"⚠️ No suitable {option_type} found near ${target_strike:.0f} (closest: ${float(closest_contract.strike_price):.0f})")
            else:
                progress_print(f"❌ No contracts found for {option_type} ${target_strike:.0f}")
                
        except Exception as e:
            progress_print(f"❌ Error fetching {option_type} ${target_strike:.0f}: {str(e)}")
        
        return None

    def _update_cache_with_specific_option(self, option_data: Option, current_date: datetime):
        """
        Update the cache with a specific option contract.
        
        Args:
            option_data: The option data to add to cache
            current_date: The date for the cache entry
        """
        try:
            # Load existing cache data
            existing_cache = self.cache_manager.load_date_from_cache(
                current_date,
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            
            if existing_cache is None:
                # Create new cache entry
                existing_cache = {'calls': [], 'puts': []}
            
            # Add the new option data to the appropriate list
            if option_data.option_type == OptionType.CALL:
                existing_cache['calls'].append(option_data.to_dict())
            else:
                existing_cache['puts'].append(option_data.to_dict())
            
            # Save updated cache
            self.cache_manager.save_date_to_cache(
                current_date,
                existing_cache,
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            
            progress_print(f"💾 Updated cache with {option_data.option_type.value} ${option_data.strike:.0f}")
            
        except Exception as e:
            progress_print(f"⚠️ Warning: Could not update cache: {str(e)}")

    def _fetch_historical_contracts_data(self, contracts: List, current_date: datetime, current_price: float) -> OptionChain:
        """Fetch historical data for a list of contracts"""
        chain_data = {'calls': [], 'puts': []}
        
        if not contracts:
            return chain_data
            
        try:
            # Get target expiry (closest to 30 days)
            target_date = current_date + timedelta(days=30)
            expiry_dates = set()
            
            for contract in contracts:
                if hasattr(contract, 'expiration_date'):
                    expiry_dates.add(contract.expiration_date)
            
            if not expiry_dates:
                print(f"No expiration dates available for {current_date.date()}")
                return OptionChain.from_dict_w_options(chain_data)
                
            # Convert to datetime objects for comparison
            expiry_dates = [pd.Timestamp(exp).tz_localize(None) for exp in expiry_dates]
            
            # Find closest expiry to target date
            target_expiry = min(expiry_dates, key=lambda x: abs((x - target_date).days))
            target_expiry_str = target_expiry.strftime('%Y-%m-%d')
            
            progress_print(f"Closest expiry: {target_expiry_str}")
            
            # Filter for target expiry
            target_expiry_contracts = [
                c for c in contracts 
                if hasattr(c, 'expiration_date') and 
                pd.Timestamp(c.expiration_date).tz_localize(None) == target_expiry
            ]
            
            if not target_expiry_contracts:
                progress_print(f"No contracts found for target expiry {target_expiry_str}")
                return OptionChain.from_dict_w_options(chain_data)
            
            # Step 3: Separate calls and puts, sorted by proximity to current price
            calls = [c for c in target_expiry_contracts if hasattr(c, 'contract_type') and c.contract_type.lower() == 'call']
            puts = [c for c in target_expiry_contracts if hasattr(c, 'contract_type') and c.contract_type.lower() == 'put']
            
            # Sort by distance from current price (closest first)
            calls_sorted = sorted(calls, key=lambda x: abs(float(x.strike_price) - current_price))
            puts_sorted = sorted(puts, key=lambda x: abs(float(x.strike_price) - current_price))
            
            progress_print(f"Found {len(calls_sorted)} calls and {len(puts_sorted)} puts for target expiry")
            
            # Process calls
            for call in calls_sorted[:5]:  # Limit to 5 closest calls
                try:
                    call_data = self._fetch_historical_contract_data(call, current_date)
                    if call_data:
                        chain_data['calls'].append(call_data)
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        progress_print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                        return OptionChain.from_dict_w_options(chain_data)
                    raise
                    
            # Process puts
            for put in puts_sorted[:5]:  # Limit to 5 closest puts
                try:
                    put_data = self._fetch_historical_contract_data(put, current_date)
                    if put_data:
                        chain_data['puts'].append(put_data)
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        progress_print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                        return OptionChain.from_dict_w_options(chain_data)
                    raise
                    
        except Exception as e:
            print(f"Error in _fetch_historical_contracts_data: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
        return OptionChain.from_dict_w_options(chain_data)

    def _get_contracts_from_cache(self, current_date: datetime, current_price: float) -> Optional[List]:
        """Get the list of contracts from cache if available, otherwise fetch from API"""
        try:
            # Try to get from cache first
            contracts = self.cache_manager.load_date_from_cache(
                current_date, 
                '_contracts',
                'options',
                self.symbol
            )
            
            if contracts is not None:
                progress_print(f"Loading cached contracts list for {current_date.date()}")
                return contracts
                    
            # If not in cache, fetch from API with retries
            progress_print(f"Fetching contracts for {current_date.date()}")
            
            def fetch_func():
                try:                    
                    # Calculate strike price range (7% around current price)
                    price_range = current_price * 0.07  # 7% range for broader coverage
                    min_strike = current_price - price_range
                    max_strike = current_price + price_range
                    
                    # Calculate expiration date range (focus on options around 30-day target)
                    min_expiry = current_date + timedelta(days=24)  # Minimum 24 days out (closer to 30-day target)
                    max_expiry = current_date + timedelta(days=36)  # Maximum 36 days out
                    
                    progress_print(f"Filtering contracts: strikes ${min_strike:.0f}-${max_strike:.0f}, expiry {min_expiry.strftime('%Y-%m-%d')} to {max_expiry.strftime('%Y-%m-%d')} (targeting ~30 days)")
                    
                    contracts_response = self.client.list_options_contracts(
                        underlying_ticker=self.symbol,
                        as_of=current_date.strftime('%Y-%m-%d'),
                        params={
                            "strike_price.gte": min_strike,
                            "strike_price.lte": max_strike,
                            "expiration_date.gte": min_expiry.strftime('%Y-%m-%d'),
                            "expiration_date.lte": max_expiry.strftime('%Y-%m-%d')
                        },
                        expired=False,
                        limit=200  # Limit results per page to reduce pagination
                    )
                    
                    contracts = []
                    page_count = 0
                    max_pages = 2  # Limit to 5 pages to stay under rate limits
                    
                    if contracts_response:
                        try:
                            for contract in contracts_response:
                                contracts.append(contract)
                                
                                # Check if we've reached a page boundary (every 250 items)
                                if len(contracts) % 200 == 0:
                                    page_count += 1
                                    progress_print(f"Completed page {page_count}, got {len(contracts)} contracts so far")
                                    
                                    # Stop if we've reached max pages to avoid rate limits
                                    if page_count >= max_pages:
                                        progress_print(f"Reached max pages ({max_pages}), stopping to avoid rate limits")
                                        break
                                    
                                    # Rate limit between pages
                                    if self.retry_handler.use_rate_limit:
                                        progress_print("Rate limiting: waiting 13 seconds between paginated requests")
                                        time.sleep(13)
                                    
                        except Exception as e:
                            print(f"Error iterating through contracts: {str(e)}")
                            print(f"Error type: {type(e)}")
                            import traceback
                            traceback.print_exc()
                            return []
                    
                    if contracts:
                        print(f"Caching {len(contracts)} contracts for {current_date.date()}")
                        self.cache_manager.save_date_to_cache(
                            current_date,
                            contracts,
                            '_contracts',
                            'options',
                            self.symbol
                        )
                    else:
                        progress_print(f"No contracts available for {current_date.date()}")

                    return contracts

                except Exception as e:
                    print(f"Error in fetch_func for contracts: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    return []
                
            return self.retry_handler.fetch_with_retry(
                fetch_func,
                f"Error fetching contracts for {current_date.date()}"
            )
            
        except Exception as e:
            print(f"Error in _get_contracts_from_cache for {current_date}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _get_option_chain_with_cache(self, current_date: datetime, current_price: float) -> OptionChain:
        """Get option chain data for a date, using cache if available"""
        try:
            # Try to get from cache first
            chain_data = self.cache_manager.load_date_from_cache(
                current_date,
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            
            if chain_data is not None:
                return OptionChain.from_dict(chain_data)

            chain_data = OptionChain.from_dict({'calls': [], 'puts': []})

            try:
                # Get contracts (will use cache if available)
                contracts = self._get_contracts_from_cache(current_date, current_price)
                
                if not contracts:
                    progress_print(f"No options data available for {current_date.date()}")
                    return chain_data
                
                # Fetch historical data for all contracts
                chain_data = self._fetch_historical_contracts_data(contracts, current_date, current_price)
                
                # Cache the data if we got any contracts
                if chain_data.calls or chain_data.puts:
                    print(f"Caching option chain data for {current_date.date()}")
                    self.cache_manager.save_date_to_cache(
                        current_date,
                        chain_data.to_dict(),
                        '',  # No suffix for main chain data
                        'options',
                        self.symbol
                    )
                
            except Exception as e:
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            chain_data = OptionChain.from_dict({'calls': [], 'puts': []})
            
        return chain_data
        
    def _get_target_expiry(self, current_date: datetime, chain_data: OptionChain) -> Optional[str]:
        """Get the target expiration date closest to 30 days out"""
        if not chain_data.calls and not chain_data.puts:
            return None
            
        target_date = current_date + timedelta(days=30)
        
        # Get unique expiration dates
        expiry_dates = set()
        expiry_dates.update(opt.expiration for opt in chain_data.puts)
        expiry_dates.update(opt.expiration for opt in chain_data.calls)
            
        if not expiry_dates:
            print(f"No expiration dates available for {current_date.date()}")
            return None
            
        # Convert to datetime objects for comparison
        expiry_dates = [pd.Timestamp(exp).tz_localize(None) for exp in expiry_dates]
        
        # Find closest expiry to target date
        closest_expiry = min(expiry_dates, key=lambda x: abs((x - target_date).days))
        progress_print(f"Closest expiry: {closest_expiry.strftime('%Y-%m-%d')}")
        return closest_expiry.strftime('%Y-%m-%d')
        
    def _get_atm_options(self, current_price: float, chain_data: Dict, expiry: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get at-the-money call and put options for a given price and expiration"""
        if not chain_data['calls'] or not chain_data['puts']:
            return None, None
            
        # Filter for target expiry
        calls = [opt for opt in chain_data['calls'] if opt['expiration'] == expiry]
        puts = [opt for opt in chain_data['puts'] if opt['expiration'] == expiry]
        
        if not calls or not puts:
            return None, None
            
        # Find closest strike to current price
        calls = sorted(calls, key=lambda x: abs(x['strike'] - current_price))
        puts = sorted(puts, key=lambda x: abs(x['strike'] - current_price))
        
        atm_call = calls[0]
        atm_put = puts[0]
        
        # Validate minimum requirements
        if (atm_call['volume'] < 10 or atm_put['volume'] < 10 or
            atm_call['last_price'] < 0.10 or atm_put['last_price'] < 0.10):
            return None, None
            
        return atm_call, atm_put
            
    def _get_strategy_option_strikes(self, current_price: float, chain_data: OptionChain, expiry: str, current_date: datetime) -> Dict:
        """Get multiple option strikes needed for realistic strategy calculations
        If strikes are missing, fetch them from Polygon API and update cache"""
        option_strikes = {}
        
        # Filter for target expiry
        calls = [opt for opt in chain_data.calls if opt.expiration == expiry]
        puts = [opt for opt in chain_data.puts if opt.expiration == expiry]
        
        if not calls or not puts:
            return option_strikes
        
        # Sort by strike price
        calls_sorted = sorted(calls, key=lambda x: x.strike)
        puts_sorted = sorted(puts, key=lambda x: x.strike)
        
        # Define target strikes for strategies
        target_strikes = {
            'atm': current_price,                    # ATM for straddles
            'call_atm_plus5': current_price + 5,     # For call debit spreads
            'call_atm_plus10': current_price + 10,   # For iron butterfly protection calls
            'put_atm_minus5': current_price - 5,     # For put debit spreads  
            'put_atm_minus10': current_price - 10,   # For iron butterfly protection puts
        }
        
        # Track missing strikes that need to be fetched
        missing_strikes = []
        
        # Find closest options to each target strike
        for strike_name, target_strike in target_strikes.items():
            if 'call' in strike_name or strike_name == 'atm':
                # Look for calls
                closest_call = min(calls_sorted, 
                                 key=lambda x: abs(x.strike - target_strike), 
                                 default=None)
                if closest_call and abs(closest_call.strike - target_strike) <= 5.0:  # Within $5
                    if strike_name == 'atm':
                        option_strikes['atm_call'] = closest_call
                    else:
                        option_strikes[strike_name] = closest_call
                else:
                    # Check if this strike is cached as missing before making API call
                    if not self._is_strike_cached_as_missing(strike_name, target_strike, 'call', current_date):
                        missing_strikes.append((strike_name, target_strike, 'call'))
                    else:
                        progress_print(f"⏭️ Skipping {strike_name}(${target_strike:.0f} call) - cached as missing")
            
            if 'put' in strike_name or strike_name == 'atm':
                # Look for puts
                closest_put = min(puts_sorted, 
                                key=lambda x: abs(x.strike - target_strike), 
                                default=None)
                if closest_put and abs(closest_put.strike - target_strike) <= 5.0:  # Within $5
                    if strike_name == 'atm':
                        option_strikes['atm_put'] = closest_put
                    else:
                        option_strikes[strike_name] = closest_put
                else:
                    # Check if this strike is cached as missing before making API call
                    if not self._is_strike_cached_as_missing(strike_name, target_strike, 'put', current_date):
                        missing_strikes.append((strike_name, target_strike, 'put'))
                    else:
                        progress_print(f"⏭️ Skipping {strike_name}(${target_strike:.0f} put) - cached as missing")
        
        # Fetch missing strikes from Polygon API and update cache
        if missing_strikes:
            # Create a detailed list of missing strikes for the print statement
            missing_strikes_details = []
            for strike_name, target_strike, option_type in missing_strikes:
                missing_strikes_details.append(f"{strike_name}(${target_strike:.0f} {option_type})")
            
            progress_print(f"🔄 Fetching {len(missing_strikes)} missing option strikes from Polygon API: {', '.join(missing_strikes_details)}")
            additional_options = self._fetch_missing_strikes(missing_strikes, expiry, current_date)
            
            # Add fetched options to our results
            option_strikes.update(additional_options)
            
            # Update the cache with additional options
            self._update_cache_with_additional_strikes(chain_data, additional_options, current_date)
        
        return option_strikes

    def _fetch_missing_strikes(self, missing_strikes: List, expiry: str, current_date: datetime) -> Dict:
        """Fetch specific missing option strikes from Polygon API and cache missing strikes to avoid repeated API calls"""
        global total_api_calls_made
        additional_options = {}
        missing_strikes_to_cache = []  # Track strikes that don't exist for caching
        
        for strike_name, target_strike, option_type in missing_strikes:
            try:
                progress_print(f"🌐 Fetching {option_type} strike ${target_strike:.0f} for {strike_name}")
                
                # Find contracts close to target strike
                contracts = self._fetch_contracts_for_strike(target_strike, expiry, option_type, current_date)
                total_api_calls_made += 1  # Track total API calls
                
                if contracts:
                    # Get the closest contract to our target strike
                    closest_contract = min(contracts, 
                                         key=lambda x: abs(float(x.strike_price) - target_strike))
                    
                    if abs(float(closest_contract.strike_price) - target_strike) <= 5.0:  # Within $5
                        # Try to fetch historical data for this contract
                        option_data = self._fetch_historical_contract_data(closest_contract, current_date)
                        total_api_calls_made += 1  # Track total API calls
                        
                        # If no data available, try alternative strikes
                        if not option_data:
                            progress_print(f"⚠️ No data for {option_type} ${float(closest_contract.strike_price):.0f}, trying alternative strikes...")
                            
                            # Sort contracts by distance from target strike
                            sorted_contracts = sorted(contracts, 
                                                    key=lambda x: abs(float(x.strike_price) - target_strike))
                            
                            # Try up to 3 alternative strikes
                            for alt_contract in sorted_contracts[1:4]:  # Skip the first one we already tried
                                progress_print(f"Trying alternative {option_type} ${float(alt_contract.strike_price):.0f}")
                                option_data = self._fetch_historical_contract_data(alt_contract, current_date)
                                total_api_calls_made += 1  # Track total API calls
                                if option_data:
                                    progress_print(f"✅ Found data for alternative {option_type} ${float(alt_contract.strike_price):.0f}")
                                    break
                        
                        if option_data:
                            if strike_name == 'atm':
                                additional_options[f'atm_{option_type}'] = option_data
                            else:
                                additional_options[strike_name] = option_data
                            progress_print(f"✅ Successfully fetched {option_type} ${float(closest_contract.strike_price):.0f} @ ${option_data['last_price']:.2f}")
                        else:
                            progress_print(f"❌ No historical data available for any {option_type} strikes near ${target_strike:.0f}")
                            # Cache this missing strike to avoid future API calls
                            missing_strikes_to_cache.append((strike_name, target_strike, option_type))
                    else:
                        progress_print(f"⚠️ No suitable {option_type} found near ${target_strike:.0f} (closest: ${float(closest_contract.strike_price):.0f})")
                else:
                    progress_print(f"❌ No contracts found for {option_type} ${target_strike:.0f}")
                    # Cache this missing strike to avoid future API calls
                    missing_strikes_to_cache.append((strike_name, target_strike, option_type))
                    
            except Exception as e:
                progress_print(f"❌ Error fetching {option_type} ${target_strike:.0f}: {str(e)}")
                continue
        
        # Cache missing strikes to avoid repeated API calls
        if missing_strikes_to_cache:
            self._cache_missing_strikes(missing_strikes_to_cache, current_date)
        
        return additional_options

    def _cache_missing_strikes(self, missing_strikes: List, current_date: datetime):
        """Cache missing strikes to avoid repeated API calls for strikes that don't exist"""
        try:
            # Load existing missing strikes cache
            missing_strikes_cache = self.cache_manager.load_date_from_cache(
                current_date,
                '_missing_strikes',
                'options',
                self.symbol
            )
            
            if missing_strikes_cache is None:
                missing_strikes_cache = []
            
            # Add new missing strikes to cache
            for strike_name, target_strike, option_type in missing_strikes:
                missing_strike_key = f"{strike_name}_{target_strike}_{option_type}"
                if missing_strike_key not in missing_strikes_cache:
                    missing_strikes_cache.append(missing_strike_key)
            
            # Save updated cache
            self.cache_manager.save_date_to_cache(
                current_date,
                missing_strikes_cache,
                '_missing_strikes',
                'options',
                self.symbol
            )
            
            progress_print(f"💾 Cached {len(missing_strikes)} missing strikes to avoid future API calls")
            
        except Exception as e:
            progress_print(f"⚠️ Warning: Could not cache missing strikes: {str(e)}")

    def _is_strike_cached_as_missing(self, strike_name: str, target_strike: float, option_type: str, current_date: datetime) -> bool:
        """Check if a strike is cached as missing to avoid unnecessary API calls"""
        try:
            missing_strikes_cache = self.cache_manager.load_date_from_cache(
                current_date,
                '_missing_strikes',
                'options',
                self.symbol
            )
            
            if missing_strikes_cache is None:
                return False
            
            missing_strike_key = f"{strike_name}_{target_strike}_{option_type}"
            return missing_strike_key in missing_strikes_cache
            
        except Exception as e:
            progress_print(f"⚠️ Warning: Could not check missing strikes cache: {str(e)}")
            return False

    def _fetch_contracts_for_strike(self, target_strike: float, expiry: str, option_type: str, current_date: datetime) -> List:
        """Fetch option contracts for a specific strike price and expiration date from Polygon API.
        
        This method queries the Polygon API to retrieve available option contracts that match
        the specified criteria. It uses the retry handler to handle API rate limits and failures.
        
        Args:
            target_strike (float): The target strike price to search for (e.g., 420.0)
            expiry (str): The expiration date in 'YYYY-MM-DD' format (e.g., '2025-01-17')
            option_type (str): The option type - either 'call' or 'put'
            current_date (datetime): The date as of which to search for contracts
            
        Returns:
            List: A list of option contract objects from Polygon API. Each contract contains
                  metadata like strike_price, expiration_date, contract_type, ticker, etc.
                  Returns empty list if no contracts found or API call fails.
                  
        Note:
            - Uses retry logic to handle API rate limits and temporary failures
            - Limits results to 14 contracts to avoid overwhelming responses
            - Only fetches non-expired contracts (expired=False)
            - Logs the number of contracts found for debugging purposes
        """
        def fetch_func():
            try:                                
                contracts_response = self.client.list_options_contracts(
                    underlying_ticker=self.symbol,
                    as_of=current_date.strftime('%Y-%m-%d'),
                    params={
                        "contract_type": option_type,
                        "strike_price": target_strike,
                        "expiration_date": expiry
                    },
                    expired=False,
                    limit=50
                )

                contracts = []
                if contracts_response:
                    for contract in contracts_response:
                        contracts.append(contract)
                        if len(contracts) >= 14:  # Limit to avoid too many results
                            break
                
                progress_print(f"Found {len(contracts)} contracts for {option_type} ${target_strike:.0f}")
                return contracts
                
            except Exception as e:
                print(f"Error in fetch_func for {option_type} ${target_strike:.0f}: {str(e)}")
                return []
                
        return self.retry_handler.fetch_with_retry(
            fetch_func,
            f"Error fetching contracts for {option_type} ${target_strike:.0f}"
        )

    def _update_cache_with_additional_strikes(self, chain_data: OptionChain, additional_options: Dict, current_date: datetime):
        """Update the cached chain data with newly fetched option strikes"""
        if not additional_options:
            return

        print(f"💾 Updating cache with {len(additional_options)} additional option strikes")

        # Add additional options to chain_data
        for strike_name, option_data in additional_options.items():
            if option_data['type'] == 'call':
                chain_data.calls.append(option_data)
            else:
                chain_data.puts.append(option_data)

        # Update the cache
        try:
            self.cache_manager.save_date_to_cache(
                current_date,
                chain_data.to_dict(),
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            print(f"✅ Cache updated with additional strikes for {current_date.date()}")
        except Exception as e:
            print(f"⚠️ Warning: Could not update cache: {str(e)}")

    def calculate_option_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, OptionChain]]:
        """Calculate option-related features with multi-strike data for strategy modeling
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, OptionChain]]: 
                - lstm_data: DataFrame with calculated option features
                - options_data: Dictionary mapping date strings to OptionChain DTOs
        """
        # Initialize progress tracker with user-specified quiet mode
        progress = ProgressTracker(
            start_date=data.index[0],
            end_date=data.index[-1],
            total_dates=len(data.index),
            desc="Processing options data",
            quiet_mode=self.quiet_mode
        )
        
        # Set as global progress tracker so other methods can use it
        set_global_progress_tracker(progress)
        
        # Dictionary to store OptionChain DTOs for each date
        options_data = {}
        # Track actual API calls globally for this method
        global total_api_calls_made
        total_api_calls_made = 0
        
        for current_date in data.index:
            progress_print(f"\n****Start processing {current_date}****")
            # Ensure current_date is naive datetime for comparison
            current_date_naive = pd.Timestamp(current_date).tz_localize(None)
            # Skip dates before start_date if specified
            if self.start_date and current_date_naive < self.start_date:
                progress.update(current_date=current_date_naive)
                continue
                
            try:
                # Get current price
                current_price = data.loc[current_date, 'Close']
                print(f"Current Price: {current_price}")
                
                # Get option chain with multiple strikes
                chain_data = self._get_option_chain_with_cache(current_date, current_price)
                
                # Update progress with current status (before processing to show early)
                progress.update(
                    current_date=current_date_naive,
                    additional_info={
                        'calls': len(chain_data.calls),
                        'puts': len(chain_data.puts)
                    }
                )
                
                # Convert to OptionChain DTO and store
                if chain_data.calls or chain_data.puts:
                    option_chain = OptionChain(
                        calls=chain_data.calls,
                        puts=chain_data.puts,
                        underlying_symbol=self.symbol,
                        current_price=current_price,
                        date=current_date.strftime('%Y-%m-%d'),
                        source='options_handler'
                    )
                    progress_print(f"Option Chain: {option_chain.__str__()}")
                    
                    # Store in options_data dictionary
                    date_key = current_date.strftime('%Y-%m-%d')
                    options_data[date_key] = option_chain
                
                if not chain_data.calls or not chain_data.puts:
                    # Just continue silently in quiet mode, show brief message in postfix
                    progress.update(summary_info="No options data")
                    continue
                
                # Get target expiry
                target_expiry = self._get_target_expiry(current_date, chain_data)
                if not target_expiry:
                    progress.update(summary_info="No target expiry")
                    continue
                
                # Get comprehensive option strikes for strategy modeling (with API fetch for missing strikes)
                option_strikes = self._get_strategy_option_strikes(current_price, chain_data, target_expiry, current_date)
                
                if not option_strikes:
                    progress.update(summary_info="No option strikes")
                    continue
                
                # ATM Options (for features and straddle)
                if 'atm_call' in option_strikes and 'atm_put' in option_strikes:
                    atm_call: Option = option_strikes['atm_call']
                    atm_put: Option = option_strikes['atm_put']
                    
                    # Store ATM option features
                    data.loc[current_date, 'Call_IV'] = atm_call.implied_volatility
                    data.loc[current_date, 'Put_IV'] = atm_put.implied_volatility
                    data.loc[current_date, 'Call_Volume'] = atm_call.volume
                    data.loc[current_date, 'Put_Volume'] = atm_put.volume
                    data.loc[current_date, 'Call_OI'] = atm_call.open_interest
                    data.loc[current_date, 'Put_OI'] = atm_put.open_interest
                    data.loc[current_date, 'Call_Price'] = atm_call.last_price
                    data.loc[current_date, 'Put_Price'] = atm_put.last_price
                    data.loc[current_date, 'Call_Delta'] = atm_call.delta
                    data.loc[current_date, 'Put_Delta'] = atm_put.delta
                    data.loc[current_date, 'Call_Gamma'] = atm_call.gamma
                    data.loc[current_date, 'Put_Gamma'] = atm_put.gamma
                    data.loc[current_date, 'Call_Theta'] = atm_call.theta
                    data.loc[current_date, 'Put_Theta'] = atm_put.theta
                    data.loc[current_date, 'Call_Vega'] = atm_call.vega
                    data.loc[current_date, 'Put_Vega'] = atm_put.vega
                    data.loc[current_date, 'Option_Volume_Ratio'] = (atm_call.volume + atm_put.volume) / data.loc[current_date, 'Volume']
                    data.loc[current_date, 'Put_Call_Ratio'] = atm_put.volume / atm_call.volume if atm_call.volume > 0 else 1.0
                
                # Multi-strike options for strategy calculations (now guaranteed by API fetch)
                if 'call_atm_plus5' in option_strikes:
                    data.loc[current_date, 'Call_ATM_Plus5_Price'] = option_strikes['call_atm_plus5'].last_price
                
                if 'call_atm_plus10' in option_strikes:
                    data.loc[current_date, 'Call_ATM_Plus10_Price'] = option_strikes['call_atm_plus10'].last_price
                
                if 'put_atm_minus5' in option_strikes:
                    data.loc[current_date, 'Put_ATM_Minus5_Price'] = option_strikes['put_atm_minus5'].last_price
                
                if 'put_atm_minus10' in option_strikes:
                    data.loc[current_date, 'Put_ATM_Minus10_Price'] = option_strikes['put_atm_minus10'].last_price
                
                # Create compact summary for progress bar postfix
                strike_summary = ""
                if 'atm_call' in option_strikes and 'atm_put' in option_strikes:
                    call_price = option_strikes['atm_call'].last_price
                    put_price = option_strikes['atm_put'].last_price
                    strike = option_strikes['atm_call'].strike
                    strike_summary = f"ATM ${strike:.0f}: C${call_price:.2f}/P${put_price:.2f}"
                
                # Final update with strike summary
                progress.update(
                    summary_info=strike_summary
                )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                progress.write(f"\nError processing {current_date}: {str(e)}")
        
        # Close progress tracker and print final summary
        progress.close()
        
        # Clear the global progress tracker
        set_global_progress_tracker(None)
        
        print(f"✅ Processed {len(options_data)} days of option chain data")
        print(f"   Option chains available for: {list(options_data.keys())[:5]}{'...' if len(options_data) > 5 else ''}")
        print(f"   Total successful API calls: {total_api_calls_made}")
                
        return data, options_data

    def calculate_option_signals(self, data: pd.DataFrame, holding_period: int = 15, min_return_threshold: float = 0.08) -> pd.DataFrame:
        """Calculate trading signals based on options strategies using real multi-strike data
        
        Strategy Classes:
        0: Hold - No position (when no strategy meets criteria)
        1: Call Credit Spread - Moderately bearish (sell ATM call + buy ATM+5 call)
        2: Put Credit Spread - Moderately bullish (sell ATM put + buy ATM-5 put) 
        
        Args:
            holding_period: Number of days to hold the strategy (default: 5)
            min_return_threshold: Minimum return required to generate a signal (default: 8%)
        """
        print(f"Generating strategy labels using multi-strike data with {holding_period}-day holding period and {min_return_threshold:.1%} minimum threshold")
        
        # Initialize the signal column
        data['Option_Signal'] = 0  # Default to Hold
        
        # Check if required columns exist, if not create them with NaN values
        required_columns = ['Call_ATM_Plus5_Price', 'Put_ATM_Minus5_Price', 'Call_Price', 'Put_Price']
        for col in required_columns:
            if col not in data.columns:
                data[col] = np.nan
                print(f"⚠️ Warning: {col} column not found, using NaN values")
        
        # Multi-strike returns for realistic strategy calculations
        data['Call_Plus5_Return'] = data['Call_ATM_Plus5_Price'].pct_change(fill_method=None)
        data['Put_Minus5_Return'] = data['Put_ATM_Minus5_Price'].pct_change(fill_method=None)
        
        # Stock movement for context
        stock_return = data['Close'].shift(-holding_period) / data['Close'] - 1
        data['Future_Stock_Return'] = stock_return
        
        # Strategy Return Calculations using actual multi-strike data
        
        # 1. Call Credit Spread: Sell ATM Call + Buy ATM+5 Call
        atm_call_future = data['Call_Price'].shift(-holding_period) / data['Call_Price'] - 1
        plus5_call_future = data['Call_ATM_Plus5_Price'].shift(-holding_period) / data['Call_ATM_Plus5_Price'] - 1
        
        # Credit spread P&L: Short leg profit - Long leg loss (reversed from debit spread)
        data['Future_Call_Credit_Return'] = -atm_call_future + plus5_call_future
        print("✅ Using REAL Call Credit Spread calculation (Sell ATM Call + Buy ATM+5 Call)")
        
        # 2. Put Credit Spread: Sell ATM Put + Buy ATM-5 Put
        atm_put_future = data['Put_Price'].shift(-holding_period) / data['Put_Price'] - 1
        minus5_put_future = data['Put_ATM_Minus5_Price'].shift(-holding_period) / data['Put_ATM_Minus5_Price'] - 1
        
        # Credit spread P&L: Short leg profit - Long leg loss (reversed from debit spread)
        data['Future_Put_Credit_Return'] = -atm_put_future + minus5_put_future
        print("✅ Using REAL Put Credit Spread calculation (Sell ATM Put + Buy ATM-5 Put)")
        
        # Strategy counters
        strategy_counts = [0, 0, 0]  # Hold, Call Credit, Put Credit
        strategy_names = ['Hold', 'Call Credit Spread', 'Put Credit Spread']
        
        # Generate labels based on best performing strategy with market context
        valid_strategies = 0
        for i in range(len(data) - holding_period):  # Exclude last holding_period rows
            
            # Get future returns for each strategy
            call_credit_return = data['Future_Call_Credit_Return'].iloc[i]
            put_credit_return = data['Future_Put_Credit_Return'].iloc[i]
            
            # Skip if we don't have complete option data
            if (pd.isna(call_credit_return) or pd.isna(put_credit_return)):
                continue
            
            valid_strategies += 1
            
            # Market regime factors for strategy selection
            recent_trend = data['SMA20_to_SMA50'].iloc[i] if not pd.isna(data['SMA20_to_SMA50'].iloc[i]) else 1.0
            
            # Find the best strategy that meets minimum threshold
            # Credit spreads work better in sideways/trending markets with less strict requirements
            strategy_returns = {
                1: call_credit_return if recent_trend < 0.998 else call_credit_return * 0.8,      # Favor in downtrends/neutral
                2: put_credit_return if recent_trend > 1.002 else put_credit_return * 0.8,       # Favor in uptrends/neutral
            }
            
            # Find best strategy above threshold
            best_strategy = 0  # Default to Hold
            best_return = 0
            
            for strategy_id, strategy_return in strategy_returns.items():
                if strategy_return > min_return_threshold and strategy_return > best_return:
                    best_strategy = strategy_id
                    best_return = strategy_return
            
            # Assign the best strategy
            data.iloc[i, data.columns.get_loc('Option_Signal')] = best_strategy
            strategy_counts[best_strategy] += 1
        
        # Display strategy distribution
        total_signals = sum(strategy_counts)
        
        if total_signals > 0:
            print(f"\nStrategy Distribution ({valid_strategies} valid samples):")
            for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
                print(f"  {i}: {name:18} {count:4d} ({count/total_signals:.1%})")
            
            # Show average returns for each strategy when selected
            for strategy_id in range(1, 3):
                strategy_mask = data['Option_Signal'] == strategy_id
                if strategy_mask.sum() > 0:
                    if strategy_id == 1:
                        avg_return = data[strategy_mask]['Future_Call_Credit_Return'].mean()
                    else:  # strategy_id == 2
                        avg_return = data[strategy_mask]['Future_Put_Credit_Return'].mean()
                    
                    print(f"  Avg {strategy_names[strategy_id]} Return: {avg_return:.2%}")
        else:
            print("⚠️ No valid strategy signals generated - check multi-strike data availability")
        
        return data

    def _calculate_iron_butterfly_pnl(self, entry_price, exit_price, atm_strike, long_call_strike, long_put_strike, market_data):
        """Calculate realistic Iron Butterfly P&L based on actual strikes and option pricing
        
        Iron Butterfly Structure:
        - Sell ATM Call (collect premium)
        - Sell ATM Put (collect premium) 
        - Buy ATM+10 Call (protection)
        - Buy ATM-10 Put (protection)
        """
        
        # Strike configuration
        strike_width = long_call_strike - atm_strike  # Should be 10 for our strikes
        
        # Premium collected (Iron Butterfly collects more than Iron Condor since both shorts are ATM)
        max_profit = strike_width * 0.6  # Higher premium for ATM straddle sale
        max_loss = strike_width - max_profit
        
        # Calculate intrinsic values at expiration
        short_call_value = max(0, exit_price - atm_strike)      # ATM call we sold
        short_put_value = max(0, atm_strike - exit_price)       # ATM put we sold
        long_call_value = max(0, exit_price - long_call_strike) # Protection call we bought
        long_put_value = max(0, long_put_strike - exit_price)   # Protection put we bought
        
        # Net position value at expiration
        # We owe: short_call_value + short_put_value
        # We receive: long_call_value + long_put_value
        net_position_value = (long_call_value + long_put_value) - (short_call_value + short_put_value)
        
        # Total P&L = Premium collected + Net position value
        total_pnl = max_profit + net_position_value
        
        # Convert to percentage return
        capital_required = max_loss  # Margin requirement approximation
        if capital_required > 0:
            return_pct = total_pnl / capital_required
        else:
            return_pct = 0.08  # Default higher return for Iron Butterfly
            
        # Iron Butterfly benefits from low volatility and price staying near ATM
        price_distance = abs(exit_price - atm_strike)
        if price_distance < strike_width * 0.3:  # Price stayed very close to ATM
            return_pct *= 1.3  # Boost returns when prediction is very accurate
        elif hasattr(market_data, 'Volatility_Percentile') and market_data['Volatility_Percentile'] < 0.5:
            return_pct *= 1.1  # Smaller boost in low volatility
            
        # Realistic bounds (Iron Butterfly can be more profitable than Iron Condor)
        return max(-0.90, min(0.40, return_pct))  # Cap losses at -90%, gains at 40% 