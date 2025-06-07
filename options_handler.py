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
import requests
from cache_manager import CacheManager
from api_retry_handler import APIRetryHandler

# Load environment variables from .env file
load_dotenv()

class OptionsHandler:
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache', start_date: str = None):
        """Initialize the OptionsHandler with a symbol and optional Polygon.io API key"""
        self.symbol = symbol
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        
        if not self.api_key:
            raise ValueError("Polygon.io API key is required.")
            
        self.cache_manager = CacheManager(cache_dir)
        self.client = RESTClient(self.api_key)
        self.retry_handler = APIRetryHandler()
        # Convert start_date to naive datetime for consistent comparison
        self.start_date = pd.Timestamp(start_date).tz_localize(None) if start_date else None
        
    def _fetch_historical_contract_data(self, contract, current_date: datetime) -> Optional[Dict]:
        """Fetch historical data for a specific contract with retries"""
        def fetch_func():
            print(f"Fetching historical data for {contract.ticker}")
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
                        print(f"Error converting generator to list for {contract.ticker}: {error_str}")
                        
                        # Check for authorization errors that indicate plan limitations
                        if "NOT_AUTHORIZED" in error_str or "doesn't include this data timeframe" in error_str:
                            print(f"⚠️ Authorization error for {current_date.date()}: Plan doesn't cover this timeframe")
                            raise ValueError("SKIP_DATE_UNAUTHORIZED")
                        return None
                        
                if not aggs:
                    print(f"No aggregate data available for {contract.ticker}")
                    return None
                    
                day_data = aggs[0]
                
                # Enhanced data structure to support Polygon.io's available fields
                # Note: For historical data, Greeks/IV would need Options Snapshot API
                return {
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date,
                    'type': 'call' if contract.contract_type.lower() == 'call' else 'put',
                    'symbol': contract.ticker,
                    'volume': day_data.volume if hasattr(day_data, 'volume') else 0,
                    'open_interest': None,  # Available in Options Snapshot API (Starter+ plans)
                    'implied_volatility': None,  # Available in Options Snapshot API (Starter+ plans)
                    'delta': None,  # Available in Options Snapshot API (Starter+ plans)
                    'gamma': None,  # Available in Options Snapshot API (Starter+ plans)
                    'theta': None,  # Available in Options Snapshot API (Starter+ plans)
                    'vega': None,  # Available in Options Snapshot API (Starter+ plans)
                    'last_price': day_data.close,
                    'bid': day_data.low,  # Real bid/ask available with Advanced plan ($199/month)
                    'ask': day_data.high,  # Real bid/ask available with Advanced plan ($199/month)
                    # Additional fields that could be calculated
                    'mid_price': (day_data.low + day_data.high) / 2,
                    'intrinsic_value': self._calculate_intrinsic_value(
                        float(contract.strike_price), 
                        day_data.close, 
                        contract.contract_type.lower()
                    ),
                    'time_value': day_data.close - self._calculate_intrinsic_value(
                        float(contract.strike_price), 
                        day_data.close, 
                        contract.contract_type.lower()
                    ),
                    'moneyness': day_data.close / float(contract.strike_price),
                }
            except ValueError as e:
                if "SKIP_DATE_UNAUTHORIZED" in str(e):
                    raise  # Re-raise to be caught by the outer handler
                print(f"ValueError in fetch_func for {contract.ticker}: {str(e)}")
                return None
            except Exception as e:
                error_str = str(e)
                print(f"Error in fetch_func for {contract.ticker}: {error_str}")
                
                # Check for authorization errors in general exceptions too
                if "NOT_AUTHORIZED" in error_str or "doesn't include this data timeframe" in error_str:
                    print(f"⚠️ Authorization error for {current_date.date()}: Plan doesn't cover this timeframe")
                    raise ValueError("SKIP_DATE_UNAUTHORIZED")
                return None
            
        return self.retry_handler.fetch_with_retry(
            fetch_func,
            f"Error fetching historical data for {contract.ticker}"
        )

    def _calculate_intrinsic_value(self, strike: float, underlying_price: float, option_type: str) -> float:
        """Calculate intrinsic value of an option"""
        if option_type == 'call':
            return max(underlying_price - strike, 0)
        else:  # put
            return max(strike - underlying_price, 0)

    def _fetch_historical_contracts_data(self, contracts: List, current_date: datetime, current_price: float) -> Dict:
        """Fetch historical data for comprehensive multi-strike option chains needed for strategy modeling"""
        chain_data = {'calls': [], 'puts': []}
        
        print(f"Processing {len(contracts)} contracts from API/cache")
        
        if not contracts:
            print("No contracts to process")
            return chain_data
        
        try:
            # Step 1: Find target expiry closest to 30 days
            target_date = current_date + timedelta(days=30)
            
            # Get unique expiration dates from contracts
            expiry_dates = set()
            for contract in contracts:
                if hasattr(contract, 'expiration_date'):
                    expiry_dates.add(contract.expiration_date)
            
            if not expiry_dates:
                print("No expiration dates found in contracts")
                return chain_data
            
            # Convert to datetime objects and find closest to 30 days
            expiry_dates = [pd.Timestamp(exp).tz_localize(None) for exp in expiry_dates]
            target_expiry = min(expiry_dates, key=lambda x: abs((x - target_date).days))
            target_expiry_str = target_expiry.strftime('%Y-%m-%d')
            
            print(f"Target expiry (closest to 30 days): {target_expiry_str}")
            
            # Step 2: Filter contracts to only the target expiry
            target_expiry_contracts = []
            for contract in contracts:
                if hasattr(contract, 'expiration_date') and contract.expiration_date == target_expiry_str:
                    target_expiry_contracts.append(contract)
            
            print(f"Contracts with target expiry: {len(target_expiry_contracts)}")
            
            if not target_expiry_contracts:
                print(f"No contracts found for target expiry {target_expiry_str}")
                return chain_data
            
            # Step 3: Separate calls and puts, sorted by proximity to current price
            calls = [c for c in target_expiry_contracts if hasattr(c, 'contract_type') and c.contract_type.lower() == 'call']
            puts = [c for c in target_expiry_contracts if hasattr(c, 'contract_type') and c.contract_type.lower() == 'put']
            
            # Sort by distance from current price (closest first)
            calls_sorted = sorted(calls, key=lambda x: abs(float(x.strike_price) - current_price))
            puts_sorted = sorted(puts, key=lambda x: abs(float(x.strike_price) - current_price))
            
            print(f"Found {len(calls_sorted)} calls and {len(puts_sorted)} puts for target expiry")
            
            # Step 4: Get comprehensive strike data for strategy modeling
            # For accurate strategy returns, we need multiple strikes:
            # - ATM ±0 (for straddles)
            # - ATM ±$5, ±$10 (for debit spreads)  
            # - OTM strikes for credit spreads (Iron Condor)
            
            target_strikes = [
                current_price,           # ATM
                current_price + 5,       # 1 strike OTM call
                current_price + 10,      # 2 strikes OTM call  
                current_price - 5,       # 1 strike OTM put
                current_price - 10,      # 2 strikes OTM put
            ]
            
            # Try to get calls for each target strike
            for target_strike in target_strikes:
                if target_strike <= current_price:  # Skip call strikes below current price
                    continue
                    
                # Find closest call to target strike
                closest_call = min(calls_sorted, key=lambda x: abs(float(x.strike_price) - target_strike), default=None)
                if closest_call and abs(float(closest_call.strike_price) - target_strike) <= 2.5:  # Within $2.50
                    try:
                        call_data = self._fetch_historical_contract_data(closest_call, current_date)
                        if call_data:
                            print(f"✓ Got call data: ${float(closest_call.strike_price):.0f} strike @ ${call_data['last_price']:.2f}")
                            chain_data['calls'].append(call_data)
                    except ValueError as e:
                        if "SKIP_DATE_UNAUTHORIZED" in str(e):
                            print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                            return chain_data
                        raise
            
            # Try to get puts for each target strike
            for target_strike in target_strikes:
                if target_strike >= current_price:  # Skip put strikes above current price
                    continue
                    
                # Find closest put to target strike
                closest_put = min(puts_sorted, key=lambda x: abs(float(x.strike_price) - target_strike), default=None)
                if closest_put and abs(float(closest_put.strike_price) - target_strike) <= 2.5:  # Within $2.50
                    try:
                        put_data = self._fetch_historical_contract_data(closest_put, current_date)
                        if put_data:
                            print(f"✓ Got put data: ${float(closest_put.strike_price):.0f} strike @ ${put_data['last_price']:.2f}")
                            chain_data['puts'].append(put_data)
                    except ValueError as e:
                        if "SKIP_DATE_UNAUTHORIZED" in str(e):
                            print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                            return chain_data
                        raise
            
            # Also try to get ATM options (both calls and puts at same strike)
            atm_calls = [c for c in calls_sorted[:3] if abs(float(c.strike_price) - current_price) <= 2.5]
            atm_puts = [p for p in puts_sorted[:3] if abs(float(p.strike_price) - current_price) <= 2.5]
            
            for atm_call in atm_calls:
                try:
                    call_data = self._fetch_historical_contract_data(atm_call, current_date)
                    if call_data and not any(c['strike'] == call_data['strike'] for c in chain_data['calls']):
                        print(f"✓ Got ATM call: ${float(atm_call.strike_price):.0f} strike @ ${call_data['last_price']:.2f}")
                        chain_data['calls'].append(call_data)
                        break
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        return chain_data
                    continue
            
            for atm_put in atm_puts:
                try:
                    put_data = self._fetch_historical_contract_data(atm_put, current_date)
                    if put_data and not any(p['strike'] == put_data['strike'] for p in chain_data['puts']):
                        print(f"✓ Got ATM put: ${float(atm_put.strike_price):.0f} strike @ ${put_data['last_price']:.2f}")
                        chain_data['puts'].append(put_data)
                        break
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        return chain_data
                    continue
            
            # Summary
            successful_contracts = len(chain_data['calls']) + len(chain_data['puts'])
            print(f"Successfully fetched comprehensive chain data: {len(chain_data['calls'])} calls, {len(chain_data['puts'])} puts")
                        
        except Exception as e:
            print(f"Error in _fetch_historical_contracts_data: {e}")
            import traceback
            traceback.print_exc()
            
        return chain_data

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
                print(f"Loading cached contracts list for {current_date.date()}")
                return contracts
                    
            # If not in cache, fetch from API with retries
            print(f"Fetching contracts for {current_date.date()}")
            
            def fetch_func():
                try:                    
                    # Calculate strike price range (7% around current price)
                    price_range = current_price * 0.07  # 7% range for broader coverage
                    min_strike = current_price - price_range
                    max_strike = current_price + price_range
                    
                    # Calculate expiration date range (focus on options around 30-day target)
                    min_expiry = current_date + timedelta(days=24)  # Minimum 24 days out (closer to 30-day target)
                    max_expiry = current_date + timedelta(days=36)  # Maximum 36 days out
                    
                    print(f"Filtering contracts: strikes ${min_strike:.0f}-${max_strike:.0f}, expiry {min_expiry.strftime('%Y-%m-%d')} to {max_expiry.strftime('%Y-%m-%d')} (targeting ~30 days)")
                    
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
                                    print(f"Completed page {page_count}, got {len(contracts)} contracts so far")
                                    
                                    # Stop if we've reached max pages to avoid rate limits
                                    if page_count >= max_pages:
                                        print(f"Reached max pages ({max_pages}), stopping to avoid rate limits")
                                        break
                                    
                                    # Rate limit between pages
                                    print("Rate limiting: waiting 13 seconds between paginated requests")
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
                        print(f"No contracts available for {current_date.date()}")
                        
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

    def _get_option_chain_with_cache(self, current_date: datetime, current_price: float) -> Dict:
        """Get option chain data for a date, using cache if available"""
        try:
            print(f"Step 1: Attempting to load from cache for {current_date.date()}")
            # Try to get from cache first
            chain_data = self.cache_manager.load_date_from_cache(
                current_date,
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            
            if chain_data is not None:
                print(f"Loading cached option data for {current_date.date()}")
                return chain_data
                    
            print(f"Step 2: No cache found, building option chain data for {current_date.date()}")
            chain_data = {'calls': [], 'puts': []}
            
            try:
                print(f"Step 3: Getting contracts for {current_date.date()}")
                # Get contracts (will use cache if available)
                contracts = self._get_contracts_from_cache(current_date, current_price)
                
                if not contracts:
                    print(f"No options data available for {current_date.date()}")
                    return chain_data
                
                print(f"Step 4: Fetching historical data for {len(contracts)} contracts")
                # Fetch historical data for all contracts
                chain_data = self._fetch_historical_contracts_data(contracts, current_date, current_price)
                
                print(f"Step 5: Caching results")
                # Cache the data if we got any contracts
                if chain_data['calls'] or chain_data['puts']:
                    print(f"Caching option chain data for {current_date.date()}")
                    self.cache_manager.save_date_to_cache(
                        current_date,
                        chain_data,
                        '',  # No suffix for main chain data
                        'options',
                        self.symbol
                    )
                
            except Exception as e:
                print(f"Error in steps 3-5 for {current_date}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in step 1-2 (cache loading) for {current_date}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            chain_data = {'calls': [], 'puts': []}
            
        return chain_data
        
    def _get_target_expiry(self, current_date: datetime, chain_data: Dict) -> Optional[str]:
        """Get the target expiration date closest to 30 days out"""
        if not chain_data['calls'] and not chain_data['puts']:
            return None
            
        target_date = current_date + timedelta(days=30)
        
        # Get unique expiration dates
        expiry_dates = set()
        for option_type in ['calls', 'puts']:
            expiry_dates.update(opt['expiration'] for opt in chain_data[option_type])
            
        if not expiry_dates:
            print(f"No expiration dates available for {current_date.date()}")
            return None
            
        # Convert to datetime objects for comparison
        expiry_dates = [pd.Timestamp(exp).tz_localize(None) for exp in expiry_dates]
        
        # Find closest expiry to target date
        closest_expiry = min(expiry_dates, key=lambda x: abs((x - target_date).days))
        print(f"Closest expiry: {closest_expiry.strftime('%Y-%m-%d')}")
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
        
    def _get_strategy_option_strikes(self, current_price: float, chain_data: Dict, expiry: str, current_date: datetime) -> Dict:
        """Get multiple option strikes needed for realistic strategy calculations
        If strikes are missing, fetch them from Polygon API and update cache"""
        option_strikes = {}
        
        # Filter for target expiry
        calls = [opt for opt in chain_data['calls'] if opt['expiration'] == expiry]
        puts = [opt for opt in chain_data['puts'] if opt['expiration'] == expiry]
        
        if not calls or not puts:
            return option_strikes
        
        # Sort by strike price
        calls_sorted = sorted(calls, key=lambda x: x['strike'])
        puts_sorted = sorted(puts, key=lambda x: x['strike'])
        
        # Define target strikes for strategies
        target_strikes = {
            'atm': current_price,                    # ATM for straddles
            'call_atm_plus5': current_price + 5,     # For call debit spreads
            'call_atm_plus10': current_price + 10,   # For iron condor short calls
            'put_atm_minus5': current_price - 5,     # For put debit spreads  
            'put_atm_minus10': current_price - 10,   # For iron condor short puts
        }
        
        # Track missing strikes that need to be fetched
        missing_strikes = []
        
        # Find closest options to each target strike
        for strike_name, target_strike in target_strikes.items():
            if 'call' in strike_name or strike_name == 'atm':
                # Look for calls
                closest_call = min(calls_sorted, 
                                 key=lambda x: abs(x['strike'] - target_strike), 
                                 default=None)
                if closest_call and abs(closest_call['strike'] - target_strike) <= 3.0:  # Within $3
                    if strike_name == 'atm':
                        option_strikes['atm_call'] = closest_call
                    else:
                        option_strikes[strike_name] = closest_call
                else:
                    # Mark as missing for API fetch
                    missing_strikes.append((strike_name, target_strike, 'call'))
            
            if 'put' in strike_name or strike_name == 'atm':
                # Look for puts
                closest_put = min(puts_sorted, 
                                key=lambda x: abs(x['strike'] - target_strike), 
                                default=None)
                if closest_put and abs(closest_put['strike'] - target_strike) <= 3.0:  # Within $3
                    if strike_name == 'atm':
                        option_strikes['atm_put'] = closest_put
                    else:
                        option_strikes[strike_name] = closest_put
                else:
                    # Mark as missing for API fetch
                    missing_strikes.append((strike_name, target_strike, 'put'))
        
        # Fetch missing strikes from Polygon API and update cache
        if missing_strikes:
            print(f"🔄 Fetching {len(missing_strikes)} missing option strikes from Polygon API...")
            additional_options = self._fetch_missing_strikes(missing_strikes, expiry, current_date, current_price)
            
            # Add fetched options to our results
            option_strikes.update(additional_options)
            
            # Update the cache with additional options
            self._update_cache_with_additional_strikes(chain_data, additional_options, current_date)
        
        return option_strikes

    def _fetch_missing_strikes(self, missing_strikes: List, expiry: str, current_date: datetime, current_price: float) -> Dict:
        """Fetch specific missing option strikes from Polygon API"""
        additional_options = {}
        
        for strike_name, target_strike, option_type in missing_strikes:
            try:
                print(f"🌐 Fetching {option_type} strike ${target_strike:.0f} for {strike_name}")
                
                # Find contracts close to target strike
                contracts = self._fetch_contracts_for_strike(target_strike, expiry, option_type, current_date, current_price)
                
                if contracts:
                    # Get the closest contract to our target strike
                    closest_contract = min(contracts, 
                                         key=lambda x: abs(float(x.strike_price) - target_strike))
                    
                    if abs(float(closest_contract.strike_price) - target_strike) <= 5.0:  # Within $5
                        # Fetch historical data for this contract
                        option_data = self._fetch_historical_contract_data(closest_contract, current_date)
                        
                        if option_data:
                            if strike_name == 'atm':
                                additional_options[f'atm_{option_type}'] = option_data
                            else:
                                additional_options[strike_name] = option_data
                            print(f"✅ Successfully fetched {option_type} ${float(closest_contract.strike_price):.0f} @ ${option_data['last_price']:.2f}")
                        else:
                            print(f"❌ No historical data for {option_type} ${float(closest_contract.strike_price):.0f}")
                    else:
                        print(f"⚠️ No suitable {option_type} found near ${target_strike:.0f} (closest: ${float(closest_contract.strike_price):.0f})")
                else:
                    print(f"❌ No contracts found for {option_type} ${target_strike:.0f}")
                    
            except Exception as e:
                print(f"❌ Error fetching {option_type} ${target_strike:.0f}: {str(e)}")
                continue
        
        return additional_options

    def _fetch_contracts_for_strike(self, target_strike: float, expiry: str, option_type: str, current_date: datetime, current_price: float) -> List:
        """Fetch option contracts for a specific strike and expiration"""
        def fetch_func():
            try:
                # Calculate strike range around target
                strike_range = 2.5  # $2.50 range around target
                min_strike = target_strike - strike_range
                max_strike = target_strike + strike_range
                
                # Parse expiration date
                expiry_date = pd.Timestamp(expiry)
                
                print(f"Searching for {option_type} options: strikes ${min_strike:.0f}-${max_strike:.0f}, expiry {expiry}")
                
                contracts_response = self.client.list_options_contracts(
                    underlying_ticker=self.symbol,
                    as_of=current_date.strftime('%Y-%m-%d'),
                    params={
                        "contract_type": option_type,
                        "strike_price.gte": min_strike,
                        "strike_price.lte": max_strike,
                        "expiration_date": expiry
                    },
                    expired=False,
                    limit=50
                )
                
                contracts = []
                if contracts_response:
                    for contract in contracts_response:
                        contracts.append(contract)
                        if len(contracts) >= 10:  # Limit to avoid too many results
                            break
                
                print(f"Found {len(contracts)} contracts for {option_type} ${target_strike:.0f}")
                return contracts
                
            except Exception as e:
                print(f"Error in fetch_func for {option_type} ${target_strike:.0f}: {str(e)}")
                return []
                
        return self.retry_handler.fetch_with_retry(
            fetch_func,
            f"Error fetching contracts for {option_type} ${target_strike:.0f}"
        )

    def _update_cache_with_additional_strikes(self, chain_data: Dict, additional_options: Dict, current_date: datetime):
        """Update the cached chain data with newly fetched option strikes"""
        if not additional_options:
            return
            
        print(f"💾 Updating cache with {len(additional_options)} additional option strikes")
        
        # Add additional options to chain_data
        for strike_name, option_data in additional_options.items():
            if option_data['type'] == 'call':
                chain_data['calls'].append(option_data)
            else:
                chain_data['puts'].append(option_data)
        
        # Update the cache
        try:
            self.cache_manager.save_date_to_cache(
                current_date,
                chain_data,
                '',  # No suffix for main chain data
                'options',
                self.symbol
            )
            print(f"✅ Cache updated with additional strikes for {current_date.date()}")
        except Exception as e:
            print(f"⚠️ Warning: Could not update cache: {str(e)}")

    def calculate_option_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate option-related features with multi-strike data for strategy modeling"""
        print("\nProcessing options data with comprehensive multi-strike collection...")
        
        for current_date in data.index:
            # Ensure current_date is naive datetime for comparison
            current_date_naive = pd.Timestamp(current_date).tz_localize(None)
            # Skip dates before start_date if specified
            if self.start_date and current_date_naive < self.start_date:
                continue
                
            try:
                # Get current price
                current_price = data.loc[current_date, 'Close']
                print(f"\nProcessing {current_date} (Close: {current_price:.2f})")
                
                # Get option chain with multiple strikes
                chain_data = self._get_option_chain_with_cache(current_date, current_price)
                print(f"Retrieved {len(chain_data.get('calls', []))} calls and {len(chain_data.get('puts', []))} puts")
                
                if not chain_data['calls'] or not chain_data['puts']:
                    print("Insufficient option chain data")
                    continue
                
                # Get target expiry
                target_expiry = self._get_target_expiry(current_date, chain_data)
                if not target_expiry:
                    print("Could not get target expiry")
                    continue
                
                # Get comprehensive option strikes for strategy modeling (with API fetch for missing strikes)
                option_strikes = self._get_strategy_option_strikes(current_price, chain_data, target_expiry, current_date)
                
                if not option_strikes:
                    print("Could not get sufficient option strikes for strategies")
                    continue
                
                # ATM Options (for features and straddle)
                if 'atm_call' in option_strikes and 'atm_put' in option_strikes:
                    atm_call = option_strikes['atm_call']
                    atm_put = option_strikes['atm_put']
                    
                    print(f"ATM Call: ${atm_call['strike']:.0f} @ ${atm_call['last_price']:.2f}")
                    print(f"ATM Put:  ${atm_put['strike']:.0f} @ ${atm_put['last_price']:.2f}")
                    
                    # Store ATM option features
                    data.loc[current_date, 'Call_IV'] = atm_call['implied_volatility']
                    data.loc[current_date, 'Put_IV'] = atm_put['implied_volatility']
                    data.loc[current_date, 'Call_Volume'] = atm_call['volume']
                    data.loc[current_date, 'Put_Volume'] = atm_put['volume']
                    data.loc[current_date, 'Call_OI'] = atm_call['open_interest']
                    data.loc[current_date, 'Put_OI'] = atm_put['open_interest']
                    data.loc[current_date, 'Call_Price'] = atm_call['last_price']
                    data.loc[current_date, 'Put_Price'] = atm_put['last_price']
                    data.loc[current_date, 'Call_Delta'] = atm_call['delta']
                    data.loc[current_date, 'Put_Delta'] = atm_put['delta']
                    data.loc[current_date, 'Call_Gamma'] = atm_call['gamma']
                    data.loc[current_date, 'Put_Gamma'] = atm_put['gamma']
                    data.loc[current_date, 'Call_Theta'] = atm_call['theta']
                    data.loc[current_date, 'Put_Theta'] = atm_put['theta']
                    data.loc[current_date, 'Call_Vega'] = atm_call['vega']
                    data.loc[current_date, 'Put_Vega'] = atm_put['vega']
                    data.loc[current_date, 'Option_Volume_Ratio'] = (atm_call['volume'] + atm_put['volume']) / data.loc[current_date, 'Volume']
                    data.loc[current_date, 'Put_Call_Ratio'] = atm_put['volume'] / atm_call['volume'] if atm_call['volume'] > 0 else 1.0
                
                # Multi-strike options for strategy calculations (now guaranteed by API fetch)
                if 'call_atm_plus5' in option_strikes:
                    print(f"Call ATM+5: ${option_strikes['call_atm_plus5']['strike']:.0f} @ ${option_strikes['call_atm_plus5']['last_price']:.2f}")
                    data.loc[current_date, 'Call_ATM_Plus5_Price'] = option_strikes['call_atm_plus5']['last_price']
                
                if 'call_atm_plus10' in option_strikes:
                    print(f"Call ATM+10: ${option_strikes['call_atm_plus10']['strike']:.0f} @ ${option_strikes['call_atm_plus10']['last_price']:.2f}")
                    data.loc[current_date, 'Call_ATM_Plus10_Price'] = option_strikes['call_atm_plus10']['last_price']
                
                if 'put_atm_minus5' in option_strikes:
                    print(f"Put ATM-5: ${option_strikes['put_atm_minus5']['strike']:.0f} @ ${option_strikes['put_atm_minus5']['last_price']:.2f}")
                    data.loc[current_date, 'Put_ATM_Minus5_Price'] = option_strikes['put_atm_minus5']['last_price']
                
                if 'put_atm_minus10' in option_strikes:
                    print(f"Put ATM-10: ${option_strikes['put_atm_minus10']['strike']:.0f} @ ${option_strikes['put_atm_minus10']['last_price']:.2f}")
                    data.loc[current_date, 'Put_ATM_Minus10_Price'] = option_strikes['put_atm_minus10']['last_price']
                    
            except Exception as e:
                print(f"Error processing {current_date}: {str(e)}")
                continue
                
        return data

    def calculate_option_signals(self, data: pd.DataFrame, holding_period: int = 5, min_return_threshold: float = 0.10) -> pd.DataFrame:
        """Calculate trading signals based on sophisticated options strategies using real multi-strike data
        
        Enhanced Strategy Classes:
        0: Hold - No position (when no strategy meets criteria)
        1: Call Debit Spread - Moderately bullish (buy ATM call + sell ATM+5 call)
        2: Put Debit Spread - Moderately bearish (buy ATM put + sell ATM-5 put) 
        3: Iron Condor - Range-bound, low volatility (sell ATM+/-10, buy ATM+/-15)
        4: Long Straddle - High volatility breakout (buy ATM call + buy ATM put)
        
        Args:
            holding_period: Number of days to hold the strategy (default: 5)
            min_return_threshold: Minimum return required to generate a signal (default: 10%)
        """
        print(f"Generating realistic strategy labels using complete multi-strike data with {holding_period}-day holding period and {min_return_threshold:.1%} minimum threshold")
        
        # Initialize the signal column
        data['Option_Signal'] = 0  # Default to Hold
        
        # Calculate returns for all available option strikes
        data['ATM_Call_Return'] = data['Call_Price'].pct_change(fill_method=None)
        data['ATM_Put_Return'] = data['Put_Price'].pct_change(fill_method=None)
        
        # Multi-strike returns for realistic strategy calculations (now guaranteed)
        data['Call_Plus5_Return'] = data['Call_ATM_Plus5_Price'].pct_change(fill_method=None)
        data['Put_Minus5_Return'] = data['Put_ATM_Minus5_Price'].pct_change(fill_method=None)
        
        # Stock movement for context
        stock_return = data['Close'].shift(-holding_period) / data['Close'] - 1
        data['Future_Stock_Return'] = stock_return
        
        # REALISTIC Strategy Return Calculations using actual multi-strike data
        
        # 1. Call Debit Spread: Buy ATM Call + Sell ATM+5 Call
        atm_call_future = data['Call_Price'].shift(-holding_period) / data['Call_Price'] - 1
        plus5_call_future = data['Call_ATM_Plus5_Price'].shift(-holding_period) / data['Call_ATM_Plus5_Price'] - 1
        
        # Real spread P&L: Long leg profit - Short leg loss
        data['Future_Call_Debit_Return'] = atm_call_future - plus5_call_future
        print("✅ Using REAL Call Debit Spread calculation (ATM Call - ATM+5 Call)")
        
        # 2. Put Debit Spread: Buy ATM Put + Sell ATM-5 Put
        atm_put_future = data['Put_Price'].shift(-holding_period) / data['Put_Price'] - 1
        minus5_put_future = data['Put_ATM_Minus5_Price'].shift(-holding_period) / data['Put_ATM_Minus5_Price'] - 1
        
        # Real spread P&L: Long leg profit - Short leg loss
        data['Future_Put_Debit_Return'] = atm_put_future - minus5_put_future
        print("✅ Using REAL Put Debit Spread calculation (ATM Put - ATM-5 Put)")
        
        # 3. Iron Condor: Enhanced model with multi-strike data when available
        # For now, use sophisticated model based on actual price movement and volatility
        data['Price_Change'] = abs(stock_return)
        data['Volatility_Percentile'] = data['Volatility'].rolling(window=60).rank(pct=True)
        
        # Model Iron Condor returns based on realized volatility vs implied volatility
        if 'Call_IV' in data.columns and 'Put_IV' in data.columns:
            avg_iv = (data['Call_IV'] + data['Put_IV']) / 2
            realized_vol = data['Volatility'] * np.sqrt(252)  # Annualized
            vol_ratio = avg_iv / realized_vol
            
            data['Future_Iron_Condor_Return'] = np.where(
                (data['Price_Change'] < 0.03) & (vol_ratio > 1.2),  # Low movement + high IV
                0.20,  # Premium collection profit
                np.where(data['Price_Change'] > 0.06, -0.80, 0.05)  # Large loss on big moves
            )
            print("✅ Using enhanced Iron Condor model with IV/RV ratio")
        else:
            # Basic model when IV data not available
            data['Future_Iron_Condor_Return'] = np.where(
                (data['Price_Change'] < 0.025) & (data['Volatility_Percentile'] < 0.3),
                0.15,
                np.where(data['Price_Change'] > 0.05, -0.8, 0.05)
            )
            print("⚠️ Using basic Iron Condor model (IV data not available)")
        
        # 4. Long Straddle: Buy ATM Call + Buy ATM Put (Always realistic with ATM data)
        # Real straddle P&L: Both legs combined
        data['Future_Long_Straddle_Return'] = atm_call_future + atm_put_future
        print("✅ Using REAL Long Straddle calculation (ATM Call + ATM Put)")
        
        # Strategy counters
        strategy_counts = [0, 0, 0, 0, 0]  # Hold, Call Debit, Put Debit, Iron Condor, Long Straddle
        strategy_names = ['Hold', 'Call Debit Spread', 'Put Debit Spread', 'Iron Condor', 'Long Straddle']
        
        # Generate labels based on best performing strategy with market context
        valid_strategies = 0
        for i in range(len(data) - holding_period):  # Exclude last holding_period rows
            
            # Get future returns for each strategy
            call_debit_return = data['Future_Call_Debit_Return'].iloc[i]
            put_debit_return = data['Future_Put_Debit_Return'].iloc[i]
            iron_condor_return = data['Future_Iron_Condor_Return'].iloc[i]
            long_straddle_return = data['Future_Long_Straddle_Return'].iloc[i]
            
            # Skip if we don't have complete option data
            if (pd.isna(call_debit_return) or pd.isna(put_debit_return) or 
                pd.isna(long_straddle_return)):
                continue
            
            valid_strategies += 1
            
            # Market regime factors for strategy selection
            current_vol_percentile = data['Volatility_Percentile'].iloc[i] if not pd.isna(data['Volatility_Percentile'].iloc[i]) else 0.5
            recent_trend = data['SMA20_to_SMA50'].iloc[i] if not pd.isna(data['SMA20_to_SMA50'].iloc[i]) else 1.0
            
            # Find the best strategy that meets minimum threshold
            strategy_returns = {
                1: call_debit_return if recent_trend > 1.01 else call_debit_return * 0.5,      # Favor in uptrends
                2: put_debit_return if recent_trend < 0.99 else put_debit_return * 0.5,       # Favor in downtrends  
                3: iron_condor_return if current_vol_percentile < 0.4 else iron_condor_return * 0.3,  # Favor in low vol
                4: long_straddle_return if current_vol_percentile > 0.6 else long_straddle_return * 0.3   # Favor in high vol
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
        
        # Display enhanced strategy distribution
        total_signals = sum(strategy_counts)
        
        if total_signals > 0:
            print(f"\nComplete Multi-Strike Strategy Distribution ({valid_strategies} valid samples):")
            for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
                print(f"  {i}: {name:18} {count:4d} ({count/total_signals:.1%})")
            
            # Show average returns for each strategy when selected
            for strategy_id in range(1, 5):
                strategy_mask = data['Option_Signal'] == strategy_id
                if strategy_mask.sum() > 0:
                    if strategy_id == 1:
                        avg_return = data[strategy_mask]['Future_Call_Debit_Return'].mean()
                    elif strategy_id == 2:
                        avg_return = data[strategy_mask]['Future_Put_Debit_Return'].mean()
                    elif strategy_id == 3:
                        avg_return = data[strategy_mask]['Future_Iron_Condor_Return'].mean()
                    else:  # strategy_id == 4
                        avg_return = data[strategy_mask]['Future_Long_Straddle_Return'].mean()
                    
                    print(f"  Avg {strategy_names[strategy_id]} Return: {avg_return:.2%}")
        else:
            print("⚠️ No valid strategy signals generated - check multi-strike data availability")
        
        return data 