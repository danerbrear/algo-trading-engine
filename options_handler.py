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
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache', start_date: str = None, disable_options: bool = False):
        """Initialize the OptionsHandler with a symbol and optional Polygon.io API key"""
        self.symbol = symbol
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.disable_options = disable_options
        
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
                        print(f"Error converting generator to list for {contract.ticker}: {str(e)}")
                        return None
                        
                if not aggs:
                    print(f"No aggregate data available for {contract.ticker}")
                    return None
                    
                day_data = aggs[0]
                return {
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date,
                    'type': 'call' if contract.contract_type.lower() == 'call' else 'put',
                    'symbol': contract.ticker,
                    'volume': day_data.volume if hasattr(day_data, 'volume') else 0,
                    'open_interest': 0,  # Not available in historical data
                    'implied_volatility': None,  # Not available in historical data
                    'delta': None,  # Not available in historical data
                    'gamma': None,  # Not available in historical data
                    'theta': None,  # Not available in historical data
                    'vega': None,  # Not available in historical data
                    'last_price': day_data.close,
                    'bid': day_data.low,  # Using day low as proxy for bid
                    'ask': day_data.high,  # Using day high as proxy for ask
                }
            except Exception as e:
                print(f"Error in fetch_func for {contract.ticker}: {str(e)}")
                return None
            
        return self.retry_handler.fetch_with_retry(
            fetch_func,
            f"Error fetching historical data for {contract.ticker}"
        )

    def _fetch_historical_contracts_data(self, contracts: List, current_date: datetime, current_price: float) -> Dict:
        """Fetch historical data for a list of contracts and organize them into calls and puts"""
        chain_data = {'calls': [], 'puts': []}
        
        print(f"Processing {len(contracts)} contracts from API/cache")
        
        # Local filtering as safety net for cached data that might have many contracts
        # This ensures we don't fetch historical data for hundreds of irrelevant contracts
        try:
            # Filter contracts to only those within 10% of current price to reduce API calls
            price_range = current_price * 0.10  # 10% range for historical data fetching
            min_strike = current_price - price_range
            max_strike = current_price + price_range
            
            filtered_contracts = []
            for contract in contracts:
                if hasattr(contract, 'strike_price'):
                    strike = float(contract.strike_price)
                    if min_strike <= strike <= max_strike:
                        filtered_contracts.append(contract)
            
            print(f"Filtered to {len(filtered_contracts)} relevant contracts (strikes ${min_strike:.0f}-${max_strike:.0f}) for historical data fetching")
            contracts = filtered_contracts
            
        except Exception as e:
            print(f"Error filtering contracts: {e}, using all contracts")
        
        for contract in contracts:
            contract_data = self._fetch_historical_contract_data(contract, current_date)
            if contract_data:
                if contract_data['type'] == 'call':
                    chain_data['calls'].append(contract_data)
                else:
                    chain_data['puts'].append(contract_data)
                    
        return chain_data

    def _get_contracts_from_cache(self, current_date: datetime, current_price: float) -> Optional[List]:
        """Get the list of contracts from cache if available, otherwise fetch from API"""
        try:
            print(f"Contracts Step 1: Checking cache for contracts on {current_date.date()}")
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
                    
            print(f"Contracts Step 2: No cached contracts, fetching from API for {current_date.date()}")
            # If not in cache, fetch from API with retries
            print(f"Fetching contracts for {current_date.date()}")
            
            def fetch_func():
                try:
                    print(f"Contracts Step 3: Calling Polygon API for contracts")
                    
                    # Calculate strike price range (9% around current price)
                    price_range = current_price * 0.08  # 8% range for broader coverage
                    min_strike = current_price - price_range
                    max_strike = current_price + price_range
                    
                    # Calculate expiration date range (focus on options around 30-day target)
                    min_expiry = current_date + timedelta(days=20)  # Minimum 20 days out (closer to 30-day target)
                    max_expiry = current_date + timedelta(days=40)  # Maximum 40 days out
                    
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
                    
                    print(f"Contracts Step 4: Manually iterating through paginated results with rate limiting")
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
                                    
                            print(f"Contracts Step 5: Successfully converted, got {len(contracts)} contracts")
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
        
    def calculate_option_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate option-related features with batch processing"""
        if self.disable_options:
            print("Options processing disabled - skipping options data")
            # Add empty columns for options data to maintain DataFrame structure
            option_columns = [
                'Call_IV', 'Put_IV', 'Call_Volume', 'Put_Volume', 'Call_OI', 'Put_OI',
                'Call_Price', 'Put_Price', 'Call_Delta', 'Put_Delta', 'Call_Gamma', 'Put_Gamma',
                'Call_Theta', 'Put_Theta', 'Call_Vega', 'Put_Vega', 'Option_Volume_Ratio', 'Put_Call_Ratio'
            ]
            for col in option_columns:
                data[col] = None
            return data
            
        print("\nProcessing options data...")
        
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
                
                # Get option chain
                chain_data = self._get_option_chain_with_cache(current_date, current_price)
                print(f"Retrieved {len(chain_data.get('calls', []))} calls and {len(chain_data.get('puts', []))} puts")
                
                # Get target expiry
                target_expiry = self._get_target_expiry(current_date, chain_data)
                if not target_expiry:
                    print("Could not get target expiry")
                    continue
                    
                # Get ATM options
                atm_call, atm_put = self._get_atm_options(current_price, chain_data, target_expiry)
                if not atm_call or not atm_put:
                    print("Could not get ATM options")
                    continue
                print(f"ATM call: {atm_call}")
                print(f"ATM put: {atm_put}")
                    
                # Calculate features
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
                
            except Exception as e:
                print(f"Error processing {current_date}: {str(e)}")
                continue
                
        return data 

    def calculate_option_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on options data
        0: No Position
        1: Buy Call
        2: Buy Put
        """
        # Initialize the signal column
        data['Option_Signal'] = 0
        
        # Calculate returns for ATM options
        data['ATM_Call_Return'] = data['Call_Price'].pct_change()
        data['ATM_Put_Return'] = data['Put_Price'].pct_change()
        
        # Simple signal generation based on volume ratios and price movements
        for i in range(1, len(data)):
            # High call volume relative to puts might indicate bullish sentiment
            if (data['Call_Volume'].iloc[i] > data['Put_Volume'].iloc[i] * 1.5 and
                data['Option_Volume_Ratio'].iloc[i] > data['Option_Volume_Ratio'].iloc[i-1]):
                data.iloc[i, data.columns.get_loc('Option_Signal')] = 1  # Buy Call
                
            # High put volume relative to calls might indicate bearish sentiment
            elif (data['Put_Volume'].iloc[i] > data['Call_Volume'].iloc[i] * 1.5 and
                  data['Option_Volume_Ratio'].iloc[i] > data['Option_Volume_Ratio'].iloc[i-1]):
                data.iloc[i, data.columns.get_loc('Option_Signal')] = 2  # Buy Put
        
        return data 