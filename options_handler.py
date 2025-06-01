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

# Load environment variables from .env file
load_dotenv()

class OptionsHandler:
    def __init__(self, symbol: str, api_key: Optional[str] = None, cache_dir: str = 'data_cache', start_date: str = None):
        """Initialize the OptionsHandler with a symbol and optional Polygon.io API key"""
        self.symbol = symbol
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon.io API key is required.")
            
        self.cache_dir = Path(cache_dir)
        self.options_cache_dir = self.cache_dir / 'options' / symbol
        self.options_cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = RESTClient(self.api_key)
        # Convert start_date to naive datetime for consistent comparison
        self.start_date = pd.Timestamp(start_date).tz_localize(None) if start_date else None
        
    def _get_cache_path(self, date: datetime, suffix: str = '') -> Path:
        """Get the cache file path for a given date"""
        date_str = date.strftime('%Y-%m-%d')
        return self.options_cache_dir / f"{date_str}{suffix}.pkl"

    def _get_contracts_from_cache(self, current_date: datetime) -> Optional[List]:
        """Get the list of contracts from cache if available"""
        cache_path = self._get_cache_path(current_date, '_contracts')
        if cache_path.exists():
            print(f"Loading cached contracts list for {current_date.date()}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_contracts_to_cache(self, current_date: datetime, contracts: List) -> None:
        """Save the list of contracts to cache"""
        cache_path = self._get_cache_path(current_date, '_contracts')
        with open(cache_path, 'wb') as f:
            pickle.dump(contracts, f)

    def _get_option_chain_with_cache(self, current_date: datetime) -> Dict:
        """Get option chain data for a date, using cache if available"""
        chain_cache_path = self._get_cache_path(current_date)
        
        if chain_cache_path.exists():
            print(f"Loading cached option data for {current_date.date()}")
            with open(chain_cache_path, 'rb') as f:
                return pickle.load(f)
                
        print(f"Fetching option data for {current_date.date()}")
        
        max_retries = 3
        retry_delay = 60  # seconds
        chain_data = {'calls': [], 'puts': []}
        
        for attempt in range(max_retries):
            if attempt > 0:  # Wait before any retry attempt
                print(f"Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                
            try:
                # Try to get contracts from cache first
                contracts = self._get_contracts_from_cache(current_date)
                if contracts is None:
                    # Get all available contracts for the date and convert generator to list
                    contracts = list(self.client.list_options_contracts(
                        underlying_ticker=self.symbol,
                        as_of=current_date.strftime('%Y-%m-%d')
                    ))
                    # Cache the contracts list if we got any
                    if contracts:
                        self._save_contracts_to_cache(current_date, contracts)
                
                if not contracts:
                    print(f"No options data available for {current_date.date()}")
                    return chain_data
                
                # Process each contract
                for contract in contracts:
                    try:
                        aggs = list(self.client.get_aggs(
                            ticker=contract.ticker,
                            multiplier=1,
                            timespan="day",
                            from_=current_date.strftime('%Y-%m-%d'),
                            to=current_date.strftime('%Y-%m-%d'),
                            limit=1
                        ))
                        
                        if not aggs:
                            continue  # No data available, skip to next contract
                            
                        day_data = aggs[0]
                        
                        contract_data = {
                            'strike': float(contract.strike_price),
                            'expiration': contract.expiration_date,  # Already in YYYY-MM-DD format
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
                        
                        if contract_data['type'] == 'call':
                            chain_data['calls'].append(contract_data)
                        else:
                            chain_data['puts'].append(contract_data)
                            
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 429:  # Rate limit exceeded
                            print(f"Rate limit exceeded for {contract.ticker}, waiting {retry_delay} seconds")
                            time.sleep(retry_delay)
                            continue
                        print(f"Error fetching historical data for {contract.ticker}: {str(e)}")
                    except Exception as e:
                        print(f"Error fetching historical data for {contract.ticker}: {str(e)}")
                
                # If we got here without a rate limit error, we can break the retry loop
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        print(f"Rate limit exceeded for options list")
                        continue
                    else:
                        print(f"Max retries ({max_retries}) exceeded for options list")
                raise  # Re-raise the exception if we've exhausted our retries
            except Exception as e:
                print(f"Error processing {current_date}: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    return chain_data
        
        # Cache the data if we got any contracts
        if chain_data['calls'] or chain_data['puts']:
            with open(chain_cache_path, 'wb') as f:
                pickle.dump(chain_data, f)
        
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
                chain_data = self._get_option_chain_with_cache(current_date)
                
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