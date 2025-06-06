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

    def _fetch_historical_contracts_data(self, contracts: List, current_date: datetime, current_price: float) -> Dict:
        """Fetch historical data for only the most relevant contracts (closest to 30-day expiry and ATM strikes)"""
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
            
            # Step 4: Try to get valid call data with fallback
            call_data = None
            if calls_sorted:
                try:
                    for i, call_contract in enumerate(calls_sorted):
                        print(f"Trying Call #{i+1}: strike ${float(call_contract.strike_price):.2f}, symbol {call_contract.ticker}")
                        call_data = self._fetch_historical_contract_data(call_contract, current_date)
                        if call_data:
                            print(f"✓ Successfully got call data for strike ${float(call_contract.strike_price):.2f}")
                            chain_data['calls'].append(call_data)
                            break
                        else:
                            print(f"✗ No historical data available, trying next closest call...")
                            if i >= 4:  # Limit to trying 5 closest options
                                print(f"Tried {i+1} calls, stopping search")
                                break
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                        return chain_data  # Return empty data immediately
                    raise  # Re-raise other ValueErrors
                
                if not call_data:
                    print("Could not find any call with historical data")
            
            # Step 5: Try to get valid put data with fallback  
            put_data = None
            if puts_sorted:
                try:
                    for i, put_contract in enumerate(puts_sorted):
                        print(f"Trying Put #{i+1}: strike ${float(put_contract.strike_price):.2f}, symbol {put_contract.ticker}")
                        put_data = self._fetch_historical_contract_data(put_contract, current_date)
                        if put_data:
                            print(f"✓ Successfully got put data for strike ${float(put_contract.strike_price):.2f}")
                            chain_data['puts'].append(put_data)
                            break
                        else:
                            print(f"✗ No historical data available, trying next closest put...")
                            if i >= 4:  # Limit to trying 5 closest options
                                print(f"Tried {i+1} puts, stopping search")
                                break
                except ValueError as e:
                    if "SKIP_DATE_UNAUTHORIZED" in str(e):
                        print(f"🚫 Skipping {current_date.date()} - Plan doesn't include this timeframe")
                        return chain_data  # Return empty data immediately
                    raise  # Re-raise other ValueErrors
                
                if not put_data:
                    print("Could not find any put with historical data")
            
            # Summary
            successful_contracts = len(chain_data['calls']) + len(chain_data['puts'])
            print(f"Successfully fetched historical data for {successful_contracts}/2 target contracts")
                        
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
                print(f"ATM Call: ${atm_call['strike']:.0f} exp {atm_call['expiration']} @ ${atm_call['last_price']:.2f}")
                print(f"ATM Put:  ${atm_put['strike']:.0f} exp {atm_put['expiration']} @ ${atm_put['last_price']:.2f}")
                    
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

    def calculate_option_signals(self, data: pd.DataFrame, holding_period: int = 5, min_return_threshold: float = 0.05) -> pd.DataFrame:
        """Calculate trading signals based on sophisticated options strategies
        
        Strategy Classes:
        0: Hold - No position
        1: Call Debit Spread - Buy lower strike call, sell higher strike call (bullish)
        2: Put Debit Spread - Buy higher strike put, sell lower strike put (bearish) 
        3: Iron Condor - Sell OTM call spread + sell OTM put spread (neutral, low volatility)
        4: Iron Butterfly - Sell ATM straddle + buy protective wings (neutral, very low volatility)
        
        Args:
            holding_period: Number of days to hold the strategy (default: 5)
            min_return_threshold: Minimum return required to generate a signal (default: 5%)
        """
        print(f"Generating strategy labels based on {holding_period}-day forward returns with {min_return_threshold:.1%} minimum threshold")
        
        # Initialize the signal column
        data['Option_Signal'] = 0  # Default to Hold
        
        # Calculate returns for ATM options
        data['ATM_Call_Return'] = data['Call_Price'].pct_change(fill_method=None)
        data['ATM_Put_Return'] = data['Put_Price'].pct_change(fill_method=None)
        
        # We'll calculate strategy returns based on simplified models
        # In practice, you'd need full option chains with multiple strikes
        
        # Simplified strategy return calculations using ATM options as proxies
        # Call Debit Spread: Long call profit - short call loss (improved calculation)
        data['Future_Call_Debit_Return'] = (data['Call_Price'].shift(-holding_period) / data['Call_Price'] - 1) * 0.75  # Less penalty (was 0.6)
        
        # Put Debit Spread: Long put profit - short put loss (improved calculation)  
        data['Future_Put_Debit_Return'] = (data['Put_Price'].shift(-holding_period) / data['Put_Price'] - 1) * 0.75  # Less penalty (was 0.6)
        
        # Iron Condor: Profits when price stays between strikes (balanced conditions)
        data['Price_Change'] = abs(data['Close'].shift(-holding_period) / data['Close'] - 1)
        data['Future_Iron_Condor_Return'] = np.where(
            data['Price_Change'] < 0.03,  # Profits when price moves less than 3%
            0.12,  # Moderate return when successful
            -0.50   # Moderate loss when price moves too much
        )
        
        # Iron Butterfly: Profits when price stays very close to current price (more restrictive)
        data['Future_Iron_Butterfly_Return'] = np.where(
            data['Price_Change'] < 0.015,  # More restrictive: less than 1.5% (was 2.5%)
            0.20,  # Good return for tight range
            -0.60   # Moderate loss when price moves
        )
        
        # Stock return for comparison
        data['Future_Stock_Return'] = data['Close'].shift(-holding_period) / data['Close'] - 1
        
        # Strategy counters
        strategy_counts = [0, 0, 0, 0, 0]  # Hold, Call Debit, Put Debit, Iron Condor, Iron Butterfly
        
        # Generate labels based on best performing strategy
        for i in range(len(data) - holding_period):  # Exclude last holding_period rows
            
            # Get future returns for each strategy
            call_debit_return = data['Future_Call_Debit_Return'].iloc[i]
            put_debit_return = data['Future_Put_Debit_Return'].iloc[i]
            iron_condor_return = data['Future_Iron_Condor_Return'].iloc[i]
            iron_butterfly_return = data['Future_Iron_Butterfly_Return'].iloc[i]
            
            # Skip if we don't have option data
            if pd.isna(call_debit_return) or pd.isna(put_debit_return):
                continue
            
            # Find the best strategy that meets minimum threshold
            strategy_returns = {
                1: call_debit_return,      # Call Debit Spread
                2: put_debit_return,       # Put Debit Spread  
                3: iron_condor_return,     # Iron Condor
                4: iron_butterfly_return   # Iron Butterfly
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
        strategy_names = ['Hold', 'Call Debit Spread', 'Put Debit Spread', 'Iron Condor', 'Iron Butterfly']
        total_signals = sum(strategy_counts)
        
        if total_signals > 0:
            print(f"\nStrategy Distribution:")
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
                        avg_return = data[strategy_mask]['Future_Iron_Butterfly_Return'].mean()
                    
                    print(f"  Avg {strategy_names[strategy_id]} Return: {avg_return:.2%}")
        
        return data 