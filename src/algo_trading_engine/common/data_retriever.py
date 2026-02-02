import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import sys
import os
from typing import Optional
# Add the algo_trading_engine directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, '..')
sys.path.insert(0, package_dir)

# Add try/except for linter compatibility
try:
    from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier
    from algo_trading_engine.ml_models.calendar_features import CalendarFeatureProcessor
    from algo_trading_engine.common.cache.cache_manager import CacheManager
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(package_dir, '..'))
    from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier
    from algo_trading_engine.ml_models.calendar_features import CalendarFeatureProcessor
    from algo_trading_engine.common.cache.cache_manager import CacheManager
from algo_trading_engine.common.models import TreasuryRates
from algo_trading_engine.enums import BarTimeInterval
from pathlib import Path

class DataRetriever:
    """Handles data retrieval, feature calculation, and preparation for LSTM and HMM models."""
    
    def __init__(
        self, 
        symbol='SPY', 
        hmm_start_date='2010-01-01', 
        lstm_start_date='2020-01-01', 
        use_free_tier=False, 
        quiet_mode=True,
        bar_interval: BarTimeInterval = BarTimeInterval.DAY
    ):
        """Initialize DataRetriever with separate date ranges for HMM and LSTM
        
        Args:
            symbol: Stock symbol to analyze
            hmm_start_date: Start date for HMM training data (market state classification)
            lstm_start_date: Start date for LSTM training data (options signal prediction)
            use_free_tier: Whether to use free tier rate limiting (13 second timeout)
            quiet_mode: Whether to suppress detailed output for cleaner progress display
            bar_interval: Time interval for market data bars (DAY, HOUR, or MINUTE)
        """
        self.symbol = symbol
        self.hmm_start_date = hmm_start_date
        self.lstm_start_date = lstm_start_date
        self.start_date = lstm_start_date  # Backward compatibility
        self.bar_interval = bar_interval
        self.scaler = StandardScaler()
        self.data = None
        self.hmm_data = None  # Separate data for HMM training
        self.lstm_data = None  # Separate data for LSTM training
        self.features = None
        self.ticker = None
        self.cache_manager = CacheManager()
        self.calendar_processor = None  # Initialize lazily when needed
        self.treasury_rates: Optional[TreasuryRates] = None  # Store treasury rates data
    
    def _get_yfinance_interval(self) -> str:
        """Convert BarTimeInterval enum to yfinance interval string."""
        interval_map = {
            BarTimeInterval.MINUTE: "1m",
            BarTimeInterval.HOUR: "1h",
            BarTimeInterval.DAY: "1d",
        }
        return interval_map[self.bar_interval]
    
    def _get_cache_interval_dir(self) -> str:
        """Get the cache subdirectory name for the current bar interval."""
        interval_dir_map = {
            BarTimeInterval.MINUTE: "minute",
            BarTimeInterval.HOUR: "hourly",
            BarTimeInterval.DAY: "daily",
        }
        return interval_dir_map[self.bar_interval]

    def load_treasury_rates(self, start_date: datetime, end_date: datetime = None):
        """
        Load treasury rates from cache for the specified date range.
        
        Args:
            start_date: Start date for treasury data
            end_date: End date for treasury data (optional, defaults to start_date)
        """
        if end_date is None:
            end_date = start_date
            
        print(f"📈 Loading treasury rates for date range: {start_date.date()} to {end_date.date()}")
        
        # Try to load treasury rates for the start date
        treasury_data = self.cache_manager.load_date_from_cache(
            start_date, '_treasury_rates', 'treasury', 'rates'
        )
        
        if treasury_data is not None:
            self.treasury_rates = TreasuryRates(treasury_data)
            print(f"✅ Loaded treasury rates for {start_date.date()}")
        else:
            # Fallback: try to find the closest available treasury data
            self._find_closest_treasury_data(start_date)

    def _find_closest_treasury_data(self, target_date: datetime):
        """
        Find the closest available treasury data file to the target date.
        """
        treasury_dir = self.cache_manager.get_cache_dir('treasury', 'rates')
        available_files = list(treasury_dir.glob('*_treasury_rates.pkl'))
        
        if not available_files:
            print("⚠️  No treasury rate files found in cache")
            return
            
        # Parse dates from filenames and find the closest
        closest_file = None
        min_date_diff = float('inf')
        
        for file_path in available_files:
            try:
                date_str = file_path.stem.split('_')[0]  # Extract date from filename
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                date_diff = abs((target_date - file_date).days)
                
                if date_diff < min_date_diff:
                    min_date_diff = date_diff
                    closest_file = file_path
            except (ValueError, IndexError):
                continue
        
        if closest_file:
            treasury_data = self.cache_manager.load_from_cache(
                closest_file.name, 'treasury', 'rates'
            )
            if treasury_data is not None:
                self.treasury_rates = TreasuryRates(treasury_data)
                print(f"✅ Loaded closest treasury rates from {closest_file.name} (diff: {min_date_diff} days)")
        else:
            print("⚠️  Could not find any treasury rate files")
    
    def _load_cached_data_range(self, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load cached data for a date range from the interval-specific cache directory.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with cached data, or None if cache is incomplete
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        interval_dir = self._get_cache_interval_dir()
        cache_base = self.cache_manager.get_cache_dir('stocks', self.symbol)
        cache_dir = cache_base / interval_dir
        
        if not cache_dir.exists():
            return None
        
        # For daily bars, use single file per start_date (like old implementation)
        if self.bar_interval == BarTimeInterval.DAY:
            # Look for a cache file that matches this start_date
            cache_file = cache_dir / f"{start_date}.pkl"
            if cache_file.exists():
                try:
                    data = pd.read_pickle(cache_file)
                    # Filter to requested end_date if needed
                    end_ts = pd.Timestamp(end_date)
                    if data.index[-1] < end_ts:
                        # Cache doesn't have enough data, need to re-fetch
                        return None
                    # Filter to exact range
                    data = data[data.index <= end_ts]
                    return data
                except Exception as e:
                    print(f"⚠️  Failed to load {cache_file.name}: {e}")
                    return None
            return None
        
        # For hourly/minute bars, load and concatenate granular files
        cache_files = sorted(cache_dir.glob('*.pkl'))
        
        if not cache_files:
            return None
        
        # Load and concatenate all cache files within the date range
        dfs = []
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        for cache_file in cache_files:
            try:
                # Extract date from filename
                # Format: YYYY-MM-DD_HHMM.pkl (hourly/minute)
                file_date_str = cache_file.stem.split('_')[0]  # Get date part
                file_date = pd.Timestamp(file_date_str)
                
                if start_ts <= file_date <= end_ts:
                    df = pd.read_pickle(cache_file)
                    dfs.append(df)
            except Exception as e:
                # If any file fails to load, return None to trigger re-fetch
                print(f"⚠️  Failed to load {cache_file.name}: {e}")
                return None
        
        if not dfs:
            return None
        
        # Concatenate and sort by index
        result = pd.concat(dfs).sort_index()
        return result

    def prepare_data_for_lstm(self, sequence_length=60, state_classifier=None):
        """Prepare data for LSTM model with basic features (no options features)
        
        Note: This method now only prepares basic technical and calendar features.
        Options-specific features should be calculated by the strategy that needs them
        (e.g., CreditSpreadStrategy.__init__).
        
        Args:
            sequence_length: Length of sequences for LSTM
            state_classifier: Trained MarketStateClassifier instance for market state prediction
            
        Returns:
            pd.DataFrame: DataFrame with calculated features for LSTM training
        """
        print(f"\n📊 Phase 1: Preparing LSTM training data from {self.lstm_start_date}")
        # Fetch LSTM training data (more recent data for options trading)
        self.lstm_data = self.fetch_data_for_period(self.lstm_start_date)
        self.calculate_features_for_data(self.lstm_data)

        print(f"\n🔮 Phase 2: Applying trained HMM to LSTM data")
        # Apply the trained HMM to the LSTM data
        if state_classifier is not None:
            self.lstm_data['Market_State'] = state_classifier.predict_states(self.lstm_data)
        else:
            print("⚠️  No state classifier provided, skipping market state prediction")
            self.lstm_data['Market_State'] = 0  # Default state

        print(f"\n📅 Phase 3: Adding economic calendar features")
        # Add all calendar features at once (CPI and CC)
        if self.calendar_processor is None:
            self.calendar_processor = CalendarFeatureProcessor()
        self.lstm_data = self.calendar_processor.calculate_all_features(self.lstm_data)

        # Use LSTM data as the main dataset for training
        self.data = self.lstm_data

        return self.lstm_data

    def fetch_data_for_period(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch data for a specific period with caching.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with OHLCV data for the specified period
            
        Note: Caching is now process-agnostic using interval-based subdirectories.
        """
        # Try to load from cache first
        cached_data = self._load_cached_data_range(start_date, end_date)
        
        if cached_data is not None:
            interval_str = self._get_yfinance_interval()
            print(f"📋 Loaded {len(cached_data)} cached {interval_str} bars")
            return cached_data
        
        # Fetch from yfinance
        interval_str = self._get_yfinance_interval()
        interval_dir = self._get_cache_interval_dir()
        
        print(f"🌐 Fetching {interval_str} bars from {start_date} onwards...")
        
        if self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        
        # Validate ticker exists
        info = self.ticker.info
        if not info or len(info) == 0:
            raise ValueError(f"Invalid symbol or no info available for {self.symbol}")
        
        # Determine end date
        if end_date is None:
            end_date_ts = pd.Timestamp.now()
        else:
            end_date_ts = pd.Timestamp(end_date)
        
        start_date_ts = pd.Timestamp(start_date)
        
        # Fetch with interval parameter
        data = self.ticker.history(
            start=start_date_ts,
            end=end_date_ts,
            interval=interval_str
        )
        
        if data.empty:
            raise ValueError(
                f"No data retrieved for {self.symbol} from {start_date} with interval '{interval_str}'. "
                f"This could be due to:\n"
                f"  1. Invalid date range for this interval\n"
                f"  2. Network/API connectivity issues\n"
                f"  3. Rate limiting from Yahoo Finance"
            )
        
        data.index = data.index.tz_localize(None)
        print(f"📊 Fetched {len(data)} {interval_str} bars from {data.index[0]} to {data.index[-1]}")
        
        # Filter to requested date range
        data = data[(data.index >= start_date_ts) & (data.index <= end_date_ts)].copy()
        
        if data.empty:
            raise ValueError(f"No data available after filtering for date range {start_date} to {end_date_ts.date()}")
        
        # Save to cache using new structure
        cache_base = self.cache_manager.get_cache_dir('stocks', self.symbol)
        cache_dir = cache_base / interval_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.bar_interval == BarTimeInterval.DAY:
            # For daily bars, save one file per start_date (like old implementation)
            # This avoids reading hundreds of files for large backtests
            cache_file = cache_dir / f"{start_date}.pkl"
            data.to_pickle(cache_file)
            print(f"💾 Cached {len(data)} daily bars to {interval_dir}/{start_date}.pkl")
        else:
            # For hourly/minute, save each bar separately (granular caching)
            # This makes sense for intraday data with smaller date ranges
            for idx, row in data.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                time_str = idx.strftime('%H%M')  # e.g., '0930' for 9:30 AM
                cache_file = cache_dir / f"{date_str}_{time_str}.pkl"
                
                # Save single row as DataFrame
                pd.DataFrame([row]).to_pickle(cache_file)
                
            print(f"💾 Cached {len(data)} {interval_str} bars to {interval_dir}/")
        
        return data

    def calculate_features_for_data(self, data: pd.DataFrame, window=20):
        """Calculate technical features for a given dataset"""
        # Check if we have enough data for proper feature calculation
        min_required_samples = max(window, 50) + 10  # Need enough for rolling windows plus buffer

        if len(data) < min_required_samples:
            raise ValueError(
                f"Insufficient data for feature calculation: {len(data)} samples available, "
                f"need at least {min_required_samples} samples. "
                f"This ensures proper calculation of technical indicators like SMA50, RSI, and MACD."
            )

        # Standard feature calculation for adequate datasets
        # Basic returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility measures
        data['Volatility'] = data['Returns'].rolling(window=window, min_periods=1).std()
        data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
        
        # Trend indicators
        data['SMA20'] = data['Close'].rolling(window=window, min_periods=1).mean()
        data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['Price_to_SMA20'] = data['Close'] / data['SMA20']  # Keep for HMM, exclude from LSTM
        data['SMA20_to_SMA50'] = data['SMA20'] / data['SMA50']
        
        # Momentum indicators
        data['RSI'] = self._calculate_rsi(data['Close'], window)
        # Calculate MACD components for MACD_Hist (removed standalone MACD feature)
        macd_line, macd_signal = self._calculate_macd(data['Close'])
        data['MACD_Hist'] = macd_line - macd_signal
        
        # Time series features for SMA20_to_SMA50
        data['SMA20_to_SMA50_Lag1'] = data['SMA20_to_SMA50'].shift(1)
        data['SMA20_to_SMA50_Lag5'] = data['SMA20_to_SMA50'].shift(5)
        data['SMA20_to_SMA50_MA5'] = data['SMA20_to_SMA50'].rolling(window=5, min_periods=1).mean()
        data['SMA20_to_SMA50_MA10'] = data['SMA20_to_SMA50'].rolling(window=10, min_periods=1).mean()
        data['SMA20_to_SMA50_Std5'] = data['SMA20_to_SMA50'].rolling(window=5, min_periods=1).std()
        data['SMA20_to_SMA50_Momentum'] = data['SMA20_to_SMA50'] - data['SMA20_to_SMA50'].shift(5)
        
        # Time series features for RSI
        data['RSI_Lag1'] = data['RSI'].shift(1)
        data['RSI_Lag5'] = data['RSI'].shift(5)
        data['RSI_MA5'] = data['RSI'].rolling(window=5, min_periods=1).mean()
        data['RSI_MA10'] = data['RSI'].rolling(window=10, min_periods=1).mean()
        data['RSI_Std5'] = data['RSI'].rolling(window=5, min_periods=1).std()
        data['RSI_Momentum'] = data['RSI'] - data['RSI'].shift(5)
        data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
        data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
        
        # Volume features
        data['Volume_SMA'] = data['Volume'].rolling(window=window, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['OBV'] = self._calculate_obv(data['Close'], data['Volume'])
        
        # Drop NaN values
        original_length = len(data)
        data.dropna(inplace=True)
        
        if len(data) == 0:
            raise ValueError(
                f"All data was dropped due to NaN values after feature calculation. "
                f"This indicates data quality issues or insufficient data for proper technical analysis."
            )
        
        if len(data) < original_length * 0.8:  # If we lost more than 20% of data
            print(f"⚠️  Warning: Lost {original_length - len(data)} samples due to NaN values")
            print(f"   Keeping {len(data)} samples")
        
        print(f"✅ Calculated features for {len(data)} samples")

    

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            raise ValueError(
                f"Insufficient data for RSI calculation: {len(prices)} samples available, "
                f"need at least {window + 1} samples for window size {window}."
            )
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_obv(self, prices, volume):
        """Calculate On-Balance Volume"""
        price_change = prices.diff()
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(prices)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def get_live_price(self, symbol: str = None) -> Optional[float]:
        """Fetch live price for the current date.
        
        First tries Polygon API (if available), then falls back to yfinance.
        
        Args:
            symbol: Stock symbol to fetch price for (defaults to self.symbol)
            
        Returns:
            Optional[float]: Live price if successful, None otherwise
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            print(f"📡 Fetching live price for {symbol} using yfinance...")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try to get current price from info
            if 'currentPrice' in info and info['currentPrice'] is not None:
                live_price = float(info['currentPrice'])
                print(f"✅ Live price for {symbol} (from yfinance): ${live_price:.2f}")
                return live_price
            elif 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                live_price = float(info['regularMarketPrice'])
                print(f"✅ Live price for {symbol} (from yfinance regular market): ${live_price:.2f}")
                return live_price
            elif 'previousClose' in info and info['previousClose'] is not None:
                # Use previous close as fallback
                live_price = float(info['previousClose'])
                print(f"⚠️ Using previous close for {symbol} (from yfinance): ${live_price:.2f}")
                return live_price
            else:
                print(f"⚠️ No price data available from yfinance for {symbol}")
                return None
                
        except Exception as yfinance_error:
            print(f"❌ Error fetching live price from yfinance for {symbol}: {str(yfinance_error)}")
            return None
 