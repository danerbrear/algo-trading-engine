import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import sys
import os
from typing import Optional
# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

# Add try/except for linter compatibility
try:
    from model.market_state_classifier import MarketStateClassifier
    from model.options_handler import OptionsHandler as LegacyOptionsHandler
    from model.calendar_features import CalendarFeatureProcessor
    from common.cache.cache_manager import CacheManager
    from common.options_handler import OptionsHandler
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(src_dir, '..'))
    from src.model.market_state_classifier import MarketStateClassifier
    from src.model.options_handler import OptionsHandler as LegacyOptionsHandler
    from src.model.calendar_features import CalendarFeatureProcessor
    from src.common.cache.cache_manager import CacheManager
    from src.common.options_handler import OptionsHandler
from src.common.models import TreasuryRates

class DataRetriever:
    """Handles data retrieval, feature calculation, and preparation for LSTM and HMM models."""
    
    def __init__(self, symbol='SPY', hmm_start_date='2010-01-01', lstm_start_date='2020-01-01', use_free_tier=False, quiet_mode=True):
        """Initialize DataRetriever with separate date ranges for HMM and LSTM
        
        Args:
            symbol: Stock symbol to analyze
            hmm_start_date: Start date for HMM training data (market state classification)
            lstm_start_date: Start date for LSTM training data (options signal prediction)
            use_free_tier: Whether to use free tier rate limiting (13 second timeout)
            quiet_mode: Whether to suppress detailed output for cleaner progress display
        """
        self.symbol = symbol
        self.hmm_start_date = hmm_start_date
        self.lstm_start_date = lstm_start_date
        self.start_date = lstm_start_date  # Backward compatibility
        self.scaler = StandardScaler()
        self.data = None
        self.hmm_data = None  # Separate data for HMM training
        self.lstm_data = None  # Separate data for LSTM training
        self.features = None
        self.ticker = None
        self.cache_manager = CacheManager()
        
        # Create BOTH handlers:
        # 1. Legacy handler for LSTM training (has calculate_option_features, calculate_option_signals)
        self._lstm_options_handler = LegacyOptionsHandler(symbol, start_date=lstm_start_date, cache_dir=self.cache_manager.base_dir, use_free_tier=use_free_tier, quiet_mode=quiet_mode)
        
        # 2. Modern handler for backtesting strategies (has get_contract_list_for_date)
        # Note: Modern handler doesn't need start_date or quiet_mode parameters
        self.options_handler = OptionsHandler(symbol, cache_dir=self.cache_manager.base_dir, use_free_tier=use_free_tier)
        
        self.calendar_processor = None  # Initialize lazily when needed
        self.options_data = {}  # Store OptionChain DTOs for each date
        self.treasury_rates: Optional[TreasuryRates] = None  # Store treasury rates data

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

    def prepare_data_for_lstm(self, sequence_length=60, state_classifier=None):
        """Prepare data for LSTM model with enhanced features using separate date ranges
        
        Args:
            sequence_length: Length of sequences for LSTM
            state_classifier: Trained MarketStateClassifier instance for market state prediction
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, OptionChain]]: 
                - lstm_data: DataFrame with calculated features for LSTM training
                - options_data: Dictionary mapping date strings to OptionChain DTOs
        """
        print(f"\n📊 Phase 1: Preparing LSTM training data from {self.lstm_start_date}")
        # Fetch LSTM training data (more recent data for options trading)
        self.lstm_data = self.fetch_data_for_period(self.lstm_start_date, 'lstm')
        self.calculate_features_for_data(self.lstm_data)

        # Calculate option features for LSTM data
        self.lstm_data, self.options_data = self._lstm_options_handler.calculate_option_features(self.lstm_data, min_dte=5, max_dte=10)

        print(f"\n🔮 Phase 2: Applying trained HMM to LSTM data")
        # Apply the trained HMM to the LSTM data
        if state_classifier is not None:
            self.lstm_data['Market_State'] = state_classifier.predict_states(self.lstm_data)
        else:
            print("⚠️  No state classifier provided, skipping market state prediction")
            self.lstm_data['Market_State'] = 0  # Default state

        print(f"\n💰 Phase 3: Generating option signals for LSTM data")
        # Calculate option trading signals for LSTM data
        self.lstm_data = self._lstm_options_handler.calculate_option_signals(self.lstm_data)

        print(f"\n📅 Phase 4: Adding economic calendar features")
        # Add all calendar features at once (CPI and CC)
        if self.calendar_processor is None:
            self.calendar_processor = CalendarFeatureProcessor()
        self.lstm_data = self.calendar_processor.calculate_all_features(self.lstm_data)

        # Use LSTM data as the main dataset for training
        self.data = self.lstm_data

        return self.lstm_data, self.options_data

    def fetch_data_for_period(self, start_date: str, data_type: str = 'general'):
        """Fetch data for a specific period with caching"""
        cache_suffix = f'_{data_type}_data'
        
        # Try to load from cache first
        cached_data = self.cache_manager.load_date_from_cache(
            pd.Timestamp(start_date),
            cache_suffix,
            'stocks',
            self.symbol
        )
        
        if cached_data is not None:
            print(f"📋 Loading cached {data_type.upper()} data from {start_date} ({len(cached_data)} samples)")
            return cached_data
            
        print(f"🌐 Fetching {data_type.upper()} data from {start_date} onwards...")
        if self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        
        # Always fetch full history and filter manually (yfinance bug workaround)
        data = self.ticker.history(period='max')
        if data.empty:
            raise ValueError(f"No data retrieved for {self.symbol} from {start_date}")
        
        data.index = data.index.tz_localize(None)
        print(f"📊 Initial {data_type} data range: {data.index[0]} to {data.index[-1]}")
        
        # Filter data to start from the specified date
        start_date_ts = pd.Timestamp(start_date).tz_localize(None)
        mask = data.index >= start_date_ts
        data = data[mask].copy()
        print(f"✂️ Filtered {data_type} data range: {data.index[0]} to {data.index[-1]} ({len(data)} samples)")

        if data.empty:
            raise ValueError(f"No data available after filtering for dates >= {start_date}")

        # Cache the filtered data
        self.cache_manager.save_date_to_cache(
            pd.Timestamp(start_date),
            data,
            cache_suffix,
            'stocks',
            self.symbol
        )
        
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
        data['Volume_Change'] = data['Volume'].pct_change()  # Required for HMM
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
            
        # Try Polygon API first (if available and has paid plan)
        if hasattr(self, '_lstm_options_handler') and self._lstm_options_handler:
            try:
                client = self._lstm_options_handler.client
                
                # Try snapshot endpoint first
                try:
                    snapshot = client.get_snapshot_ticker(ticker=symbol, market_type='stocks')
                    if snapshot and hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                        # Use last quote bid/ask midpoint as live price
                        bid = float(snapshot.last_quote.bid) if snapshot.last_quote.bid else 0
                        ask = float(snapshot.last_quote.ask) if snapshot.last_quote.ask else 0
                        if bid > 0 and ask > 0:
                            live_price = (bid + ask) / 2
                            print(f"✅ Live price for {symbol} (from Polygon quote): ${live_price:.2f}")
                            return live_price
                    elif snapshot and hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                        # Fallback to last trade price
                        live_price = float(snapshot.last_trade.price)
                        print(f"✅ Live price for {symbol} (from Polygon trade): ${live_price:.2f}")
                        return live_price
                except Exception as snapshot_error:
                    print(f"⚠️ Polygon snapshot endpoint failed: {str(snapshot_error)}")
                
                # Fallback to last trade endpoint
                try:
                    last_trade = client.get_last_trade(symbol)
                    if last_trade and hasattr(last_trade, 'price'):
                        live_price = float(last_trade.price)
                        print(f"✅ Live price for {symbol} (from Polygon last trade): ${live_price:.2f}")
                        return live_price
                except Exception as trade_error:
                    print(f"⚠️ Polygon last trade endpoint failed: {str(trade_error)}")
                    
            except Exception as polygon_error:
                print(f"⚠️ Polygon API error: {str(polygon_error)}")
        
        # Fallback to yfinance for live price (free)
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
 