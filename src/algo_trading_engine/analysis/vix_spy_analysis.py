"""
VIX Big Move and SPY Return Analysis

This module analyzes SPY returns following big moves in VIX.
A "big move" is defined as a VIX return greater than a specified threshold
(default: 50%) over 1-3 days, AND SPY must be down on the same day.

The analysis calculates:
- Average SPY returns for 3, 6, and 12 days following big VIX moves
- Frequency distribution of SPY log returns (rounded to nearest 0.2%)
- Optional lag period: delay SPY return calculation by N days after VIX move
- Optional market state filter: include only events in specified market state(s)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import data retrieval classes
from algo_trading_engine.common.data_retriever import DataRetriever
from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier


@dataclass
class BigMoveEvent:
    """Represents a big move event in VIX"""
    date: pd.Timestamp
    vix_log_return: float
    days_in_move: int  # 1, 2, or 3 days
    vix_std_devs: float  # How many std deviations above mean
    spy_return_3d: float
    spy_return_6d: float
    spy_return_12d: float


@dataclass
class ReturnStats:
    """Statistics for SPY returns following big VIX moves"""
    days_forward: int
    average_return: float
    median_return: float
    total_events: int
    frequency_distribution: Dict[float, int]  # Return bucket -> count


class VIXSPYAnalyzer:
    """
    Analyzes SPY returns following big moves in VIX.
    
    Big move definition: VIX log return > 2 standard deviations from mean
    over 1-3 day periods, with cooldown filter to ensure independent signals.
    """
    
    def __init__(self, lookback_years: int = 5, cooldown_days: int = 12, lag_days: int = 0, return_threshold: float = 0.50):
        """
        Initialize the VIX-SPY Analyzer.
        
        Args:
            lookback_years: Number of years of historical data to analyze (default: 5)
            cooldown_days: Minimum days between independent signals (default: 12)
            lag_days: Number of days to lag SPY return calculation after VIX move (default: 0)
            return_threshold: Minimum return (as decimal) to qualify as big move (default: 0.50 = 50%)
        """
        # Calculate start date
        today = datetime.now()
        start_date = today - timedelta(days=lookback_years * 365)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.lookback_years = lookback_years
        self.cooldown_days = cooldown_days
        self.lag_days = lag_days
        self.return_threshold = return_threshold
        self.return_threshold_log = np.log(1 + return_threshold)  # Convert to log return threshold
        
        # Data retrievers for VIX and SPY
        self.vix_retriever = DataRetriever(symbol='^VIX', lstm_start_date=self.start_date)
        self.spy_retriever = DataRetriever(symbol='SPY', lstm_start_date=self.start_date)
        
        self.vix_data = None
        self.spy_data = None
        self.big_move_events = []
        self.raw_signals = []  # Store raw signals before cooldown filter
        
        # HMM market state classifier
        self.state_classifier = None
        self.market_states = None  # Predicted market states for analysis period
        self.market_state_filter = None  # Market state filter (if set)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load VIX and SPY daily close price data.
        
        Returns:
            Tuple of (VIX DataFrame, SPY DataFrame)
        """
        print(f"Loading VIX and SPY data from {self.start_date}...")
        
        # Fetch VIX data
        self.vix_data = self.vix_retriever.fetch_data_for_period(self.start_date, 'vix_analysis')
        if 'Close' not in self.vix_data.columns:
            raise ValueError("No Close price data available for VIX")
        
        # Fetch SPY data
        self.spy_data = self.spy_retriever.fetch_data_for_period(self.start_date, 'vix_analysis')
        if 'Close' not in self.spy_data.columns:
            raise ValueError("No Close price data available for SPY")
        
        # Align data on common dates
        common_dates = self.vix_data.index.intersection(self.spy_data.index)
        self.vix_data = self.vix_data.loc[common_dates]
        self.spy_data = self.spy_data.loc[common_dates]
        
        print(f"Loaded {len(self.vix_data)} days of aligned VIX and SPY data")
        print(f"   Date range: {self.vix_data.index[0]} to {self.vix_data.index[-1]}")
        
        return self.vix_data, self.spy_data
    
    def _fetch_vix_data(self, start_date: str) -> pd.DataFrame:
        """Fetch VIX data using yfinance (matching run_market_state_classifier.py)"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        print(f"📊 Fetching VIX data from {start_date}...")
        vix_ticker = yf.Ticker('^VIX')
        
        # Try period='max' first, then fallback to date range
        try:
            vix_data = vix_ticker.history(period='max')
        except Exception as e:
            print(f"⚠️  Failed to fetch VIX with period='max', trying date range: {e}")
            start_date_ts = pd.Timestamp(start_date)
            end_date_ts = pd.Timestamp.now()
            vix_data = vix_ticker.history(start=start_date_ts, end=end_date_ts)
        
        if vix_data.empty:
            raise ValueError("No VIX data retrieved")
        
        # Normalize timezone - remove timezone info for easier date comparisons
        if vix_data.index.tz is not None:
            vix_data.index = pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in vix_data.index])
        
        # Filter to start_date if needed
        if start_date:
            start_date_ts = pd.Timestamp(start_date)
            vix_data = vix_data[vix_data.index >= start_date_ts]
        
        print(f"✅ Retrieved {len(vix_data)} VIX samples from {vix_data.index[0].date()} to {vix_data.index[-1].date()}")
        return vix_data
    
    def _calculate_hmm_features(self, data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None, window=20) -> pd.DataFrame:
        """Calculate technical features required for MarketStateClassifier (matching run_market_state_classifier.py)"""
        # Basic returns
        data['Returns'] = data['Close'].pct_change()
        
        # Volatility - use VIX Close price if available, otherwise fallback to calculated volatility
        if vix_data is not None and not vix_data.empty:
            # Merge VIX data with main data
            vix_close = vix_data[['Close']].rename(columns={'Close': 'VIX'})
            data = data.join(vix_close, how='left')
            
            # Use VIX price as Volatility
            data['Volatility'] = data['VIX']
            
            # Forward fill missing VIX values (in case of market holidays)
            data['Volatility'] = data['Volatility'].ffill()
            
            # If still missing, use backward fill
            data['Volatility'] = data['Volatility'].bfill()
            
            # Drop VIX column as we've copied it to Volatility
            data.drop(columns=['VIX'], inplace=True, errors='ignore')
            
            print(f"✅ Using VIX price for Volatility feature")
        else:
            # Fallback to calculated volatility
            data['Volatility'] = data['Returns'].rolling(window=window, min_periods=1).std()
            print(f"⚠️  VIX data not available, using calculated volatility")
        
        # Trend indicators
        data['SMA20'] = data['Close'].rolling(window=window, min_periods=1).mean()
        data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['Price_to_SMA20'] = data['Close'] / data['SMA20']
        data['SMA20_to_SMA50'] = data['SMA20'] / data['SMA50']
        
        # Volume features
        data['Volume_SMA'] = data['Volume'].rolling(window=window, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def train_hmm_model(self, hmm_lookback_years: int = 10, max_states: int = 3):
        """
        Train HMM model on SPY data for market state classification.
        Uses VIX price for volatility feature (matching run_market_state_classifier.py).
        
        Args:
            hmm_lookback_years: Number of years of historical data for HMM training (default: 10)
            max_states: Maximum number of states to evaluate (default: 5)
        """
        print(f"\n🎯 Training HMM model on {hmm_lookback_years} years of SPY data...")
        
        # Calculate HMM training start date
        today = datetime.now()
        hmm_start_date = (today - timedelta(days=hmm_lookback_years * 365)).strftime('%Y-%m-%d')
        
        # Fetch SPY data for HMM training
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        print(f"📈 Fetching SPY data from {hmm_start_date}...")
        spy_ticker = yf.Ticker('SPY')
        
        try:
            spy_data = spy_ticker.history(period='max')
        except Exception as e:
            print(f"⚠️  Failed to fetch with period='max', trying date range: {e}")
            start_date_ts = pd.Timestamp(hmm_start_date)
            end_date_ts = pd.Timestamp.now()
            spy_data = spy_ticker.history(start=start_date_ts, end=end_date_ts)
        
        if spy_data.empty:
            raise ValueError("No SPY data retrieved for HMM training")
        
        # Normalize timezone
        if spy_data.index.tz is not None:
            spy_data.index = pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in spy_data.index])
        
        # Filter to start_date
        if hmm_start_date:
            start_date_ts = pd.Timestamp(hmm_start_date)
            spy_data = spy_data[spy_data.index >= start_date_ts]
        
        print(f"✅ Retrieved {len(spy_data)} SPY samples from {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
        
        if len(spy_data) < 60:
            raise ValueError(f"Insufficient data for HMM training: {len(spy_data)} samples")
        
        # Fetch VIX data for volatility feature
        try:
            vix_data = self._fetch_vix_data(hmm_start_date)
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch VIX data: {e}")
            print(f"   Falling back to calculated volatility")
            vix_data = None
        
        # Calculate features using VIX price for volatility
        print(f"\n📊 Calculating features...")
        hmm_training_data = self._calculate_hmm_features(spy_data.copy(), vix_data=vix_data)
        
        if len(hmm_training_data) < 50:
            raise ValueError(f"Insufficient data after feature calculation: {len(hmm_training_data)} samples. Need at least 50.")
        
        print(f"✅ Prepared {len(hmm_training_data)} samples with features")
        
        # Verify required columns exist
        required_cols = ['Returns', 'Volatility', 'Price_to_SMA20', 'SMA20_to_SMA50', 'Volume_Ratio']
        missing_cols = [col for col in required_cols if col not in hmm_training_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for HMM training: {missing_cols}")
        
        # Analyze feature correlations
        print(f"\n📊 Feature Correlation Analysis:")
        feature_corr = hmm_training_data[required_cols].corr()
        print(f"\nCorrelation Matrix:")
        print(feature_corr.round(3))
        
        # Identify high correlations
        high_corr_pairs = []
        for i, col1 in enumerate(required_cols):
            for col2 in required_cols[i+1:]:
                corr = feature_corr.loc[col1, col2]
                if abs(corr) > 0.7:
                    high_corr_pairs.append((col1, col2, corr))
        
        if high_corr_pairs:
            print(f"\n⚠️  High correlations (>0.7) detected:")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} ↔ {col2}: {corr:.3f}")
            print(f"  (Consider removing one if correlation >0.8)")
        else:
            print(f"\n✅ No high correlations detected - features are well-separated")
        
        # Initialize and train the HMM model
        self.state_classifier = MarketStateClassifier(max_states=max_states)
        optimal_states = self.state_classifier.train_hmm_model(hmm_training_data)
        
        print(f"✅ HMM model trained with {optimal_states} optimal states")
        
        # Print descriptions for each market state
        print(f"\n📊 Market State Descriptions:")
        print("-" * 80)
        
        # Get predicted states for training data to generate descriptions
        training_states = self.state_classifier.predict_states(hmm_training_data)
        
        # Print description for each state
        for state_id in range(optimal_states):
            description = self.state_classifier.get_state_description(
                state_id, hmm_training_data, training_states
            )
            state_count = np.sum(training_states == state_id)
            state_pct = (state_count / len(training_states)) * 100
            print(f"  {description}")
            print(f"    Occurrences: {state_count} ({state_pct:.1f}% of training period)")
            print()
        
        print("-" * 80)
        
        return optimal_states
    
    def predict_market_states(self):
        """
        Predict market states for the current analysis period using trained HMM model.
        Uses VIX price for volatility feature (matching run_market_state_classifier.py).
        
        Returns:
            numpy array of predicted market states
        """
        if self.state_classifier is None or not self.state_classifier.is_trained:
            raise ValueError("HMM model not trained. Call train_hmm_model() first.")
        
        if self.spy_data is None:
            raise ValueError("SPY data not loaded. Call load_data() first.")
        
        print(f"\n🔮 Predicting market states for analysis period...")
        
        # Use existing spy_data and calculate features with VIX
        analysis_data = self.spy_data.copy()
        
        # Fetch VIX data for the analysis period
        try:
            vix_data = self._fetch_vix_data(self.start_date)
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch VIX data: {e}")
            print(f"   Falling back to calculated volatility")
            vix_data = None
        
        # Calculate features using VIX price for volatility
        print(f"\n📊 Calculating features for analysis period...")
        analysis_data = self._calculate_hmm_features(analysis_data, vix_data=vix_data)
        
        # Verify required columns exist
        required_cols = ['Returns', 'Volatility', 'Price_to_SMA20', 'SMA20_to_SMA50', 'Volume_Ratio']
        missing_cols = [col for col in required_cols if col not in analysis_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for state prediction: {missing_cols}")
        
        # Predict states
        predicted_states = self.state_classifier.predict_states(analysis_data)
        
        # Create a Series aligned with analysis_data index
        self.market_states = pd.Series(predicted_states, index=analysis_data.index)
        
        print(f"✅ Predicted market states for {len(self.market_states)} days")
        
        # Print state distribution
        state_counts = self.market_states.value_counts().sort_index()
        print(f"\n📊 Market State Distribution in Analysis Period:")
        print("-" * 80)
        
        for state_id, count in state_counts.items():
            state_pct = (count / len(self.market_states)) * 100
            print(f"  State {int(state_id)}: {count} days ({state_pct:.1f}% of analysis period)")
        
        print("-" * 80)
        
        # Print detailed descriptions for each unique state found
        print(f"\n📋 Detailed Market State Descriptions for Analysis Period:")
        print("-" * 80)
        
        # Convert predicted_states to Series with same index as analysis_data for proper alignment
        predicted_states_series = pd.Series(predicted_states, index=analysis_data.index)
        
        unique_states = sorted(state_counts.index)
        for state_id in unique_states:
            # Get state description using analysis data
            # Use the Series for boolean indexing to ensure proper alignment
            description = self.state_classifier.get_state_description(
                int(state_id), analysis_data, predicted_states_series
            )
            state_count = state_counts[state_id]
            state_pct = (state_count / len(self.market_states)) * 100
            print(f"  {description}")
            print(f"    Occurrences: {state_count} days ({state_pct:.1f}% of analysis period)")
            print()
        
        print("-" * 80)
        
        return self.market_states
    
    def calculate_log_returns(self, data: pd.DataFrame, periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculate log returns for specified periods.
        
        Args:
            data: DataFrame with Close prices
            periods: List of periods for log return calculation
            
        Returns:
            DataFrame with log returns for each period
        """
        result = data.copy()
        
        for period in periods:
            # Log return = ln(Price_t / Price_{t-period})
            result[f'log_return_{period}d'] = np.log(result['Close'] / result['Close'].shift(period))
        
        return result
    
    def filter_cooldown(self, big_moves: List[Tuple], cooldown_days: int = 12) -> List[Tuple]:
        """
        Filter big moves to ensure independent signals with cooldown period.
        
        After each signal, ignore all subsequent signals for cooldown_days.
        This ensures no overlapping forward return windows and independent samples.
        
        Args:
            big_moves: List of tuples (date, period, log_return, std_devs)
            cooldown_days: Minimum days between signals (default: 12)
            
        Returns:
            Filtered list of big moves with cooldown applied
        """
        if not big_moves:
            return []
        
        # Sort by date
        sorted_moves = sorted(big_moves, key=lambda x: x[0])
        filtered = [sorted_moves[0]]
        
        for move in sorted_moves[1:]:
            days_since_last = (move[0] - filtered[-1][0]).days
            if days_since_last >= cooldown_days:
                filtered.append(move)
        
        print(f"\nApplied {cooldown_days}-day cooldown filter:")
        print(f"   Before: {len(big_moves)} signals")
        print(f"   After:  {len(filtered)} independent signals")
        print(f"   Removed: {len(big_moves) - len(filtered)} overlapping signals")
        
        return filtered
    
    def identify_big_moves(self, vix_data: pd.DataFrame, cooldown_days: int = 12) -> List[Tuple[pd.Timestamp, int, float, float]]:
        """
        Identify big moves in VIX (returns > threshold).
        
        A big move is defined as a return exceeding the threshold (default: 50%) over 1-3 days.
        
        Applies cooldown period to ensure independent signals - after each signal,
        subsequent signals are ignored for cooldown_days.
        
        Args:
            vix_data: DataFrame with VIX data and log returns
            cooldown_days: Minimum days between signals (default: 12)
            
        Returns:
            Tuple of (big_moves, raw_signals) where each is a list of tuples:
            (date, period, log_return, return_pct)
        """
        big_moves = []
        
        print(f"\n🔍 Identifying big moves using {self.return_threshold*100:.0f}% return threshold (SPY down only)")
        print(f"   Log return threshold: {self.return_threshold_log:.4f}")
        
        # Get SPY 1-day returns for filtering
        spy_1d_returns = self.spy_data['log_return_1d'] if 'log_return_1d' in self.spy_data.columns else None
        
        for period in [1, 2, 3]:
            col = f'log_return_{period}d'
            if col not in vix_data.columns:
                continue
            
            log_returns = vix_data[col]
            
            # Identify dates where log return exceeds threshold
            big_move_mask = log_returns > self.return_threshold_log
            big_move_dates = vix_data[big_move_mask].index
            
            # Calculate stats for reporting
            max_return = log_returns.max()
            mean_return = log_returns.mean()
            
            print(f"\nVIX {period}-day returns:")
            print(f"   Mean log return: {mean_return:.6f} ({(np.exp(mean_return)-1)*100:.2f}%)")
            print(f"   Max log return: {max_return:.6f} ({(np.exp(max_return)-1)*100:.2f}%)")
            print(f"   Threshold: >{self.return_threshold*100:.0f}% (log return > {self.return_threshold_log:.4f})")
            print(f"   Found {len(big_move_dates)} raw VIX signals", end='')
            
            # Filter by SPY direction (only keep if SPY went down)
            filtered_count = 0
            for date in big_move_dates:
                # Check if SPY went down on this date
                if spy_1d_returns is not None and date in spy_1d_returns.index:
                    spy_return = spy_1d_returns.loc[date]
                    if spy_return >= 0:  # SPY went up or flat, skip this signal
                        filtered_count += 1
                        continue
                
                log_return = vix_data.loc[date, col]
                # Convert log return back to percentage for reporting
                pct_return = (np.exp(log_return) - 1) * 100
                big_moves.append((date, period, log_return, pct_return))
            
            print(f" → {len(big_move_dates) - filtered_count} after SPY direction filter (removed {filtered_count} where SPY up/flat)")
        
        print(f"\nTotal raw signals identified: {len(big_moves)} (VIX up + SPY down)")
        
        # Store raw signals before filtering
        raw_signals = big_moves.copy()
        
        # Apply cooldown filter to get independent signals
        big_moves = self.filter_cooldown(big_moves, cooldown_days=cooldown_days)
        
        return big_moves, raw_signals
    
    def calculate_forward_spy_returns(self, date: pd.Timestamp, forward_days: List[int] = [3, 6, 12]) -> Dict[int, float]:
        """
        Calculate SPY log returns for specified forward periods from a given date.
        Applies lag if configured (e.g., lag_days=1 means start calculation 1 day after the VIX move).
        
        Args:
            date: Starting date (VIX big move date)
            forward_days: List of forward periods to calculate returns
            
        Returns:
            Dictionary mapping forward days to log returns (None if data not available)
        """
        forward_returns = {}
        
        try:
            start_idx = self.spy_data.index.get_loc(date)
        except (KeyError, IndexError):
            # Date not in data
            return {days: None for days in forward_days}
        
        # Apply lag - start calculation lag_days after the VIX move
        start_idx += self.lag_days
        
        if start_idx >= len(self.spy_data):
            # Not enough data after lag
            return {days: None for days in forward_days}
        
        start_date = self.spy_data.index[start_idx]
        start_price = self.spy_data.loc[start_date, 'Close']
        
        for days in forward_days:
            target_idx = start_idx + days
            
            if target_idx < len(self.spy_data):
                target_date = self.spy_data.index[target_idx]
                end_price = self.spy_data.loc[target_date, 'Close']
                log_return = np.log(end_price / start_price)
                forward_returns[days] = log_return
            else:
                forward_returns[days] = None
        
        return forward_returns
    
    def analyze_big_moves(self) -> List[BigMoveEvent]:
        """
        Analyze all big moves and calculate corresponding SPY returns.
        
        Returns:
            List of BigMoveEvent objects
        """
        # Calculate log returns for VIX
        self.vix_data = self.calculate_log_returns(self.vix_data, periods=[1, 2, 3])
        
        # Identify big moves with cooldown filter
        big_moves, raw_signals = self.identify_big_moves(self.vix_data, cooldown_days=self.cooldown_days)
        
        # Store raw signals for visualization
        self.raw_signals = raw_signals
        
        print(f"\nCalculating SPY returns for {len(big_moves)} independent big move events...")
        
        # Calculate SPY forward returns for each big move
        events = []
        for date, period, vix_log_return, std_devs in big_moves:
            forward_returns = self.calculate_forward_spy_returns(date, forward_days=[3, 6, 12])
            
            # Only create event if we have all forward returns
            if all(r is not None for r in forward_returns.values()):
                event = BigMoveEvent(
                    date=date,
                    vix_log_return=vix_log_return,
                    days_in_move=period,
                    vix_std_devs=std_devs,
                    spy_return_3d=forward_returns[3],
                    spy_return_6d=forward_returns[6],
                    spy_return_12d=forward_returns[12]
                )
                events.append(event)
        
        self.big_move_events = events
        print(f"Analyzed {len(events)} complete independent big move events")
        
        return events
    
    def filter_events_by_market_state(self, events: List[BigMoveEvent], target_states) -> List[BigMoveEvent]:
        """
        Filter big move events to only include those occurring when market state matches target_state(s).
        
        Args:
            events: List of BigMoveEvent objects
            target_states: Market state ID(s) to filter for - can be int, list of ints, or None
            
        Returns:
            Filtered list of BigMoveEvent objects
        """
        if self.market_states is None or len(self.market_states) == 0:
            print(f"⚠️  Warning: Market states not available. Returning all events.")
            return events
        
        # Convert single int to list for consistent handling
        if isinstance(target_states, int):
            target_states = [target_states]
        
        filtered_events = []
        for event in events:
            # Check if market state on the event date is in target states
            if event.date in self.market_states.index:
                state_on_date = self.market_states.loc[event.date]
                if pd.notna(state_on_date) and int(state_on_date) in target_states:
                    filtered_events.append(event)
        
        if len(target_states) == 1:
            print(f"\nFiltered events by market state {target_states[0]}:")
            print(f"   Before: {len(events)} events")
            print(f"   After:  {len(filtered_events)} events (state {target_states[0]})")
            print(f"   Removed: {len(events) - len(filtered_events)} events")
        else:
            print(f"\nFiltered events by market states {target_states}:")
            print(f"   Before: {len(events)} events")
            print(f"   After:  {len(filtered_events)} events (states {target_states})")
            print(f"   Removed: {len(events) - len(filtered_events)} events")
        
        return filtered_events
    
    def round_to_nearest(self, value: float, increment: float = 0.002) -> float:
        """
        Round a value to the nearest increment (default 0.2% = 0.002).
        
        Args:
            value: Value to round
            increment: Rounding increment
            
        Returns:
            Rounded value
        """
        return round(value / increment) * increment
    
    def calculate_random_day_returns(self, n_samples: int, forward_days: List[int] = [3, 6, 12]) -> Dict[int, np.ndarray]:
        """
        Calculate forward returns for randomly selected days (excluding VIX spike days).
        
        Args:
            n_samples: Number of random samples to draw
            forward_days: List of forward periods to calculate returns
            
        Returns:
            Dictionary mapping forward days to arrays of returns
        """
        print(f"\nCalculating forward returns for {n_samples} random days (control group)...")
        
        # Get dates of VIX spike events to exclude them
        spike_dates = set([e.date for e in self.big_move_events])
        
        # Get all available dates that have sufficient forward data
        max_forward = max(forward_days)
        available_dates = self.spy_data.index[:-max_forward]  # Exclude last N days
        
        # Exclude VIX spike dates
        random_dates = [d for d in available_dates if d not in spike_dates]
        
        # Randomly sample dates
        np.random.seed(42)  # For reproducibility
        sampled_dates = np.random.choice(random_dates, size=min(n_samples, len(random_dates)), replace=False)
        
        # Calculate forward returns for each sampled date
        random_returns = {days: [] for days in forward_days}
        
        for date in sampled_dates:
            forward_returns = self.calculate_forward_spy_returns(date, forward_days=forward_days)
            
            # Only include if all forward returns are available
            if all(r is not None for r in forward_returns.values()):
                for days in forward_days:
                    random_returns[days].append(forward_returns[days])
        
        # Convert to numpy arrays
        random_returns = {days: np.array(returns) for days, returns in random_returns.items()}
        
        print(f"Calculated {len(random_returns[forward_days[0]])} random day returns")
        return random_returns
    
    def bootstrap_analysis(self, events: List[BigMoveEvent], n_bootstrap: int = 10000) -> Dict[int, Dict]:
        """
        Perform bootstrap analysis comparing VIX spike returns vs random day returns.
        
        Args:
            events: List of BigMoveEvent objects
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with bootstrap results for each forward period
        """
        print(f"\nPerforming two-sample bootstrap analysis ({n_bootstrap} iterations)...")
        print("Comparing: VIX spike days vs. Random days")
        
        # Calculate random day returns for comparison
        random_day_returns = self.calculate_random_day_returns(n_samples=len(events) * 2)
        
        results = {}
        
        for days in [3, 6, 12]:
            # Extract returns after VIX spikes
            if days == 3:
                spike_returns = np.array([e.spy_return_3d for e in events])
            elif days == 6:
                spike_returns = np.array([e.spy_return_6d for e in events])
            else:  # 12
                spike_returns = np.array([e.spy_return_12d for e in events])
            
            # Get random day returns
            random_returns = random_day_returns[days]
            
            n_spike = len(spike_returns)
            n_random = len(random_returns)
            
            # Bootstrap statistics for both distributions
            bootstrap_spike_means = []
            bootstrap_spike_medians = []
            bootstrap_random_means = []
            bootstrap_random_medians = []
            bootstrap_diff_means = []
            bootstrap_diff_medians = []
            
            np.random.seed(42)  # For reproducibility
            for _ in range(n_bootstrap):
                # Resample VIX spike returns
                spike_sample = np.random.choice(spike_returns, size=n_spike, replace=True)
                spike_mean = np.mean(spike_sample)
                spike_median = np.median(spike_sample)
                bootstrap_spike_means.append(spike_mean)
                bootstrap_spike_medians.append(spike_median)
                
                # Resample random day returns
                random_sample = np.random.choice(random_returns, size=n_spike, replace=True)
                random_mean = np.mean(random_sample)
                random_median = np.median(random_sample)
                bootstrap_random_means.append(random_mean)
                bootstrap_random_medians.append(random_median)
                
                # Calculate differences
                bootstrap_diff_means.append(spike_mean - random_mean)
                bootstrap_diff_medians.append(spike_median - random_median)
            
            # Convert to numpy arrays
            bootstrap_spike_means = np.array(bootstrap_spike_means)
            bootstrap_spike_medians = np.array(bootstrap_spike_medians)
            bootstrap_random_means = np.array(bootstrap_random_means)
            bootstrap_random_medians = np.array(bootstrap_random_medians)
            bootstrap_diff_means = np.array(bootstrap_diff_means)
            bootstrap_diff_medians = np.array(bootstrap_diff_medians)
            
            # Calculate confidence intervals for spike returns
            spike_mean_ci = (np.percentile(bootstrap_spike_means, 2.5), 
                            np.percentile(bootstrap_spike_means, 97.5))
            spike_median_ci = (np.percentile(bootstrap_spike_medians, 2.5),
                              np.percentile(bootstrap_spike_medians, 97.5))
            
            # Calculate confidence intervals for random returns
            random_mean_ci = (np.percentile(bootstrap_random_means, 2.5),
                             np.percentile(bootstrap_random_means, 97.5))
            random_median_ci = (np.percentile(bootstrap_random_medians, 2.5),
                               np.percentile(bootstrap_random_medians, 97.5))
            
            # Calculate confidence intervals for differences
            diff_mean_ci = (np.percentile(bootstrap_diff_means, 2.5),
                           np.percentile(bootstrap_diff_means, 97.5))
            diff_median_ci = (np.percentile(bootstrap_diff_medians, 2.5),
                             np.percentile(bootstrap_diff_medians, 97.5))
            
            # Calculate p-values (one-tailed: spike > random)
            p_value_mean_greater = np.sum(bootstrap_diff_means <= 0) / n_bootstrap
            p_value_median_greater = np.sum(bootstrap_diff_medians <= 0) / n_bootstrap
            
            results[days] = {
                # Spike returns
                'spike_means': bootstrap_spike_means,
                'spike_medians': bootstrap_spike_medians,
                'spike_mean_ci': spike_mean_ci,
                'spike_median_ci': spike_median_ci,
                'spike_original_mean': np.mean(spike_returns),
                'spike_original_median': np.median(spike_returns),
                
                # Random returns
                'random_means': bootstrap_random_means,
                'random_medians': bootstrap_random_medians,
                'random_mean_ci': random_mean_ci,
                'random_median_ci': random_median_ci,
                'random_original_mean': np.mean(random_returns),
                'random_original_median': np.median(random_returns),
                
                # Differences
                'diff_means': bootstrap_diff_means,
                'diff_medians': bootstrap_diff_medians,
                'diff_mean_ci': diff_mean_ci,
                'diff_median_ci': diff_median_ci,
                'diff_original_mean': np.mean(spike_returns) - np.mean(random_returns),
                'diff_original_median': np.median(spike_returns) - np.median(random_returns),
                
                # P-values
                'p_value_mean_greater': p_value_mean_greater,
                'p_value_median_greater': p_value_median_greater,
                
                # Sample sizes
                'n_spike': n_spike,
                'n_random': n_random
            }
        
        print("Bootstrap analysis complete!")
        return results
    
    def print_bootstrap_results(self, bootstrap_results: Dict[int, Dict]):
        """
        Print bootstrap analysis results comparing VIX spikes vs random days.
        
        Args:
            bootstrap_results: Dictionary with bootstrap results
        """
        print("\n" + "="*80)
        print("TWO-SAMPLE BOOTSTRAP ANALYSIS")
        print("="*80)
        print(f"Bootstrap Iterations: {len(bootstrap_results[3]['spike_means'])}")
        print(f"VIX Spike Days: {bootstrap_results[3]['n_spike']}")
        print(f"Random Days (Control): {bootstrap_results[3]['n_random']}")
        print("\nNull Hypothesis: Returns after VIX spikes = Returns on random days")
        
        for days in [3, 6, 12]:
            result = bootstrap_results[days]
            print("\n" + "="*80)
            print(f"{days}-DAY FORWARD RETURNS")
            print("="*80)
            
            # VIX Spike Returns
            print("\nVIX SPIKE DAYS:")
            print(f"  Mean Return:   {result['spike_original_mean']*100:.2f}%")
            print(f"  95% CI:        [{result['spike_mean_ci'][0]*100:.2f}%, {result['spike_mean_ci'][1]*100:.2f}%]")
            print(f"  Median Return: {result['spike_original_median']*100:.2f}%")
            print(f"  95% CI:        [{result['spike_median_ci'][0]*100:.2f}%, {result['spike_median_ci'][1]*100:.2f}%]")
            
            # Random Day Returns
            print("\nRANDOM DAYS (CONTROL):")
            print(f"  Mean Return:   {result['random_original_mean']*100:.2f}%")
            print(f"  95% CI:        [{result['random_mean_ci'][0]*100:.2f}%, {result['random_mean_ci'][1]*100:.2f}%]")
            print(f"  Median Return: {result['random_original_median']*100:.2f}%")
            print(f"  95% CI:        [{result['random_median_ci'][0]*100:.2f}%, {result['random_median_ci'][1]*100:.2f}%]")
            
            # Difference (Spike - Random)
            print("\nDIFFERENCE (VIX Spike - Random):")
            print(f"  Mean Difference:   {result['diff_original_mean']*100:.2f}%")
            print(f"  95% CI:            [{result['diff_mean_ci'][0]*100:.2f}%, {result['diff_mean_ci'][1]*100:.2f}%]")
            print(f"  P-value:           {result['p_value_mean_greater']:.4f}")
            
            if result['p_value_mean_greater'] < 0.05:
                if result['diff_original_mean'] > 0:
                    print(f"  SIGNIFICANT: VIX spikes have BETTER returns (p < 0.05)")
                else:
                    print(f"  SIGNIFICANT: VIX spikes have WORSE returns (p < 0.05)")
            else:
                print(f"  NOT SIGNIFICANT: No difference from random days (p >= 0.05)")
            
            print(f"\n  Median Difference: {result['diff_original_median']*100:.2f}%")
            print(f"  95% CI:            [{result['diff_median_ci'][0]*100:.2f}%, {result['diff_median_ci'][1]*100:.2f}%]")
            print(f"  P-value:           {result['p_value_median_greater']:.4f}")
            
            if result['p_value_median_greater'] < 0.05:
                if result['diff_original_median'] > 0:
                    print(f"  SIGNIFICANT: VIX spikes have BETTER median returns (p < 0.05)")
                else:
                    print(f"  SIGNIFICANT: VIX spikes have WORSE median returns (p < 0.05)")
            else:
                print(f"  NOT SIGNIFICANT: No difference from random days (p >= 0.05)")
        
        print("\n" + "="*80)
    
    def plot_bootstrap_distributions(self, bootstrap_results: Dict[int, Dict], save_path: str = None):
        """
        Plot bootstrap distributions comparing VIX spikes vs random days.
        
        Args:
            bootstrap_results: Dictionary with bootstrap results
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        colors_spike = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        colors_random = ['#FFA07A', '#87CEEB', '#98D8C8']
        
        for idx, days in enumerate([3, 6, 12]):
            result = bootstrap_results[days]
            
            # Row 0: Compare mean distributions (Spike vs Random)
            ax_mean = axes[0, idx]
            ax_mean.hist(result['spike_means'] * 100, bins=40, 
                        color=colors_spike[idx], alpha=0.6, edgecolor='black', 
                        linewidth=0.5, label='VIX Spikes')
            ax_mean.hist(result['random_means'] * 100, bins=40,
                        color=colors_random[idx], alpha=0.6, edgecolor='black',
                        linewidth=0.5, label='Random Days')
            ax_mean.axvline(result['spike_original_mean'] * 100, color='red', 
                           linestyle='--', linewidth=2, label='Spike Mean')
            ax_mean.axvline(result['random_original_mean'] * 100, color='blue',
                           linestyle='--', linewidth=2, label='Random Mean')
            ax_mean.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            ax_mean.set_xlabel('Mean Return (%)', fontsize=10, fontweight='bold')
            ax_mean.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax_mean.set_title(f'{days}-Day: Mean Distributions\n' +
                             f'Spike: {result["spike_original_mean"]*100:.2f}% | ' +
                             f'Random: {result["random_original_mean"]*100:.2f}%',
                             fontsize=10, fontweight='bold')
            ax_mean.legend(fontsize=7, loc='upper left')
            ax_mean.grid(axis='y', alpha=0.3)
            
            # Row 1: Compare median distributions (Spike vs Random)
            ax_median = axes[1, idx]
            ax_median.hist(result['spike_medians'] * 100, bins=40,
                          color=colors_spike[idx], alpha=0.6, edgecolor='black',
                          linewidth=0.5, label='VIX Spikes')
            ax_median.hist(result['random_medians'] * 100, bins=40,
                          color=colors_random[idx], alpha=0.6, edgecolor='black',
                          linewidth=0.5, label='Random Days')
            ax_median.axvline(result['spike_original_median'] * 100, color='red',
                             linestyle='--', linewidth=2, label='Spike Median')
            ax_median.axvline(result['random_original_median'] * 100, color='blue',
                             linestyle='--', linewidth=2, label='Random Median')
            ax_median.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            ax_median.set_xlabel('Median Return (%)', fontsize=10, fontweight='bold')
            ax_median.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax_median.set_title(f'{days}-Day: Median Distributions\n' +
                               f'Spike: {result["spike_original_median"]*100:.2f}% | ' +
                               f'Random: {result["random_original_median"]*100:.2f}%',
                               fontsize=10, fontweight='bold')
            ax_median.legend(fontsize=7, loc='upper left')
            ax_median.grid(axis='y', alpha=0.3)
            
            # Row 2: Difference distribution (Spike - Random)
            ax_diff = axes[2, idx]
            ax_diff.hist(result['diff_means'] * 100, bins=50,
                        color=colors_spike[idx], alpha=0.7, edgecolor='black',
                        linewidth=0.5)
            ax_diff.axvline(result['diff_original_mean'] * 100, color='red',
                           linestyle='--', linewidth=2, label='Observed Diff')
            ax_diff.axvline(result['diff_mean_ci'][0] * 100, color='blue',
                           linestyle=':', linewidth=2, label='95% CI')
            ax_diff.axvline(result['diff_mean_ci'][1] * 100, color='blue',
                           linestyle=':', linewidth=2)
            ax_diff.axvline(0, color='black', linestyle='-', linewidth=2)
            ax_diff.set_xlabel('Mean Difference (%)', fontsize=10, fontweight='bold')
            ax_diff.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax_diff.set_title(f'{days}-Day: Difference (Spike - Random)\n' +
                             f'Diff: {result["diff_original_mean"]*100:.2f}% | ' +
                             f'p-value: {result["p_value_mean_greater"]:.4f}',
                             fontsize=10, fontweight='bold')
            ax_diff.legend(fontsize=7)
            ax_diff.grid(axis='y', alpha=0.3)
            
            # Add significance annotation
            if result['p_value_mean_greater'] < 0.05:
                significance = "SIGNIFICANT" if result['diff_original_mean'] > 0 else "SIG. WORSE"
                ax_diff.text(0.02, 0.98, significance, transform=ax_diff.transAxes,
                            fontsize=9, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen' if result['diff_original_mean'] > 0 else 'lightcoral', alpha=0.8))
            else:
                ax_diff.text(0.02, 0.98, "NOT SIG.", transform=ax_diff.transAxes,
                            fontsize=9, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        title_text = 'Two-Sample Bootstrap Analysis: VIX Spikes vs Random Days'
        if self.market_state_filter is not None:
            if isinstance(self.market_state_filter, list) and len(self.market_state_filter) > 1:
                title_text += f' (Market States {self.market_state_filter} only)'
            elif isinstance(self.market_state_filter, list):
                title_text += f' (Market State {self.market_state_filter[0]} only)'
            else:
                title_text += f' (Market State {self.market_state_filter} only)'
        if self.lag_days > 0:
            title_text += f' ({self.lag_days}-day lag)'
        title_text += f'\n10,000 Bootstrap Iterations | n_spike={bootstrap_results[3]["n_spike"]}, ' + \
                     f'n_random={bootstrap_results[3]["n_random"]}'
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nBootstrap plot saved to: {save_path}")
        else:
            plt.show()
    
    def calculate_return_statistics(self, events: List[BigMoveEvent]) -> Dict[int, ReturnStats]:
        """
        Calculate statistics for SPY returns at different forward periods.
        
        Args:
            events: List of BigMoveEvent objects
            
        Returns:
            Dictionary mapping forward days to ReturnStats
        """
        stats = {}
        
        for days in [3, 6, 12]:
            # Extract returns for this period
            if days == 3:
                returns = [e.spy_return_3d for e in events]
            elif days == 6:
                returns = [e.spy_return_6d for e in events]
            else:  # 12
                returns = [e.spy_return_12d for e in events]
            
            # Calculate statistics
            avg_return = np.mean(returns)
            median_return = np.median(returns)
            
            # Round returns to nearest 0.2% and count frequencies
            rounded_returns = [self.round_to_nearest(r, 0.002) for r in returns]
            frequency_dist = Counter(rounded_returns)
            
            # Sort frequency distribution
            frequency_dist = dict(sorted(frequency_dist.items()))
            
            stats[days] = ReturnStats(
                days_forward=days,
                average_return=avg_return,
                median_return=median_return,
                total_events=len(returns),
                frequency_distribution=frequency_dist
            )
        
        return stats
    
    def plot_vix_timeline(self, raw_signals: List[Tuple] = None, save_path: str = None):
        """
        Plot VIX and SPY over time with signal markers and market states.
        
        Args:
            raw_signals: Optional list of raw signals before cooldown filter
            save_path: Optional path to save the plot (if None, will display)
        """
        # Create figure with 3 subplots if market states available, otherwise 2
        if self.market_states is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        
        # Add market state background colors if available
        if self.market_states is not None and len(self.market_states) > 0:
            # Define colors for different market states
            state_colors = {
                0: '#E8F5E9',  # Light green - Low volatility uptrend
                1: '#C8E6C9',  # Medium green - Momentum uptrend
                2: '#FFF9C4',  # Light yellow - Consolidation
                3: '#FFCDD2',  # Light red - High volatility downtrend
                4: '#F8BBD0',  # Light pink - High volatility rally
            }
            
            # Align market states with vix_data dates
            aligned_states = self.market_states.reindex(self.vix_data.index, method='ffill')
            
            # Create background regions for each state
            prev_state = None
            prev_date = None
            for date in self.vix_data.index:
                if date in aligned_states.index and pd.notna(aligned_states.loc[date]):
                    current_state = int(aligned_states.loc[date])
                    if current_state != prev_state:
                        if prev_date is not None and prev_state is not None:
                            # Fill region from previous date to current date
                            ax1.axvspan(prev_date, date, 
                                      color=state_colors.get(int(prev_state), '#FFFFFF'),
                                      alpha=0.3, zorder=0)
                        prev_state = current_state
                        prev_date = date
                    elif prev_date is None:
                        # First valid state
                        prev_state = current_state
                        prev_date = date
            
            # Fill final region
            if prev_date is not None and prev_state is not None:
                ax1.axvspan(prev_date, self.vix_data.index[-1],
                          color=state_colors.get(int(prev_state), '#FFFFFF'),
                          alpha=0.3, zorder=0)
        
        # Plot VIX price on primary y-axis
        color_vix = '#2E86AB'
        ax1.plot(self.vix_data.index, self.vix_data['Close'], 
                color=color_vix, linewidth=2, alpha=0.8, label='VIX', zorder=3)
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('VIX Level', fontsize=12, fontweight='bold', color=color_vix)
        ax1.tick_params(axis='y', labelcolor=color_vix)
        
        # Mark filtered (independent) signal dates on VIX line
        signal_dates = [event.date for event in self.big_move_events]
        signal_vix_values = [self.vix_data.loc[date, 'Close'] for date in signal_dates]
        
        ax1.scatter(signal_dates, signal_vix_values, 
                   color='#A23B72', s=150, marker='o', 
                   edgecolors='white', linewidths=2, 
                   zorder=5, label=f'Independent Signals (n={len(signal_dates)})')
        
        # If raw signals provided, mark them too
        if raw_signals:
            raw_dates = [sig[0] for sig in raw_signals]
            raw_vix_values = [self.vix_data.loc[date, 'Close'] for date in raw_dates]
            
            ax1.scatter(raw_dates, raw_vix_values,
                       color='#F18F01', s=80, marker='x', alpha=0.6,
                       linewidths=2, zorder=4, 
                       label=f'Raw Signals (n={len(raw_dates)})')
        
        # Add horizontal line at mean VIX
        mean_vix = self.vix_data['Close'].mean()
        ax1.axhline(y=mean_vix, color=color_vix, linestyle='--', 
                   linewidth=1, alpha=0.3)
        
        # Create secondary y-axis for SPY
        ax1_spy = ax1.twinx()
        color_spy = '#27AE60'
        ax1_spy.plot(self.spy_data.index, self.spy_data['Close'],
                    color=color_spy, linewidth=2, alpha=0.7, label='SPY', linestyle='-')
        ax1_spy.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold', color=color_spy)
        ax1_spy.tick_params(axis='y', labelcolor=color_spy)
        
        # Add SPY markers at signal dates
        signal_spy_values = [self.spy_data.loc[date, 'Close'] for date in signal_dates]
        ax1_spy.scatter(signal_dates, signal_spy_values,
                       color='#A23B72', s=100, marker='v',
                       edgecolors='white', linewidths=1.5,
                       zorder=5, alpha=0.7)
        
        # Set title and limits
        ax1.set_title('VIX & SPY Timeline with Signal Markers\n' + 
                     f'Analysis Period: {self.vix_data.index[0].strftime("%Y-%m-%d")} to {self.vix_data.index[-1].strftime("%Y-%m-%d")}',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(self.vix_data.index[0], self.vix_data.index[-1])
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_spy.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
        
        # Middle panel: Signal distribution over time
        ax2.hist([event.date for event in self.big_move_events], 
                bins=30, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Signal Count', fontsize=12, fontweight='bold')
        ax2.set_title(f'Independent Signal Distribution (Cooldown: {self.cooldown_days} days)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xlim(self.vix_data.index[0], self.vix_data.index[-1])
        
        # Add statistics box
        stats_text = f'Total Independent Signals: {len(self.big_move_events)}\n'
        stats_text += f'Cooldown Period: {self.cooldown_days} days\n'
        if raw_signals:
            stats_text += f'Filtered Out: {len(raw_signals) - len(self.big_move_events)} signals'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bottom panel: Market states over time (if available)
        if self.market_states is not None and len(self.market_states) > 0:
            # Align market states with vix_data dates
            aligned_states = self.market_states.reindex(self.vix_data.index, method='ffill')
            
            # Filter out NaN values for plotting
            valid_states = aligned_states.dropna()
            
            if len(valid_states) > 0:
                # Plot market states as a line
                ax3.plot(valid_states.index, valid_states.values, 
                        color='#7B1FA2', linewidth=1.5, alpha=0.8, marker='o', markersize=2)
                ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Market State', fontsize=12, fontweight='bold')
                ax3.set_title('HMM Market State Classification', fontsize=12, fontweight='bold')
                
                # Set y-ticks based on unique states
                unique_states = sorted(valid_states.unique())
                ax3.set_yticks(unique_states)
                ax3.set_ylim(min(unique_states) - 0.5, max(unique_states) + 0.5)
                
                ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
                ax3.set_xlim(self.vix_data.index[0], self.vix_data.index[-1])
                
                # Add state distribution info
                state_counts = valid_states.value_counts().sort_index()
                state_info = f"States: {', '.join([f'{int(s)}: {c}' for s, c in state_counts.items()])}"
                ax3.text(0.02, 0.98, state_info, transform=ax3.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVIX timeline plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_results(self, stats: Dict[int, ReturnStats], save_path: str = None):
        """
        Create comprehensive visualizations of the VIX-SPY analysis results.
        
        Args:
            stats: Dictionary of ReturnStats for each forward period
            save_path: Optional path to save the plot (if None, will display)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Plot 1: Average Returns Bar Chart
        ax_avg = fig.add_subplot(gs[0, :])
        days_list = [3, 6, 12]
        avg_returns = [stats[d].average_return * 100 for d in days_list]
        median_returns = [stats[d].median_return * 100 for d in days_list]
        
        x = np.arange(len(days_list))
        width = 0.35
        
        bars1 = ax_avg.bar(x - width/2, avg_returns, width, label='Average', color=colors, alpha=0.8)
        bars2 = ax_avg.bar(x + width/2, median_returns, width, label='Median', 
                          color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
        
        ax_avg.set_xlabel('Days After Big VIX Move', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('SPY Return (%)', fontsize=12, fontweight='bold')
        title_text = f'Average SPY Returns Following Big VIX Moves\n(VIX return > {self.return_threshold*100:.0f}%'
        if self.market_state_filter is not None:
            if isinstance(self.market_state_filter, list) and len(self.market_state_filter) > 1:
                title_text += f', Market States {self.market_state_filter} only'
            elif isinstance(self.market_state_filter, list):
                title_text += f', Market State {self.market_state_filter[0]} only'
            else:
                title_text += f', Market State {self.market_state_filter} only'
        if self.lag_days > 0:
            title_text += f', {self.lag_days}-day lag'
        title_text += ')'
        ax_avg.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        ax_avg.set_xticks(x)
        ax_avg.set_xticklabels([f'{d} Days' for d in days_list])
        ax_avg.legend(fontsize=10)
        ax_avg.grid(axis='y', alpha=0.3, linestyle='--')
        ax_avg.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_avg.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}%',
                          ha='center', va='bottom' if height >= 0 else 'top',
                          fontsize=9, fontweight='bold')
        
        # Plots 2-4: Frequency Distribution Histograms
        for idx, days in enumerate([3, 6, 12]):
            ax = fig.add_subplot(gs[1, idx])
            stat = stats[days]
            
            # Convert to percentage and prepare data
            returns_pct = np.array(list(stat.frequency_distribution.keys())) * 100
            frequencies = np.array(list(stat.frequency_distribution.values()))
            
            # Create histogram
            bars = ax.bar(returns_pct, frequencies, width=0.18, color=colors[idx], 
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add vertical line for mean and median
            ax.axvline(stat.average_return * 100, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {stat.average_return*100:.2f}%')
            ax.axvline(stat.median_return * 100, color='blue', linestyle='--', 
                      linewidth=2, label=f'Median: {stat.median_return*100:.2f}%')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('SPY Return (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency (Count)', fontsize=10, fontweight='bold')
            if self.lag_days > 0:
                ax.set_title(f'{days}-Day Returns (Day {self.lag_days} to Day {self.lag_days + days})\n(n={stat.total_events} events)', 
                            fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{days}-Day Forward Returns\n(n={stat.total_events} events)', 
                            fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Overall title
        title_text = 'VIX Big Move → SPY Return Analysis'
        if self.market_state_filter is not None:
            if isinstance(self.market_state_filter, list) and len(self.market_state_filter) > 1:
                title_text += f' (Market States {self.market_state_filter} only)'
            elif isinstance(self.market_state_filter, list):
                title_text += f' (Market State {self.market_state_filter[0]} only)'
            else:
                title_text += f' (Market State {self.market_state_filter} only)'
        if self.lag_days > 0:
            title_text += f' ({self.lag_days}-day lag)'
        title_text += f'\nAnalysis Period: {self.start_date} to {self.vix_data.index[-1].strftime("%Y-%m-%d")}'
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()
    
    def print_results(self, stats: Dict[int, ReturnStats]):
        """
        Print analysis results in a readable format.
        
        Args:
            stats: Dictionary of ReturnStats for each forward period
        """
        print("\n" + "="*80)
        print("VIX BIG MOVE -> SPY RETURN ANALYSIS")
        print("="*80)
        print(f"\nAnalysis Period: {self.start_date} to {self.vix_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Big Move Definition: VIX return > {self.return_threshold*100:.0f}% over 1-3 days")
        print(f"Cooldown Period: {self.cooldown_days} days (ensures independent signals)")
        if self.lag_days > 0:
            print(f"Lag Period: {self.lag_days} day(s) - SPY returns start {self.lag_days} day(s) after VIX move")
        else:
            print(f"Lag Period: 0 days - SPY returns start on the day of VIX move")
        # Note: market_state_filter info is printed earlier in the pipeline
        print(f"Total Independent Signals Analyzed: {len(self.big_move_events)}")
        
        for days in [3, 6, 12]:
            stat = stats[days]
            print("\n" + "-"*80)
            if self.lag_days > 0:
                print(f"SPY RETURNS OVER {days} DAYS, STARTING {self.lag_days} DAY(S) AFTER VIX MOVE")
                print(f"(Period: Day {self.lag_days} to Day {self.lag_days + days} after VIX move)")
            else:
                print(f"SPY RETURNS {days} DAYS AFTER BIG VIX MOVE")
            print("-"*80)
            print(f"Average Return: {stat.average_return:.4f} ({stat.average_return*100:.2f}%)")
            print(f"Median Return:  {stat.median_return:.4f} ({stat.median_return*100:.2f}%)")
            print(f"Total Events:   {stat.total_events}")
            
            print(f"\nFrequency Distribution (rounded to nearest 0.2%):")
            print(f"{'Return Bucket':<20} {'Count':<10} {'Frequency %':<15}")
            print("-"*50)
            
            for return_bucket, count in stat.frequency_distribution.items():
                frequency_pct = (count / stat.total_events) * 100
                return_pct = return_bucket * 100
                print(f"{return_pct:>6.1f}%{'':<13} {count:<10} {frequency_pct:>6.2f}%")
        
        print("\n" + "="*80)
    
    def run_analysis(self, create_plot: bool = True, save_plot_path: str = None, 
                     run_bootstrap: bool = True, n_bootstrap: int = 10000,
                     market_state_filter = None):
        """
        Run the complete VIX-SPY analysis pipeline.
        
        Args:
            create_plot: Whether to create visualizations (default: True)
            save_plot_path: Optional path to save the plot (if None, will display)
            run_bootstrap: Whether to run bootstrap significance tests (default: True)
            n_bootstrap: Number of bootstrap iterations (default: 10000)
            market_state_filter: Optional market state ID(s) to filter events - can be int or list of ints (default: None, no filter)
        """
        print("Starting VIX Big Move -> SPY Return Analysis\n")
        
        # Load data
        self.load_data()
        
        # Train HMM model on 10 years of data
        try:
            self.train_hmm_model(hmm_lookback_years=15)
            # Predict market states for the analysis period
            self.predict_market_states()
        except Exception as e:
            print(f"⚠️  Warning: Could not train/predict HMM market states: {str(e)}")
            print("   Continuing analysis without market state visualization...")
            self.state_classifier = None
            self.market_states = None
        
        # Analyze big moves and calculate SPY returns
        events = self.analyze_big_moves()
        
        if not events:
            print("No big move events found with complete data")
            return
        
        # Store filter value for use in plots
        self.market_state_filter = market_state_filter
        
        # Filter events by market state if filter is specified
        if market_state_filter is not None:
            if self.market_states is not None and len(self.market_states) > 0:
                if isinstance(market_state_filter, list) and len(market_state_filter) > 1:
                    print(f"\n🔍 Filtering events to only include market states {market_state_filter}...")
                elif isinstance(market_state_filter, list):
                    print(f"\n🔍 Filtering events to only include market state {market_state_filter[0]}...")
                else:
                    print(f"\n🔍 Filtering events to only include market state {market_state_filter}...")
                filtered_events = self.filter_events_by_market_state(events, target_states=market_state_filter)
                if len(filtered_events) == 0:
                    filter_str = str(market_state_filter) if isinstance(market_state_filter, list) else market_state_filter
                    print(f"⚠️  Warning: No events found with market state(s) {filter_str}. Using all events.")
                    filtered_events = events
                else:
                    events = filtered_events
                    self.big_move_events = events  # Update stored events for visualization
            else:
                print("⚠️  Warning: Market states not available. Cannot apply filter. Using all events.")
        
        # Calculate statistics
        stats = self.calculate_return_statistics(events)
        
        # Print results
        self.print_results(stats)
        
        # Run bootstrap analysis
        bootstrap_results = None
        if run_bootstrap:
            bootstrap_results = self.bootstrap_analysis(events, n_bootstrap=n_bootstrap)
            self.print_bootstrap_results(bootstrap_results)
            
            if create_plot:
                print("\nGenerating bootstrap distribution plots...")
                bootstrap_plot_path = save_plot_path.replace('.png', '_bootstrap.png') if save_plot_path else None
                self.plot_bootstrap_distributions(bootstrap_results, save_path=bootstrap_plot_path)
        
        # Create visualizations
        if create_plot:
            print("\nGenerating summary visualizations...")
            self.plot_results(stats, save_path=save_plot_path)
            
            # Create VIX timeline plot
            print("Generating VIX timeline plot...")
            timeline_path = save_plot_path.replace('.png', '_timeline.png') if save_plot_path else None
            self.plot_vix_timeline(raw_signals=self.raw_signals, save_path=timeline_path)
        
        print("\nAnalysis complete!")
        
        return stats, bootstrap_results


def main(create_plot: bool = True, save_plot: bool = True, run_bootstrap: bool = True,
         lookback_years: int = 5, cooldown_days: int = 12):
    """
    Main entry point for VIX-SPY analysis.
    
    Args:
        create_plot: Whether to create visualizations (default: True)
        save_plot: Whether to save the plot to file (default: True)
        run_bootstrap: Whether to run bootstrap significance tests (default: True)
        lookback_years: Years of historical data to analyze (default: 5)
        cooldown_days: Minimum days between independent signals (default: 12)
    """
    analyzer = VIXSPYAnalyzer(lookback_years=lookback_years, cooldown_days=cooldown_days)
    
    # Determine save path
    save_path = None
    if create_plot and save_plot:
        save_path = 'predictions/vix_spy_analysis.png'
        print(f"\nPlot will be saved to: {save_path}")
    
    stats, bootstrap_results = analyzer.run_analysis(
        create_plot=create_plot, 
        save_plot_path=save_path,
        run_bootstrap=run_bootstrap
    )
    return analyzer, stats, bootstrap_results


if __name__ == '__main__':
    analyzer, stats, bootstrap_results = main()
