#!/usr/bin/env python3
"""
Standalone script to run the MarketStateClassifier.

This script can be run directly to train and evaluate the HMM-based market state classifier.

Usage:
    python -m algo_trading_engine.ml_models.run_market_state_classifier
    python -m algo_trading_engine.ml_models.run_market_state_classifier --symbol SPY --start-date 2010-01-01 --max-states 5
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, package_dir)

from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier


def fetch_vix_data(start_date: str) -> pd.DataFrame:
    """Fetch VIX data using yfinance"""
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


def calculate_features(data: pd.DataFrame, vix_data: pd.DataFrame = None, window=20) -> pd.DataFrame:
    """Calculate technical features required for MarketStateClassifier"""
    # Basic returns
    data['Returns'] = data['Close'].pct_change()
    
    # Volatility - use VIX Close price if available, otherwise fallback to calculated volatility
    if vix_data is not None and not vix_data.empty:
        # Merge VIX data with main data
        vix_close = vix_data[['Close']].rename(columns={'Close': 'VIX'})
        data = data.join(vix_close, how='left')
        
        # Use VIX price as Volatility
        data['Volatility'] = data['VIX']
        
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


def fetch_market_data(symbol: str, start_date: str) -> pd.DataFrame:
    """Fetch market data using yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    print(f"📈 Fetching {symbol} data from {start_date}...")
    ticker = yf.Ticker(symbol)
    
    # Try period='max' first, then fallback to date range
    try:
        data = ticker.history(period='max')
    except Exception as e:
        print(f"⚠️  Failed to fetch with period='max', trying date range: {e}")
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp.now()
        data = ticker.history(start=start_date_ts, end=end_date_ts)
    
    if data.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # Normalize timezone - remove timezone info for easier date comparisons
    if data.index.tz is not None:
        # Convert timezone-aware index to naive by creating new index from values
        data.index = pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in data.index])
    
    # Filter to start_date if needed
    if start_date:
        start_date_ts = pd.Timestamp(start_date)
        data = data[data.index >= start_date_ts]
    
    print(f"✅ Retrieved {len(data)} samples from {data.index[0].date()} to {data.index[-1].date()}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate MarketStateClassifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (SPY, 2010-01-01, max 5 states)
  python -m algo_trading_engine.ml_models.run_market_state_classifier
  
  # Custom symbol and date range
  python -m algo_trading_engine.ml_models.run_market_state_classifier --symbol QQQ --start-date 2015-01-01
  
  # More states to evaluate
  python -m algo_trading_engine.ml_models.run_market_state_classifier --max-states 7
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2010-01-01',
        help='Start date for training data (default: 2010-01-01)'
    )
    
    parser.add_argument(
        '--max-states',
        type=int,
        default=5,
        help='Maximum number of states to evaluate (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Fetch data
    data = fetch_market_data(args.symbol, args.start_date)
    
    # Fetch VIX data for volatility feature
    try:
        vix_data = fetch_vix_data(args.start_date)
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch VIX data: {e}")
        print(f"   Falling back to calculated volatility")
        vix_data = None
    
    # Calculate required features
    print(f"\n📊 Calculating features...")
    data = calculate_features(data, vix_data=vix_data)
    
    if len(data) < 50:
        raise ValueError(f"Insufficient data after feature calculation: {len(data)} samples. Need at least 50.")
    
    print(f"✅ Prepared {len(data)} samples with features")
    
    # Verify required columns exist
    required_columns = ['Returns', 'Volatility', 'Price_to_SMA20', 'SMA20_to_SMA50', 'Volume_Ratio']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Analyze feature correlations
    print(f"\n📊 Feature Correlation Analysis:")
    feature_corr = data[required_columns].corr()
    print(f"\nCorrelation Matrix:")
    print(feature_corr.round(3))
    
    # Identify high correlations
    high_corr_pairs = []
    for i, col1 in enumerate(required_columns):
        for col2 in required_columns[i+1:]:
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
    
    # Train the classifier
    print(f"\n🎯 Training MarketStateClassifier...")
    classifier = MarketStateClassifier(max_states=args.max_states)
    optimal_states = classifier.train_hmm_model(data)
    
    # Predict states on the training data
    print(f"\n🔮 Predicting market states...")
    states = classifier.predict_states(data)
    
    # Display state distribution
    print(f"\n📊 State Distribution:")
    unique, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique, counts):
        percentage = (count / len(states)) * 100
        print(f"  State {state}: {count:4d} samples ({percentage:5.2f}%)")
    
    # Display state characteristics
    print(f"\n📈 State Characteristics:")
    state_chars = classifier._get_state_characteristics(data, states)
    for state in range(optimal_states):
        chars = state_chars[state]
        description = classifier.get_state_description(state, data, states)
        print(f"\n{description}")
        print(f"  Proportion: {chars['proportion']:.2%}")
        print(f"  Avg Return: {chars['avg_return']:.4%}")
        print(f"  Volatility: {chars['volatility']:.4f}")
    
    print(f"\n✅ Market state classification complete!")
    return classifier


if __name__ == '__main__':
    main()
