#!/usr/bin/env python3
"""
Script to examine the contents of cached data files
"""

import pickle
import pandas as pd
from pathlib import Path

def examine_cache_file(filepath):
    """Examine the contents of a cached pickle file"""
    print(f"🔍 Examining: {filepath}")
    print("=" * 60)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Successfully loaded data from {filepath}")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print(f"📈 DataFrame shape: {data.shape}")
            print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
            print(f"📋 Columns: {list(data.columns)}")
            
            print(f"\n📊 First 5 rows:")
            print(data.head())
            
            print(f"\n📊 Last 5 rows:")
            print(data.tail())
            
            print(f"\n📈 Summary statistics:")
            print(data.describe())
            
        else:
            print(f"📦 Data content: {data}")
            if hasattr(data, '__len__'):
                print(f"📏 Length: {len(data)}")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def main():
    # Path to the cached file
    cache_file = Path("data_cache/stocks/SPY/2021-06-01_lstm_data.pkl")
    
    if cache_file.exists():
        examine_cache_file(cache_file)
    else:
        print(f"❌ File not found: {cache_file}")
        
        # List available files in the directory
        cache_dir = cache_file.parent
        if cache_dir.exists():
            print(f"\n📁 Available files in {cache_dir}:")
            for file in cache_dir.glob("*.pkl"):
                print(f"   {file.name}")

if __name__ == "__main__":
    main() 