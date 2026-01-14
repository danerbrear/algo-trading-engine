#!/usr/bin/env python3
"""
Example: Paper trade custom strategy

This example shows how to use the algo-trading-engine package for paper trading
in a child repository. The package provides a clean public API for running
strategies against live market data.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import importlib.util

# Import from the public API
from algo_trading_engine import PaperTradingEngine, PaperTradingConfig

# Import custom strategy using absolute path
strategy_path = Path(__file__).parent.parent / "strategies" / "custom_strategy.py"
spec = importlib.util.spec_from_file_location("custom_strategy", strategy_path)
custom_strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_strategy_module)
MyCustomStrategy = custom_strategy_module.MyCustomStrategy

# Load environment variables from .env file
load_dotenv()

def main():
    """Run paper trading using the custom strategy."""
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    
    # Create configuration
    config = PaperTradingConfig(
        symbol="SPY",
        strategy_type=MyCustomStrategy(),  # Custom strategy instance
        api_key=polygon_api_key
    )

    # Create and run engine - all data fetching and setup is handled internally
    engine = PaperTradingEngine.from_config(config)
    success = engine.run()
    
    if success:
        print("\n✅ Paper trading completed successfully!")
        print("   Check predictions/decisions/ for decision records")


if __name__ == "__main__":
    main()