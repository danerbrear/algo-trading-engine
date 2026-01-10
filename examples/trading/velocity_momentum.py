#!/usr/bin/env python3
"""
Example: Paper trade velocity momentum strategy

This example shows how to use the algo-trading-engine package for paper trading
in a child repository. The package provides a clean public API for running
strategies against live market data.
"""

import os
from dotenv import load_dotenv

# Import from the public API
from algo_trading_engine import PaperTradingEngine, PaperTradingConfig

# Load environment variables from .env file
load_dotenv()


def main():
    """Run paper trading using the velocity momentum strategy."""
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    
    # Create configuration
    config = PaperTradingConfig(
        initial_capital=100000,
        symbol="SPY",
        strategy_type="velocity_momentum",  # Built-in strategy name
        max_position_size=0.40,
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