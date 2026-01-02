#!/usr/bin/env python3
"""Example usage in a child repository."""

import os
from dotenv import load_dotenv
from algo_trading_engine.core import PaperTradingEngine
from algo_trading_engine.models import PaperTradingConfig

# Load environment variables from .env file
load_dotenv()

def main():
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    
    config = PaperTradingConfig(
        initial_capital=100000,
        symbol="SPY",
        strategy_type="velocity_momentum",
        max_position_size=0.40,
        api_key=polygon_api_key
    )

    # Create and run engine
    engine = PaperTradingEngine.from_config(config)
    success = engine.run()
    
    if success:
        print("\n✅ Paper trading completed successfully!")
        print("   Check predictions/decisions/ for decision records")

if __name__ == "__main__":
    main()