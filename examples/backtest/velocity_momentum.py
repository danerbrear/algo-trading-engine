#!/usr/bin/env python3
"""
Example: Backtest velocity momentum strategy

This example shows how to use the algo-trading-engine package in a child repository.
The package provides a clean public API for backtesting trading strategies.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Import from the public API
from algo_trading_engine import BacktestEngine, BacktestConfig

# Load environment variables from .env file
load_dotenv()


def main():
    """Run backtest using the velocity momentum strategy."""
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    
    # Create configuration
    config = BacktestConfig(
        initial_capital=3000,
        start_date=datetime(2025, 1, 2),
        end_date=datetime(2026, 1, 2),
        max_position_size=0.20,
        symbol="SPY",
        strategy_type="velocity_momentum",  # Built-in strategy name
        api_key=polygon_api_key
    )

    # Create and run engine - all data fetching and setup is handled internally
    engine = BacktestEngine.from_config(config)
    engine.run()

if __name__ == "__main__":
    main()