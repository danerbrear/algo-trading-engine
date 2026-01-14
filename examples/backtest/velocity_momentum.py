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
    success = engine.run()
    
    if success:
        # Get performance metrics
        metrics = engine.get_performance_metrics()

        print("\n=== Backtest Results ===")
        print(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"Win Rate: {metrics.win_rate:.1f}%")
        print(f"Total Positions: {metrics.total_positions}")

        # Strategy-specific stats
        for strategy_stat in metrics.strategy_stats:
            print(f"\n{strategy_stat.strategy_type.value}:")
            print(f"  Win Rate: {strategy_stat.win_rate:.1f}%")
            print(f"  Total P&L: ${strategy_stat.total_pnl:,.2f}")


if __name__ == "__main__":
    main()