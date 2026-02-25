#!/usr/bin/env python3
"""
Example: Custom strategy backtest

This example shows how to create a custom strategy by inheriting from the
Strategy base class and using it with the BacktestEngine.
"""

import os
from datetime import datetime
import sys
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

# Import from the public API
from algo_trading_engine import BacktestEngine, BacktestConfig
from algo_trading_engine.enums import BarTimeInterval

# Import custom strategy using absolute path
strategy_path = Path(__file__).parent.parent / "strategies" / "custom_strategy.py"
spec = importlib.util.spec_from_file_location("custom_strategy", strategy_path)
custom_strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_strategy_module)
MyCustomStrategy = custom_strategy_module.MyCustomStrategy

# Load environment variables from .env file
load_dotenv()


def main():
    """Run backtest using a custom strategy instance."""
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    
    # Create your custom strategy instance
    custom_strategy = MyCustomStrategy(
        profit_target=0.5,
        stop_loss=0.6
    )
    
    # Create configuration with strategy instance
    # Note: strategy_type can be either a string (built-in) or a Strategy instance (custom)
    config = BacktestConfig(
        initial_capital=3000,
        start_date=datetime(2025, 8, 1),
        end_date=datetime(2025, 12, 1),
        max_position_size=0.20,
        symbol="SPY",
        strategy_type=custom_strategy,  # Pass strategy instance
        api_key=polygon_api_key,
        bar_interval=BarTimeInterval.HOUR
    )

    # Create and run engine - all data fetching and setup is handled internally
    engine = BacktestEngine.from_config(config)

    try:
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
        else:
            print("Backtest failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Backtest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
