#!/usr/bin/env python3
"""
Example: Custom strategy backtest

This example shows how to create a custom strategy by inheriting from the
Strategy base class and using it with the BacktestEngine.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Import from the public API
from algo_trading_engine import BacktestEngine, BacktestConfig, Strategy

# Load environment variables from .env file
load_dotenv()


class MyCustomStrategy(Strategy):
    """
    Example custom strategy.
    
    This is a simple example - you would implement your own logic here.
    The Strategy base class provides the interface that BacktestEngine expects.
    """
    
    def __init__(self, profit_target=0.5, stop_loss=0.6, start_date_offset=60):
        """Initialize the custom strategy."""
        super().__init__(profit_target, stop_loss, start_date_offset)
        # Add any custom initialization here
    
    def on_new_date(self, date, positions, add_position, remove_position):
        """
        Called for each trading day.
        
        Implement your strategy logic here:
        - Analyze current market conditions using self.data
        - Check existing positions
        - Add new positions via add_position()
        - Close positions via remove_position()
        """
        # Example: Simple logic (replace with your own)
        # Close profitable positions
        for position in positions:
            if self._profit_target_hit(position, position.exit_price):
                remove_position(date, position, position.exit_price)
    
    def on_end(self, positions, remove_position, date):
        """
        Called at the end of backtest to close remaining positions.
        """
        for position in positions:
            remove_position(date, position, position.exit_price)
    
    def validate_data(self, data):
        """
        Validate that the data has the required columns/features.
        
        Returns:
            True if valid, False otherwise
        """
        # Check for required columns
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        return all(col in data.columns for col in required_columns)


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
        start_date=datetime(2025, 1, 2),
        end_date=datetime(2026, 1, 2),
        max_position_size=0.20,
        symbol="SPY",
        strategy_type=custom_strategy,  # Pass strategy instance
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


if __name__ == "__main__":
    main()

