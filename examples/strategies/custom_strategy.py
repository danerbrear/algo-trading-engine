from algo_trading_engine import Strategy

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