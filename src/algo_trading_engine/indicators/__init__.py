"""
Technical Indicators for the Algo Trading Engine.

This sub-package provides technical indicators for use in custom strategies.
All indicators inherit from the Indicator base class and can be added to
strategies via the indicators parameter.

Example Usage:
--------------
    from algo_trading_engine import Strategy
    from algo_trading_engine.indicators import ATRIndicator
    from algo_trading_engine.enums import BarTimeInterval
    
    # Create indicator
    atr = ATRIndicator(
        period=14, 
        period_unit=BarTimeInterval.HOUR,
        reset_daily=True
    )
    
    # Use in custom strategy
    class MyStrategy(Strategy):
        def __init__(self):
            super().__init__(indicators=[atr])
        
        def on_new_date(self, date, positions, add_position, remove_position):
            super().on_new_date(date, positions, add_position, remove_position)
            
            # Access indicator value
            current_atr = self.indicators[0].value
            if current_atr and current_atr > 5.0:
                # High volatility - adjust strategy
                pass
"""

# Import base indicator class
from algo_trading_engine.core.indicators.indicator import Indicator

# Import public indicators
from algo_trading_engine.core.indicators.average_true_return_indicator import ATRIndicator

# Define public API
__all__ = [
    "Indicator",
    "ATRIndicator",
]
