"""
Technical Indicators for the Algo Trading Engine.

This sub-package provides technical indicators for use in custom strategies.
All indicators inherit from the Indicator base class and can be added to
strategies via the add_indicator() method.

Example Usage:
--------------
    from algo_trading_engine import Strategy
    from algo_trading_engine.indicators import ATRIndicator
    from algo_trading_engine.enums import BarTimeInterval
    
    # Create custom strategy with indicators
    class MyStrategy(Strategy):
        def __init__(self):
            super().__init__()
            
            # Add indicators using add_indicator()
            atr = ATRIndicator(
                period=14, 
                period_unit=BarTimeInterval.HOUR,
                reset_daily=True
            )
            self.add_indicator(atr)
        
        def on_new_date(self, date, positions, add_position, remove_position):
            super().on_new_date(date, positions, add_position, remove_position)
            
            # Access indicator value using get_indicator()
            atr = self.get_indicator(ATRIndicator)
            if atr and atr.value and atr.value > 5.0:
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
