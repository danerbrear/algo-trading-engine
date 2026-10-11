"""
Technical Indicators for the Algo Trading Engine.

This sub-package provides technical indicators for use in custom strategies.
All indicators inherit from the Indicator base class and can be added to
strategies via the add_indicator() method.
"""

from algo_trading_engine.indicators.average_true_return_indicator import ATRIndicator
from algo_trading_engine.indicators.indicator import Indicator
from algo_trading_engine.indicators.sma_indicator import SMAIndicator

__all__ = [
    "Indicator",
    "ATRIndicator",
    "SMAIndicator",
]
