"""
Public Enums for the Algo Trading Engine.

This sub-package provides all public enums needed for strategy development.
Child repositories can import these without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.enums import StrategyType, OptionType, MarketStateType, SignalType
    
    # Use in custom strategy
    if strategy_type == StrategyType.CALL_CREDIT_SPREAD:
        ...
    
    # Use OptionType enum
    if option.option_type == OptionType.CALL:
        ...
"""

# Import from backtest models
from algo_trading_engine.backtest.models import StrategyType

# Import from common models
from algo_trading_engine.common.models import (
    OptionType,
    MarketStateType,
    SignalType,
)

# Define public API
__all__ = [
    "StrategyType",
    "OptionType",
    "MarketStateType",
    "SignalType",
]
