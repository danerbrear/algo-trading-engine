"""
Public Enums for the Algo Trading Engine.

This sub-package provides all public enums needed for strategy development.
Child repositories can import these without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.enums import StrategyType, OptionType, MarketStateType, SignalType

    if strategy_type == StrategyType.CALL_CREDIT_SPREAD:
        ...
"""

from algo_trading_engine.enums.bar_time_interval import BarTimeInterval
from algo_trading_engine.enums.market_state_type import MarketStateType
from algo_trading_engine.enums.option_type import OptionType
from algo_trading_engine.enums.signal_type import SignalType
from algo_trading_engine.enums.strategy_type import StrategyType
from algo_trading_engine.enums.universal_close_condition import UniversalCloseCondition

__all__ = [
    "StrategyType",
    "OptionType",
    "MarketStateType",
    "SignalType",
    "BarTimeInterval",
    "UniversalCloseCondition",
]
