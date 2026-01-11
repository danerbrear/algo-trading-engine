"""
Runtime Types and Value Objects for the Algo Trading Engine.

This sub-package provides value objects, enums, and runtime data models
needed for strategy development. Child repositories can import these
without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.types import StrategyType, OptionType, Position, Option, TreasuryRates
    
    # Use in custom strategy
    position = Position(
        symbol="SPY",
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        ...
    )
    
    # Use OptionType enum
    if option.option_type == OptionType.CALL:
        ...
"""

# Import from backtest models
from algo_trading_engine.backtest.models import (
    StrategyType,
    Position,
)

# Import from common models
from algo_trading_engine.common.models import (
    Option,
    OptionType,
    TreasuryRates,
)

# Define public API
__all__ = [
    # Enums
    "StrategyType",
    "OptionType",
    # Runtime Objects
    "Position",
    "Option",
    # Value Objects
    "TreasuryRates",
]
