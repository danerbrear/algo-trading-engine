"""
Runtime Types and Value Objects for the Algo Trading Engine.

This sub-package provides value objects, enums, and runtime data models
needed for strategy development. Child repositories can import these
without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.types import StrategyType, Position, Option, TreasuryRates
    
    # Use in custom strategy
    position = Position(
        symbol="SPY",
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        ...
    )
"""

# Import from backtest models
from algo_trading_engine.backtest.models import (
    StrategyType,
    Position,
)

# Import from common models
from algo_trading_engine.common.models import (
    Option,
    TreasuryRates,
)

# Define public API
__all__ = [
    # Enums
    "StrategyType",
    # Runtime Objects
    "Position",
    "Option",
    # Value Objects
    "TreasuryRates",
]
