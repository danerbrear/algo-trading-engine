"""
Value Objects and Runtime Types for the Algo Trading Engine.

This sub-package provides value objects and runtime data models
needed for strategy development. Child repositories can import these
without accessing internal modules.

Note: Enums are now in algo_trading_engine.enums package.

Example Usage:
--------------
    from algo_trading_engine.vo import Position, Option, TreasuryRates, StrikePrice, MarketState
    from algo_trading_engine.enums import StrategyType, OptionType
    
    # Use in custom strategy
    position = Position(
        symbol="SPY",
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        ...
    )
    
    # Use Value Objects
    strike = StrikePrice(Decimal('100'))
    market_state = MarketState(...)
"""

# Import from common models
from algo_trading_engine.common.models import (
    Option,
    TreasuryRates,
)

# Lazy import Position to avoid circular dependency with dto
# Position is imported from backtest.models which imports from dto
def _get_position():
    from algo_trading_engine.backtest.models import Position
    return Position

# Use __getattr__ for lazy loading to break circular imports
def __getattr__(name: str):
    if name == "Position":
        return _get_position()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Import public Value Objects
from algo_trading_engine.vo.value_objects import (
    StrikePrice,
    ExpirationDate,
    MarketState,
    TradingSignal,
    PriceRange,
    Volatility,
)

# Define public API
__all__ = [
    # Runtime Objects
    "Position",
    "Option",
    # Value Objects
    "TreasuryRates",
    "StrikePrice",
    "ExpirationDate",
    "MarketState",
    "TradingSignal",
    "PriceRange",
    "Volatility",
]
