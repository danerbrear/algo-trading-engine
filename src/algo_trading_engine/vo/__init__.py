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

# Import Position classes from position module
from algo_trading_engine.vo.position import (
    Position,
    CreditSpreadPosition,
    DebitSpreadPosition,
    LongCallPosition,
    ShortCallPosition,
    LongPutPosition,
    ShortPutPosition,
    create_position,
)

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
    # Runtime Objects - Position classes (options only)
    "Position",
    "CreditSpreadPosition",
    "DebitSpreadPosition",
    "LongCallPosition",
    "ShortCallPosition",
    "LongPutPosition",
    "ShortPutPosition",
    "create_position",
    # Runtime Objects - Other
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
