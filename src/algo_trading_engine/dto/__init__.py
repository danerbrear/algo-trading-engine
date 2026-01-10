"""
Data Transfer Objects (DTOs) for the Algo Trading Engine.

This sub-package provides all DTOs needed for strategy development.
Child repositories can import these without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.dto import OptionContractDTO, ExpirationRangeDTO
"""

# Import options DTOs
from algo_trading_engine.common.options_dtos import (
    OptionContractDTO,
    OptionBarDTO,
    StrikeRangeDTO,
    ExpirationRangeDTO,
    OptionsChainDTO,
)

# Define public API
__all__ = [
    "OptionContractDTO",
    "OptionBarDTO",
    "StrikeRangeDTO",
    "ExpirationRangeDTO",
    "OptionsChainDTO",
]
