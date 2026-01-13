"""
Public Data Transfer Objects (DTOs) for the Algo Trading Engine.

This sub-package provides all public DTOs needed for strategy development.
Child repositories can import these without accessing internal modules.

Note: Internal DTOs (e.g., prediction-related DTOs) are kept in their respective modules
and are not part of the public API.

Example Usage:
--------------
    from algo_trading_engine.dto import OptionContractDTO, ExpirationRangeDTO
"""

# Import public options DTOs
from algo_trading_engine.dto.options_dtos import (
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
