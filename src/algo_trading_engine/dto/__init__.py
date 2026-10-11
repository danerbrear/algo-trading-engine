"""
Public Data Transfer Objects (DTOs) for the Algo Trading Engine.

This sub-package provides all public DTOs needed for strategy development.
Child repositories can import these without accessing internal modules.

Example Usage:
--------------
    from algo_trading_engine.dto import (
        OptionContractDTO,
        ExpirationRangeDTO,
        ProposedPositionRequestDTO,
        DecisionResponseDTO,
    )
"""

from algo_trading_engine.dto.decisions import DecisionResponseDTO
from algo_trading_engine.dto.options_dtos import (
    OptionContractDTO,
    OptionBarDTO,
    StrikeRangeDTO,
    ExpirationRangeDTO,
    OptionsChainDTO,
)
from algo_trading_engine.dto.trade import ProposedPositionRequestDTO

__all__ = [
    "OptionContractDTO",
    "OptionBarDTO",
    "StrikeRangeDTO",
    "ExpirationRangeDTO",
    "OptionsChainDTO",
    "ProposedPositionRequestDTO",
    "DecisionResponseDTO",
]
