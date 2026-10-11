"""Public trade request DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Option


@dataclass(frozen=True)
class ProposedPositionRequestDTO:
    """Represents a proposed position to open.

    Legs reuse the existing Option VO for clarity and compatibility with the
    rest of the system. Strategy type uses the existing StrategyType enum.
    All date/time values are ISO8601 strings for JSON persistence.
    """

    symbol: str
    strategy_type: StrategyType
    legs: Tuple[Option, ...]
    credit: float
    width: float
    probability_of_profit: float
    confidence: float
    expiration_date: str
    created_at: str  # ISO timestamp
    strategy_name: str = "unknown"  # e.g., "velocity_momentum", "upward_trend_reversal"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy_type": self.strategy_type.value,
            "legs": [leg.to_dict() for leg in self.legs],
            "credit": self.credit,
            "width": self.width,
            "probability_of_profit": self.probability_of_profit,
            "confidence": self.confidence,
            "expiration_date": self.expiration_date,
            "created_at": self.created_at,
            "strategy_name": self.strategy_name,
        }

    @staticmethod
    def from_dict(data: dict) -> "ProposedPositionRequestDTO":
        return ProposedPositionRequestDTO(
            symbol=data["symbol"],
            strategy_type=StrategyType(data["strategy_type"]),
            legs=tuple(Option.from_dict(opt) for opt in data.get("legs", [])),
            credit=float(data["credit"]),
            width=float(data["width"]),
            probability_of_profit=float(data["probability_of_profit"]),
            confidence=float(data["confidence"]),
            expiration_date=str(data["expiration_date"]),
            created_at=str(data["created_at"]),
            strategy_name=data.get("strategy_name", "unknown"),
        )
