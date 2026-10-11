"""Public decision response DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from algo_trading_engine.dto.trade import ProposedPositionRequestDTO


@dataclass(frozen=True)
class DecisionResponseDTO:
    """Immutable record of an accepted decision for a proposal."""

    id: str
    proposal: ProposedPositionRequestDTO
    decided_at: str
    rationale: str
    quantity: Optional[int] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    closed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposal": self.proposal.to_dict(),
            "decided_at": self.decided_at,
            "rationale": self.rationale,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "closed_at": self.closed_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "DecisionResponseDTO":
        return DecisionResponseDTO(
            id=str(data["id"]),
            proposal=ProposedPositionRequestDTO.from_dict(data["proposal"]),
            decided_at=str(data["decided_at"]),
            rationale=str(data["rationale"]),
            quantity=data.get("quantity"),
            entry_price=data.get("entry_price"),
            exit_price=data.get("exit_price"),
            closed_at=data.get("closed_at"),
        )
