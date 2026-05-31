"""DTOs for implied forward variance term structure."""

from dataclasses import dataclass
from typing import List

from algo_trading_engine.enums.term_structure_type import TermStructureType


@dataclass(frozen=True)
class ImpliedForwardVarianceTermStructureDTO:
    """
    Implied forward variance and term-structure classification per maturity step.

    Lists align with each incremental step after the first input (index 1..n-1).
    """

    implied_forward_variances: List[float]
    term_structure_types: List[TermStructureType]

    def __post_init__(self):
        if len(self.implied_forward_variances) != len(self.term_structure_types):
            raise ValueError(
                "implied_forward_variances and term_structure_types must have the same length"
            )
        if not self.implied_forward_variances:
            raise ValueError("At least one forward variance value is required")
