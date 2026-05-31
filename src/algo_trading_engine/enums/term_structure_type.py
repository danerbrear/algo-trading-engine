"""Volatility term structure classification."""

from enum import Enum


class TermStructureType(Enum):
    """Implied volatility term structure shape between adjacent maturities."""

    CONTANGO = "contango"
    BACKWARDATION = "backwardation"
