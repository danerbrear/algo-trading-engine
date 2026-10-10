from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from algo_trading_engine.dto import OptionBarDTO


class Benchmark():
    """
    Benchmark to compare returns against
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.start_price = 0.0
        self.end_price = None

    def set_start_price(self, start_price: float):
        """
        Set the start price for the benchmark.
        """
        self.start_price = start_price

    def set_end_price(self, end_price: float):
        """
        Set the end price for the benchmark.
        """
        self.end_price = end_price

    def get_return_dollars(self) -> float:
        """
        Get the return for the benchmark.
        """
        shares = self.initial_capital / self.start_price
        return (self.end_price - self.start_price) * shares

    def get_return_percentage(self) -> float:
        """
        Get the return for the benchmark.
        """
        if self.end_price is None:
            return None
        return (self.end_price - self.start_price) / self.start_price * 100

class OptionType(Enum):
    """
    Enum for option types (API symbols: P/C).
    """
    PUT = "P"
    CALL = "C"


   