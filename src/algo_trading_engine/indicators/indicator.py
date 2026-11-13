from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd

class Indicator(ABC):
    """Base class for indicator implementations"""
    def __init__(self, name: str):
        self.name = name
        self._values = pd.Series(dtype=float)  # Series indexed by datetime to store historical values

    @property
    @abstractmethod
    def warm_up_period(self) -> int:
        """Minimum number of bars required before this indicator can produce a valid value."""

    @property
    def value(self) -> Optional[float]:
        """Get the most recent indicator value"""
        if len(self._values) == 0:
            return None
        return self._values.iloc[-1]

    @abstractmethod
    def precompute(self, data: pd.DataFrame) -> None:
        """
        Populate historical values for all bars in ``data`` in one pass.

        Implementations must fill ``_values`` and leave internal state so a
        later ``update()`` on a new bar can continue from the last value.
        """

    @abstractmethod
    def update(self, date: datetime, data: pd.DataFrame) -> float:
        """Update a value for a date"""

    @abstractmethod
    def print(self):
        pass
    
    def get_value_at(self, date: datetime) -> Optional[float]:
        """
        Get indicator value at a specific date.
        
        Args:
            date: The date to get the value for
            
        Returns:
            float: The indicator value at that date, or None if not found
        """
        if date in self._values.index:
            return self._values.loc[date]
        return None
    
    def get_values(self) -> pd.Series:
        """
        Get all historical indicator values.
        
        Returns:
            pd.Series: Series of all indicator values indexed by datetime
        """
        return self._values