from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class Indicator(ABC):
    def __init__(self, name: str):
        self.name = name
        self._values = pd.Series(dtype=float)  # Series indexed by datetime to store historical values

    @property
    def value(self) -> float:
        """Get the most recent indicator value"""
        if len(self._values) == 0:
            return None
        return self._values.iloc[-1]
    
    def get_value_at(self, date: datetime) -> float:
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

    @abstractmethod
    def update(self, date: datetime, data: pd.DataFrame) -> float:
        pass

    @abstractmethod
    def print(self):
        pass