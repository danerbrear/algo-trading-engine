from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class Indicator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def update(self, date: datetime, data: pd.DataFrame) -> float:
        pass

    @abstractmethod
    def print(self):
        pass