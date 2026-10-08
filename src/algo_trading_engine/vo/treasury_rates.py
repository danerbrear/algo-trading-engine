from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd


@dataclass(frozen=True)
class TreasuryRates:
    """
    Value Object representing treasury rates data.
    """

    rates_data: pd.DataFrame

    def __post_init__(self):
        if self.rates_data is None or len(self.rates_data) == 0:
            raise ValueError("Treasury rates data cannot be empty")

        required_columns = ["IRX_1Y", "TNX_10Y"]
        missing_columns = [col for col in required_columns if col not in self.rates_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required treasury rate columns: {missing_columns}")

    def __eq__(self, other):
        if not isinstance(other, TreasuryRates):
            return False
        return self.rates_data.equals(other.rates_data)

    def __hash__(self):
        data_hash = hash(
            (
                tuple(self.rates_data.index),
                tuple(self.rates_data.columns),
                tuple(self.rates_data.values.flatten()),
            )
        )
        return hash((TreasuryRates, data_hash))

    def get_risk_free_rate(self, date: datetime) -> Decimal:
        try:
            if date in self.rates_data.index:
                rate = self.rates_data.loc[date, "IRX_1Y"]
                return Decimal(str(rate))

            available_dates = self.rates_data.index
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                rate = self.rates_data.loc[closest_date, "IRX_1Y"]
                return Decimal(str(rate))
        except (KeyError, IndexError, ValueError):
            pass
        return Decimal("0.0")

    def get_10_year_rate(self, date: datetime) -> Decimal:
        try:
            if date in self.rates_data.index:
                rate = self.rates_data.loc[date, "TNX_10Y"]
                return Decimal(str(rate))

            available_dates = self.rates_data.index
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                rate = self.rates_data.loc[closest_date, "TNX_10Y"]
                return Decimal(str(rate))
        except (KeyError, IndexError, ValueError):
            pass
        return Decimal("0.0")

    def get_date_range(self) -> tuple[datetime, datetime]:
        return self.rates_data.index.min(), self.rates_data.index.max()

    def is_empty(self) -> bool:
        return len(self.rates_data) == 0
