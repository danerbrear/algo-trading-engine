from algo_trading_engine.core.indicators.indicator import Indicator
from datetime import datetime
import pandas as pd

from algo_trading_engine.enums import BarTimeInterval


class SMAIndicator(Indicator):
    """
    Simple Moving Average indicator for any time frame.

    Computes the arithmetic mean of a specified column over a rolling window of bars.
    Supports daily, hourly, and minute bar intervals with optional daily reset for
    intraday periods.
    """

    def __init__(
        self,
        period: int,
        period_unit: BarTimeInterval = BarTimeInterval.DAY,
        column: str = "Close",
        reset_daily: bool = False,
    ):
        """
        Args:
            period: Number of bars in the rolling window
            period_unit: Time interval for bars (DAY, HOUR, MINUTE)
            column: DataFrame column to average (default "Close")
            reset_daily: If True and period_unit is intraday, only use bars from
                         the current trading day so the SMA resets each morning.
        """
        super().__init__(name=f"SMA_{period}")

        if period is None or period <= 0:
            raise ValueError("Period must be a positive integer")

        self.period = period
        self.period_unit = period_unit
        self.column = column
        self.reset_daily = reset_daily

    def update(self, date: datetime, data: pd.DataFrame) -> None:
        """
        Compute and store the SMA value for *date*.

        Args:
            date: The current bar datetime
            data: DataFrame with at least the target column, indexed by datetime.
                  Must contain historical bars up to and including *date*.

        Raises:
            ValueError: If the target column is missing or there are fewer bars
                        than the requested period.
        """
        if self.column not in data.columns:
            raise ValueError(
                f"Column '{self.column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        filter_date = self._adjust_for_weekend(date, data)
        filtered = data[data.index <= filter_date]

        if self.reset_daily and self._is_intraday():
            filter_day = filter_date.date() if hasattr(filter_date, "date") else filter_date
            filtered = filtered[filtered.index.date == filter_day]

        if len(filtered) < self.period:
            context = ""
            if self.reset_daily and self._is_intraday():
                filter_day = filter_date.date() if hasattr(filter_date, "date") else filter_date
                context = f" for current day {filter_day}"
            raise ValueError(
                f"Insufficient data to calculate SMA({self.period}). "
                f"Need at least {self.period} bars, but only have {len(filtered)}{context}"
            )

        sma_value = filtered[self.column].iloc[-self.period:].mean()
        self._values[date] = sma_value

    def print(self):
        print(f"{self.name}: {self.value}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_intraday(self) -> bool:
        return self.period_unit in (BarTimeInterval.HOUR, BarTimeInterval.MINUTE)

    @staticmethod
    def _adjust_for_weekend(date: datetime, data: pd.DataFrame) -> datetime:
        """Fall back to the most recent trading day when *date* is a weekend."""
        if date.weekday() >= 5 and date not in data.index:
            days_back = date.weekday() - 4  # 4 = Friday
            return date - pd.Timedelta(days=days_back)
        return date
