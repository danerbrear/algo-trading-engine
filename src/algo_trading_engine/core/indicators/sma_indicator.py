from algo_trading_engine.common.logger import get_logger
from algo_trading_engine.core.indicators.indicator import Indicator
from datetime import datetime
import pandas as pd

from algo_trading_engine.enums import BarTimeInterval


class SMAIndicator(Indicator):
    """
    Simple Moving Average indicator for any time frame.

    Computes the arithmetic mean of a specified column over a rolling window of bars.
    Supports daily, hourly, and minute bar intervals.
    """

    def __init__(
        self,
        period: int,
        period_unit: BarTimeInterval = BarTimeInterval.DAY,
        column: str = "Close",
    ):
        """
        Args:
            period: Number of bars in the rolling window
            period_unit: Time interval for bars (DAY, HOUR, MINUTE)
            column: DataFrame column to average (default "Close")
        """
        super().__init__(name=f"SMA_{period}")

        if period is None or period <= 0:
            raise ValueError("Period must be a positive integer")

        self.period = period
        self.period_unit = period_unit
        self.column = column
        self._updated_dates: set = set()

    @property
    def warm_up_period(self) -> int:
        return self.period

    def update(self, date: datetime, data: pd.DataFrame) -> None:
        """
        Compute and store the SMA value for *date*.

        When ``period_unit`` is DAY the indicator only needs to recalculate once
        per calendar day.  If a strategy feeds bars more frequently (e.g. hourly),
        subsequent calls on the same calendar day are no-ops.

        Args:
            date: The current bar datetime
            data: DataFrame with at least the target column, indexed by datetime.
                  Must contain historical bars up to and including *date*.

        Raises:
            ValueError: If the target column is missing.
        """
        cal_date = None
        if self.period_unit == BarTimeInterval.DAY:
            cal_date = date.date() if hasattr(date, "date") else date
            if cal_date in self._updated_dates:
                return

        if self.column not in data.columns:
            raise ValueError(
                f"Column '{self.column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # If it's a weekend, it's a no-op
        if date.weekday() >= 5:
            return

        filtered = data[data.index <= date]

        if len(filtered) < self.period:
            get_logger().warning(f"Not enough data to calculate SMA for {date}")
            return

        sma_value = filtered[self.column].iloc[-self.period:].mean()
        self._values[date] = sma_value
        if self.period_unit == BarTimeInterval.DAY:
            self._updated_dates.add(cal_date)

    def print(self):
        print(f"{self.name}: {self.value}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_intraday(self) -> bool:
        return self.period_unit in (BarTimeInterval.HOUR, BarTimeInterval.MINUTE)
