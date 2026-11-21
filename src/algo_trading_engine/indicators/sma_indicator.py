from algo_trading_engine._internal.util.market_clock import bars_through, market_index, market_time
from algo_trading_engine.logging import get_logger
from algo_trading_engine.indicators.indicator import Indicator
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

    def precompute(self, data: pd.DataFrame) -> None:
        column = self._column_series(data)

        self._updated_dates = set()
        rolling = column.rolling(window=self.period).mean()
        if not isinstance(rolling, pd.Series):
            raise TypeError(f"Rolling mean of '{self.column}' did not produce a series")

        valid = rolling.dropna()
        valid = valid.loc[[ts.weekday() < 5 for ts in market_index(valid)]]
        if not isinstance(valid, pd.Series):
            raise TypeError("Weekday filter did not produce a series")

        if self.period_unit == BarTimeInterval.DAY:
            seen_days: set = set()
            values: dict = {}
            for ts, val in valid.items():
                cal_date = market_time(ts).date()
                if cal_date in seen_days:
                    continue
                seen_days.add(cal_date)
                self._updated_dates.add(cal_date)
                values[ts] = val
            self._values = pd.Series(values, dtype=float)
        else:
            self._values = valid.astype(float)

    def update(self, date: datetime, data: pd.DataFrame) -> None:
        """
        Compute and store the SMA value for *date*.

        When ``period_unit`` is DAY the indicator only needs to recalculate once
        per calendar day.  If a strategy feeds bars more frequently (e.g. hourly),
        subsequent calls on the same calendar day are no-ops.

        Args:
            date: The current bar datetime
            data: DataFrame with at least the target column, indexed by naive US/Eastern bar timestamps.
                  Must contain historical bars up to and including *date*.

        Raises:
            ValueError: If the target column is missing.
        """
        if self.period_unit == BarTimeInterval.DAY:
            cal_date = date.date()
            if cal_date in self._updated_dates:
                return

        self._column_series(data)

        # If it's a weekend, it's a no-op
        if date.weekday() >= 5:
            return

        filtered = bars_through(data, date)

        if len(filtered) < self.period:
            get_logger().debug(f"Not enough data to calculate SMA for {date}")
            return

        window = self._column_series(filtered).iloc[-self.period :]
        self._values.at[date] = float(window.mean())
        if self.period_unit == BarTimeInterval.DAY:
            self._updated_dates.add(date.date())

    def print(self):
        print(f"{self.name}: {self.value}")

    def _column_series(self, data: pd.DataFrame) -> pd.Series:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data. Available columns: {list(data.columns)}")
        column = data[self.column]
        if not isinstance(column, pd.Series):
            raise TypeError(f"Column '{self.column}' did not resolve to a series")
        return column

    def _is_intraday(self) -> bool:
        return self.period_unit in (BarTimeInterval.HOUR, BarTimeInterval.MINUTE)
