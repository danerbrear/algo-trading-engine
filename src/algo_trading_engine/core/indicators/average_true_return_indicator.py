from algo_trading_engine.core.indicators.indicator import Indicator
from datetime import datetime
import pandas as pd

from algo_trading_engine.enums import BarTimeInterval

class ATRIndicator(Indicator):
    """
    ATRIndicator is an indicator that calculates the average true range (ATR) of a stock.
    Answers the question: "How noisy or volatile is the stock right now?"
    """
    def __init__(self, period: int, period_unit: BarTimeInterval = BarTimeInterval.DAY, reset_daily: bool = False):
        """
        Initialize the ATR indicator.
        
        Args:
            period: Number of bars to use for ATR calculation
            period_unit: Time interval for bars (DAY, HOUR, MINUTE)
            reset_daily: If True and period_unit is intraday (HOUR/MINUTE), only use bars 
                        from the current trading day. ATR resets at the start of each new day.
                        If False, uses rolling window across all bars regardless of day boundaries.
        """
        super().__init__(name="ATR")

        if period is None or period <= 0:
            raise ValueError("Period must be a positive integer")
        
        self.period = period
        self.period_unit = period_unit
        self.reset_daily = reset_daily
        self._value = None
        self._current_date = None  # Track which day we're calculating ATR for

    @property
    def value(self) -> float:
        return self._value

    def update(self, date: datetime, data: pd.DataFrame) -> None:
        """
        Update the ATR indicator for a given bar of data.
        
        Args:
            date: The current date/time to update the indicator for
            data: DataFrame with OHLCV data indexed by datetime.
                  Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
                  The data should contain historical bars up to and including the current date.
                  
        Raises:
            ValueError: If there is insufficient data to calculate ATR
        """
        # Check if we've moved to a new trading day (reset ATR if reset_daily is True)
        current_day = date.date()
        if self.reset_daily and self._is_intraday() and self._current_date != current_day:
            # New trading day - reset ATR
            self._value = None
            self._current_date = current_day
        
        # If it's a weekend and the data doesn't have that weekend date, 
        # use the most recent trading day (Friday)
        filter_date = date
        if date.weekday() >= 5 and date not in data.index:  # 5 = Saturday, 6 = Sunday
            # Calculate days back to Friday
            days_back = date.weekday() - 4  # 4 = Friday
            filter_date = date - pd.Timedelta(days=days_back)
        
        # Filter data up to the current/adjusted date
        data_up_to_date = data[data.index <= filter_date]
        
        # If reset_daily is True and we're using intraday bars, only use today's bars
        filter_day = None
        if self.reset_daily and self._is_intraday():
            # Use the adjusted filter date's day for intraday filtering
            filter_day = filter_date.date() if hasattr(filter_date, 'date') else filter_date
            data_up_to_date = data_up_to_date[data_up_to_date.index.date == filter_day]
        
        # Need at least 2 bars to calculate TR (need previous close)
        if len(data_up_to_date) < 2:
            raise ValueError(
                f"Insufficient data to calculate True Range. Need at least 2 bars, but only have {len(data_up_to_date)} "
                f"{'for current day ' + str(filter_day) if self.reset_daily and self._is_intraday() else ''}"
            )
        
        # Calculate the true range for the current bar
        current_high = data_up_to_date['High'].iloc[-1]
        current_low = data_up_to_date['Low'].iloc[-1]
        previous_close = data_up_to_date['Close'].iloc[-2]
        
        tr = self._calculate_true_range(current_high, current_low, previous_close)
        
        # Initialize ATR if this is the first calculation
        if self._value is None:
            # Need 'period' bars to initialize
            if len(data_up_to_date) < self.period + 1:  # +1 because we need a previous close
                raise ValueError(
                    f"Insufficient data to initialize ATR. Need at least {self.period + 1} bars "
                    f"for period={self.period}, but only have {len(data_up_to_date)} bars "
                    f"{'for current day ' + str(filter_day) if self.reset_daily and self._is_intraday() else 'up to ' + str(filter_date)}"
                )
            
            # Calculate initial ATR as simple average of first 'period' TRs
            true_ranges = []
            for i in range(1, self.period + 1):
                h = data_up_to_date['High'].iloc[i]
                l = data_up_to_date['Low'].iloc[i]
                pc = data_up_to_date['Close'].iloc[i-1]
                tr_i = self._calculate_true_range(h, l, pc)
                true_ranges.append(tr_i)
            self._value = sum(true_ranges) / len(true_ranges)
        else:
            # Use Wilder's smoothing method
            # ATR = [(Prior ATR × (n-1)) + Current TR] / n
            self._value = ((self._value * (self.period - 1)) + tr) / self.period
    
    def print(self):
        print(f"{self.name}: {self._value}")

    def _is_intraday(self) -> bool:
        """
        Check if the period unit is intraday (HOUR or MINUTE).
        
        Returns:
            bool: True if period_unit is HOUR or MINUTE, False otherwise
        """
        return self.period_unit in (BarTimeInterval.HOUR, BarTimeInterval.MINUTE)

    def _calculate_true_range(self, high: float, low: float, previous_close: float) -> float:
        """
        Calculate the true range for a given bar.
        
        True Range is the greatest of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        
        Args:
            high: Current bar's high price
            low: Current bar's low price
            previous_close: Previous bar's close price
            
        Returns:
            float: The true range value
        """
        return max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )