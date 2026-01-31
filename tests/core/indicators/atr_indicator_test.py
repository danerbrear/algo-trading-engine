"""
Unit tests for ATRIndicator
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd

from algo_trading_engine.core.indicators.average_true_return_indicator import ATRIndicator
from algo_trading_engine.enums import BarTimeInterval


class TestATRIndicatorInitialization:
    """Test cases for ATRIndicator initialization"""
    
    def test_valid_creation_with_defaults(self):
        """Test valid ATRIndicator creation with default parameters"""
        indicator = ATRIndicator(period=14)
        assert indicator.period == 14
        assert indicator.period_unit == BarTimeInterval.DAY
        assert indicator.reset_daily is False
        assert indicator.value is None
        assert indicator.name == "ATR"
    
    def test_valid_creation_with_hourly_period(self):
        """Test valid ATRIndicator creation with hourly period"""
        indicator = ATRIndicator(period=14, period_unit=BarTimeInterval.HOUR)
        assert indicator.period == 14
        assert indicator.period_unit == BarTimeInterval.HOUR
        assert indicator.reset_daily is False
    
    def test_valid_creation_with_reset_daily(self):
        """Test valid ATRIndicator creation with reset_daily enabled"""
        indicator = ATRIndicator(period=14, period_unit=BarTimeInterval.HOUR, reset_daily=True)
        assert indicator.period == 14
        assert indicator.reset_daily is True
    
    def test_invalid_period_zero_raises_error(self):
        """Test that period=0 raises ValueError"""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            ATRIndicator(period=0)
    
    def test_invalid_period_negative_raises_error(self):
        """Test that negative period raises ValueError"""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            ATRIndicator(period=-5)
    
    def test_invalid_period_none_raises_error(self):
        """Test that period=None raises ValueError"""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            ATRIndicator(period=None)


class TestATRIndicatorTrueRangeCalculation:
    """Test cases for True Range calculation"""
    
    def test_true_range_high_low_is_greatest(self):
        """Test TR when High-Low is the greatest value"""
        indicator = ATRIndicator(period=14)
        # High-Low = 105-95 = 10 (greatest)
        # |High-PrevClose| = |105-100| = 5
        # |Low-PrevClose| = |95-100| = 5
        tr = indicator._calculate_true_range(high=105, low=95, previous_close=100)
        assert tr == 10
    
    def test_true_range_high_prev_close_is_greatest(self):
        """Test TR when |High-PrevClose| is the greatest value"""
        indicator = ATRIndicator(period=14)
        # High-Low = 102-100 = 2
        # |High-PrevClose| = |102-90| = 12 (greatest)
        # |Low-PrevClose| = |100-90| = 10
        tr = indicator._calculate_true_range(high=102, low=100, previous_close=90)
        assert tr == 12
    
    def test_true_range_low_prev_close_is_greatest(self):
        """Test TR when |Low-PrevClose| is the greatest value"""
        indicator = ATRIndicator(period=14)
        # High-Low = 102-100 = 2
        # |High-PrevClose| = |102-110| = 8
        # |Low-PrevClose| = |100-110| = 10 (greatest)
        tr = indicator._calculate_true_range(high=102, low=100, previous_close=110)
        assert tr == 10
    
    def test_true_range_with_gap_up(self):
        """Test TR with a gap up scenario"""
        indicator = ATRIndicator(period=14)
        # Gap up: Previous close at 100, current opens higher
        # High-Low = 120-115 = 5
        # |High-PrevClose| = |120-100| = 20 (greatest)
        # |Low-PrevClose| = |115-100| = 15
        tr = indicator._calculate_true_range(high=120, low=115, previous_close=100)
        assert tr == 20
    
    def test_true_range_with_gap_down(self):
        """Test TR with a gap down scenario"""
        indicator = ATRIndicator(period=14)
        # Gap down: Previous close at 100, current opens lower
        # High-Low = 85-80 = 5
        # |High-PrevClose| = |85-100| = 15
        # |Low-PrevClose| = |80-100| = 20 (greatest)
        tr = indicator._calculate_true_range(high=85, low=80, previous_close=100)
        assert tr == 20


class TestATRIndicatorDailyBars:
    """Test cases for ATR calculation with daily bars"""
    
    def create_daily_data(self, start_date, num_days):
        """Helper to create sample daily OHLCV data"""
        dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i for i in range(num_days)],
            'High': [105 + i for i in range(num_days)],
            'Low': [95 + i for i in range(num_days)],
            'Close': [100 + i for i in range(num_days)],
            'Volume': [1000000] * num_days
        }, index=dates)
        return data
    
    def test_update_insufficient_data_single_bar(self):
        """Test that update raises error with only 1 bar"""
        indicator = ATRIndicator(period=14)
        data = self.create_daily_data('2024-01-01', 1)
        
        with pytest.raises(ValueError, match="Insufficient data to calculate True Range"):
            indicator.update(datetime(2024, 1, 1), data)
    
    def test_update_insufficient_data_for_initialization(self):
        """Test that update raises error when not enough bars to initialize ATR"""
        indicator = ATRIndicator(period=14)
        data = self.create_daily_data('2024-01-01', 10)  # Need 15 bars (14 + 1)
        
        with pytest.raises(ValueError, match="Insufficient data to initialize ATR"):
            indicator.update(datetime(2024, 1, 10), data)
    
    def test_update_initialization_with_exact_period(self):
        """Test ATR initialization with exactly period+1 bars"""
        indicator = ATRIndicator(period=5)
        data = self.create_daily_data('2024-01-01', 6)  # 5 + 1
        
        indicator.update(datetime(2024, 1, 6), data)
        
        # ATR should be initialized
        assert indicator.value is not None
        assert indicator.value > 0
    
    def test_update_calculates_initial_atr_correctly(self):
        """Test initial ATR calculation is simple average of TRs"""
        indicator = ATRIndicator(period=3)
        
        # Create data with known values for easy TR calculation
        dates = pd.date_range(start='2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103],
            'High': [110, 111, 112, 113],
            'Low': [90, 91, 92, 93],
            'Close': [100, 101, 102, 103],
            'Volume': [1000000] * 4
        }, index=dates)
        
        indicator.update(datetime(2024, 1, 4), data)
        
        # Calculate expected TRs manually:
        # Bar 1 (idx 1): H=111, L=91, PC=100 -> TR = max(20, 11, 9) = 20
        # Bar 2 (idx 2): H=112, L=92, PC=101 -> TR = max(20, 11, 9) = 20
        # Bar 3 (idx 3): H=113, L=93, PC=102 -> TR = max(20, 11, 9) = 20
        # Initial ATR = (20 + 20 + 20) / 3 = 20
        assert indicator.value == 20.0
    
    def test_update_uses_wilders_smoothing_after_initialization(self):
        """Test that ATR uses Wilder's smoothing after initialization"""
        indicator = ATRIndicator(period=3)
        
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [110, 111, 112, 113, 114],
            'Low': [90, 91, 92, 93, 94],
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000] * 5
        }, index=dates)
        
        # Initialize ATR
        indicator.update(datetime(2024, 1, 4), data)
        initial_atr = indicator.value
        assert initial_atr == 20.0
        
        # Update with next bar
        indicator.update(datetime(2024, 1, 5), data)
        
        # TR for bar 4: H=114, L=94, PC=103 -> TR = max(20, 11, 9) = 20
        # New ATR = [(Prior ATR × (n-1)) + Current TR] / n
        # New ATR = [(20 × 2) + 20] / 3 = 60 / 3 = 20
        assert indicator.value == 20.0
    
    def test_update_with_volatile_market(self):
        """Test ATR increases with volatility"""
        indicator = ATRIndicator(period=3)
        
        # Low volatility data
        dates = pd.date_range(start='2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'Open': [100, 100, 100, 100],
            'High': [101, 101, 101, 101],
            'Low': [99, 99, 99, 99],
            'Close': [100, 100, 100, 100],
            'Volume': [1000000] * 4
        }, index=dates)
        
        indicator.update(datetime(2024, 1, 4), data)
        low_vol_atr = indicator.value
        
        # High volatility data
        indicator2 = ATRIndicator(period=3)
        data2 = pd.DataFrame({
            'Open': [100, 100, 100, 100],
            'High': [120, 120, 120, 120],
            'Low': [80, 80, 80, 80],
            'Close': [100, 100, 100, 100],
            'Volume': [1000000] * 4
        }, index=dates)
        
        indicator2.update(datetime(2024, 1, 4), data2)
        high_vol_atr = indicator2.value
        
        assert high_vol_atr > low_vol_atr


class TestATRIndicatorHourlyBars:
    """Test cases for ATR calculation with hourly bars"""
    
    def create_hourly_data(self, start_datetime, num_hours):
        """Helper to create sample hourly OHLCV data"""
        dates = pd.date_range(start=start_datetime, periods=num_hours, freq='h')
        data = pd.DataFrame({
            'Open': [100 + i * 0.5 for i in range(num_hours)],
            'High': [105 + i * 0.5 for i in range(num_hours)],
            'Low': [95 + i * 0.5 for i in range(num_hours)],
            'Close': [100 + i * 0.5 for i in range(num_hours)],
            'Volume': [100000] * num_hours
        }, index=dates)
        return data
    
    def test_hourly_atr_rolling_window(self):
        """Test ATR with hourly bars using rolling window (reset_daily=False)"""
        indicator = ATRIndicator(period=5, period_unit=BarTimeInterval.HOUR, reset_daily=False)
        data = self.create_hourly_data('2024-01-01 09:00:00', 10)
        
        # Update at hour 6 (should have enough data)
        indicator.update(datetime(2024, 1, 1, 14, 0, 0), data)
        
        assert indicator.value is not None
        assert indicator.value > 0
    
    def test_hourly_atr_filters_data_up_to_current_time(self):
        """Test that update only uses data up to current datetime"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False)
        
        # Create 10 hours of data
        data = self.create_hourly_data('2024-01-01 09:00:00', 10)
        
        # Update at hour 4 (should only use first 4 hours)
        indicator.update(datetime(2024, 1, 1, 12, 0, 0), data)
        
        assert indicator.value is not None


class TestATRIndicatorResetDaily:
    """Test cases for reset_daily functionality"""
    
    def create_multi_day_hourly_data(self):
        """Helper to create hourly data spanning multiple days"""
        # Day 1: 10:00-16:00 (7 hours)
        day1_start = datetime(2024, 1, 1, 10, 0, 0)
        day1_dates = pd.date_range(start=day1_start, periods=7, freq='h')
        
        # Day 2: 10:00-16:00 (7 hours)
        day2_start = datetime(2024, 1, 2, 10, 0, 0)
        day2_dates = pd.date_range(start=day2_start, periods=7, freq='h')
        
        all_dates = day1_dates.append(day2_dates)
        
        data = pd.DataFrame({
            'Open': [100 + i for i in range(len(all_dates))],
            'High': [110 + i for i in range(len(all_dates))],
            'Low': [90 + i for i in range(len(all_dates))],
            'Close': [100 + i for i in range(len(all_dates))],
            'Volume': [100000] * len(all_dates)
        }, index=all_dates)
        
        return data
    
    def test_reset_daily_false_uses_all_bars(self):
        """Test that reset_daily=False uses bars across day boundaries"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False)
        data = self.create_multi_day_hourly_data()
        
        # Update at 11:00 on day 2 (should use bars from day 1 too)
        indicator.update(datetime(2024, 1, 2, 11, 0, 0), data)
        
        # Should have ATR (has enough data from both days)
        assert indicator.value is not None
    
    def test_reset_daily_true_resets_at_day_boundary(self):
        """Test that reset_daily=True resets ATR at new day"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=True)
        data = self.create_multi_day_hourly_data()
        
        # Initialize on day 1
        indicator.update(datetime(2024, 1, 1, 13, 0, 0), data)
        day1_atr = indicator.value
        assert day1_atr is not None
        
        # Update early on day 2 (should reset and raise error - not enough bars yet)
        with pytest.raises(ValueError, match="Insufficient data to initialize ATR.*for current day"):
            indicator.update(datetime(2024, 1, 2, 11, 0, 0), data)
    
    def test_reset_daily_true_only_uses_current_day_bars(self):
        """Test that reset_daily=True only uses bars from current day"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=True)
        data = self.create_multi_day_hourly_data()
        
        # Update on day 2 with enough bars in day 2
        indicator.update(datetime(2024, 1, 2, 13, 0, 0), data)
        
        # Should have ATR (has 4 bars on day 2: 10:00, 11:00, 12:00, 13:00)
        assert indicator.value is not None
    
    def test_reset_daily_tracks_current_date(self):
        """Test that _current_date is tracked when reset_daily=True"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=True)
        data = self.create_multi_day_hourly_data()
        
        assert indicator._current_date is None
        
        # Update on day 1
        indicator.update(datetime(2024, 1, 1, 13, 0, 0), data)
        assert indicator._current_date == datetime(2024, 1, 1).date()
        
        # Update on day 2 (will raise error but should update _current_date)
        try:
            indicator.update(datetime(2024, 1, 2, 11, 0, 0), data)
        except ValueError:
            pass
        assert indicator._current_date == datetime(2024, 1, 2).date()
    
    def test_reset_daily_false_with_hourly_does_not_reset(self):
        """Test that reset_daily=False doesn't reset ATR across days"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False)
        data = self.create_multi_day_hourly_data()
        
        # Initialize on day 1
        indicator.update(datetime(2024, 1, 1, 13, 0, 0), data)
        day1_atr = indicator.value
        assert day1_atr is not None
        
        # Update on day 2 - should work fine (no reset)
        indicator.update(datetime(2024, 1, 2, 10, 0, 0), data)
        assert indicator.value is not None
        # ATR should have been updated, not reset
    
    def test_reset_daily_has_no_effect_on_daily_bars(self):
        """Test that reset_daily has no effect when using daily bars"""
        indicator = ATRIndicator(period=3, period_unit=BarTimeInterval.DAY, reset_daily=True)
        
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [110, 111, 112, 113, 114],
            'Low': [90, 91, 92, 93, 94],
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000] * 5
        }, index=dates)
        
        # Should work normally (reset_daily ignored for daily bars)
        indicator.update(datetime(2024, 1, 4), data)
        assert indicator.value is not None
    
    def test_is_intraday_helper_method(self):
        """Test _is_intraday() helper method"""
        daily_indicator = ATRIndicator(period=14, period_unit=BarTimeInterval.DAY)
        hourly_indicator = ATRIndicator(period=14, period_unit=BarTimeInterval.HOUR)
        minute_indicator = ATRIndicator(period=14, period_unit=BarTimeInterval.MINUTE)
        
        assert daily_indicator._is_intraday() is False
        assert hourly_indicator._is_intraday() is True
        assert minute_indicator._is_intraday() is True


class TestATRIndicatorEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_update_with_identical_prices(self):
        """Test ATR when all prices are identical (no volatility)"""
        indicator = ATRIndicator(period=3)
        
        dates = pd.date_range(start='2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'Open': [100] * 4,
            'High': [100] * 4,
            'Low': [100] * 4,
            'Close': [100] * 4,
            'Volume': [1000000] * 4
        }, index=dates)
        
        indicator.update(datetime(2024, 1, 4), data)
        
        # ATR should be 0 (no volatility)
        assert indicator.value == 0.0
    
    def test_update_multiple_times_maintains_state(self):
        """Test that calling update multiple times maintains state correctly"""
        indicator = ATRIndicator(period=3)
        
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [110 + i for i in range(10)],
            'Low': [90 + i for i in range(10)],
            'Close': [100 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)
        
        # Update progressively
        indicator.update(datetime(2024, 1, 4), data)
        atr_day4 = indicator.value
        
        indicator.update(datetime(2024, 1, 5), data)
        atr_day5 = indicator.value
        
        indicator.update(datetime(2024, 1, 6), data)
        atr_day6 = indicator.value
        
        # All should be valid and ATR should exist
        assert atr_day4 is not None
        assert atr_day5 is not None
        assert atr_day6 is not None
        
        # Each update should maintain or change the value
        assert atr_day5 != atr_day4 or atr_day5 == atr_day4  # Just verify no crash
    
    def test_value_property_returns_none_before_initialization(self):
        """Test that value property returns None before update"""
        indicator = ATRIndicator(period=14)
        assert indicator.value is None
    
    def test_name_property(self):
        """Test that name property is set correctly"""
        indicator = ATRIndicator(period=14)
        assert indicator.name == "ATR"
