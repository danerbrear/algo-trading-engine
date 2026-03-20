"""
Unit tests for SMAIndicator
"""
import pytest
from datetime import datetime
import pandas as pd

from algo_trading_engine.core.indicators.sma_indicator import SMAIndicator
from algo_trading_engine.enums import BarTimeInterval


class TestSMAIndicatorInitialization:
    """Test cases for SMAIndicator initialization"""

    def test_valid_creation_with_defaults(self):
        indicator = SMAIndicator(period=20)
        assert indicator.period == 20
        assert indicator.period_unit == BarTimeInterval.DAY
        assert indicator.column == "Close"
        assert indicator.reset_daily is False
        assert indicator.value is None
        assert indicator.name == "SMA_20"

    def test_valid_creation_with_hourly_period(self):
        indicator = SMAIndicator(period=10, period_unit=BarTimeInterval.HOUR)
        assert indicator.period == 10
        assert indicator.period_unit == BarTimeInterval.HOUR
        assert indicator.name == "SMA_10"

    def test_valid_creation_with_custom_column(self):
        indicator = SMAIndicator(period=5, column="Volume")
        assert indicator.column == "Volume"

    def test_valid_creation_with_reset_daily(self):
        indicator = SMAIndicator(
            period=10, period_unit=BarTimeInterval.HOUR, reset_daily=True
        )
        assert indicator.reset_daily is True

    def test_invalid_period_zero_raises_error(self):
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            SMAIndicator(period=0)

    def test_invalid_period_negative_raises_error(self):
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            SMAIndicator(period=-3)

    def test_invalid_period_none_raises_error(self):
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            SMAIndicator(period=None)


class TestSMAIndicatorDailyBars:
    """Test cases for SMA calculation with daily bars"""

    def _daily_data(self, start_date: str, closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, periods=len(closes), freq="D")
        return pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [1_000_000] * len(closes),
            },
            index=dates,
        )

    def test_exact_period_calculates_correctly(self):
        data = self._daily_data("2024-01-01", [10, 20, 30])
        indicator = SMAIndicator(period=3)
        indicator.update(datetime(2024, 1, 3), data)

        assert indicator.value == pytest.approx(20.0)

    def test_more_data_than_period_uses_last_n_bars(self):
        data = self._daily_data("2024-01-01", [100, 10, 20, 30])
        indicator = SMAIndicator(period=3)
        indicator.update(datetime(2024, 1, 4), data)

        # Should use last 3 bars: 10, 20, 30
        assert indicator.value == pytest.approx(20.0)

    def test_insufficient_data_raises_error(self):
        data = self._daily_data("2024-01-01", [10, 20])
        indicator = SMAIndicator(period=5)

        with pytest.raises(ValueError, match="Insufficient data to calculate SMA"):
            indicator.update(datetime(2024, 1, 2), data)

    def test_sequential_updates_maintain_history(self):
        data = self._daily_data("2024-01-01", [10, 20, 30, 40, 50])
        indicator = SMAIndicator(period=3)

        indicator.update(datetime(2024, 1, 3), data)
        assert indicator.value == pytest.approx(20.0)

        indicator.update(datetime(2024, 1, 4), data)
        assert indicator.value == pytest.approx(30.0)

        indicator.update(datetime(2024, 1, 5), data)
        assert indicator.value == pytest.approx(40.0)

        assert len(indicator.get_values()) == 3

    def test_get_value_at_returns_correct_value(self):
        data = self._daily_data("2024-01-01", [10, 20, 30, 40])
        indicator = SMAIndicator(period=3)

        indicator.update(datetime(2024, 1, 3), data)
        indicator.update(datetime(2024, 1, 4), data)

        assert indicator.get_value_at(datetime(2024, 1, 3)) == pytest.approx(20.0)
        assert indicator.get_value_at(datetime(2024, 1, 4)) == pytest.approx(30.0)

    def test_single_bar_period(self):
        data = self._daily_data("2024-01-01", [42.5])
        indicator = SMAIndicator(period=1)
        indicator.update(datetime(2024, 1, 1), data)

        assert indicator.value == pytest.approx(42.5)

    def test_custom_column(self):
        dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
        data = pd.DataFrame(
            {
                "Close": [10, 20, 30],
                "Volume": [100, 200, 300],
            },
            index=dates,
        )
        indicator = SMAIndicator(period=3, column="Volume")
        indicator.update(datetime(2024, 1, 3), data)

        assert indicator.value == pytest.approx(200.0)

    def test_missing_column_raises_error(self):
        data = self._daily_data("2024-01-01", [10, 20, 30])
        indicator = SMAIndicator(period=3, column="NonExistent")

        with pytest.raises(ValueError, match="Column 'NonExistent' not found"):
            indicator.update(datetime(2024, 1, 3), data)


class TestSMAIndicatorHourlyBars:
    """Test cases for SMA calculation with hourly bars"""

    def _hourly_data(self, start: str, closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range(start=start, periods=len(closes), freq="h")
        return pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [100_000] * len(closes),
            },
            index=dates,
        )

    def test_hourly_sma_rolling_window(self):
        data = self._hourly_data("2024-01-01 09:00", [10, 20, 30, 40, 50])
        indicator = SMAIndicator(
            period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False
        )
        indicator.update(datetime(2024, 1, 1, 13, 0), data)

        # Last 3 bars: 30, 40, 50
        assert indicator.value == pytest.approx(40.0)

    def test_hourly_sma_filters_data_up_to_current_time(self):
        data = self._hourly_data("2024-01-01 09:00", [10, 20, 30, 40, 50])
        indicator = SMAIndicator(
            period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False
        )

        indicator.update(datetime(2024, 1, 1, 11, 0), data)

        # Data up to 11:00 → [10, 20, 30]; SMA = 20
        assert indicator.value == pytest.approx(20.0)


class TestSMAIndicatorResetDaily:
    """Test cases for reset_daily functionality"""

    def _multi_day_hourly_data(self) -> pd.DataFrame:
        day1 = pd.date_range(start="2024-01-01 09:00", periods=7, freq="h")
        day2 = pd.date_range(start="2024-01-02 09:00", periods=7, freq="h")
        all_dates = day1.append(day2)

        closes = [100 + i for i in range(len(all_dates))]
        return pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [100_000] * len(all_dates),
            },
            index=all_dates,
        )

    def test_reset_daily_false_uses_bars_across_days(self):
        data = self._multi_day_hourly_data()
        indicator = SMAIndicator(
            period=3, period_unit=BarTimeInterval.HOUR, reset_daily=False
        )
        indicator.update(datetime(2024, 1, 2, 9, 0), data)

        assert indicator.value is not None

    def test_reset_daily_true_only_uses_current_day(self):
        data = self._multi_day_hourly_data()
        indicator = SMAIndicator(
            period=3, period_unit=BarTimeInterval.HOUR, reset_daily=True
        )

        # Day 2 at 11:00 → only 3 bars from day 2 (09:00, 10:00, 11:00)
        indicator.update(datetime(2024, 1, 2, 11, 0), data)

        # Day-2 closes at 09, 10, 11 are indices 7..9 → 107, 108, 109
        assert indicator.value == pytest.approx(108.0)

    def test_reset_daily_true_insufficient_bars_raises_error(self):
        data = self._multi_day_hourly_data()
        indicator = SMAIndicator(
            period=5, period_unit=BarTimeInterval.HOUR, reset_daily=True
        )

        # Day 2 at 10:00 → only 2 bars from day 2
        with pytest.raises(ValueError, match="Insufficient data.*for current day"):
            indicator.update(datetime(2024, 1, 2, 10, 0), data)

    def test_reset_daily_has_no_effect_on_daily_bars(self):
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "Close": [10, 20, 30, 40, 50],
            },
            index=dates,
        )
        indicator = SMAIndicator(
            period=3, period_unit=BarTimeInterval.DAY, column="Close", reset_daily=True
        )
        indicator.update(datetime(2024, 1, 5), data)

        assert indicator.value == pytest.approx(40.0)


class TestSMAIndicatorWeekendAdjustment:
    """Test weekend date fallback"""

    def test_weekend_date_falls_back_to_friday(self):
        dates = pd.date_range(start="2024-01-01", periods=5, freq="B")  # Mon-Fri
        data = pd.DataFrame(
            {
                "Close": [10, 20, 30, 40, 50],
            },
            index=dates,
        )
        indicator = SMAIndicator(period=3)

        # 2024-01-06 is Saturday, should fall back to 2024-01-05 (Fri)
        indicator.update(datetime(2024, 1, 6), data)

        # Last 3 business-day closes: 30, 40, 50
        assert indicator.value == pytest.approx(40.0)


class TestSMAIndicatorEdgeCases:
    """Edge cases and special scenarios"""

    def test_constant_prices_return_same_value(self):
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"Close": [100] * 5}, index=dates)
        indicator = SMAIndicator(period=5)
        indicator.update(datetime(2024, 1, 5), data)

        assert indicator.value == pytest.approx(100.0)

    def test_value_is_none_before_first_update(self):
        indicator = SMAIndicator(period=10)
        assert indicator.value is None

    def test_print_does_not_raise(self, capsys):
        indicator = SMAIndicator(period=3)
        indicator.print()
        captured = capsys.readouterr()
        assert "SMA_3" in captured.out

    def test_is_intraday_helper(self):
        assert SMAIndicator(period=5, period_unit=BarTimeInterval.DAY)._is_intraday() is False
        assert SMAIndicator(period=5, period_unit=BarTimeInterval.HOUR)._is_intraday() is True
        assert SMAIndicator(period=5, period_unit=BarTimeInterval.MINUTE)._is_intraday() is True


class TestSMAIndicatorMultiplePeriods:
    """Verify that multiple SMA instances with different periods work independently"""

    def test_different_periods_produce_different_values(self):
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        closes = [float(i) for i in range(1, 11)]  # 1..10
        data = pd.DataFrame({"Close": closes}, index=dates)

        sma_3 = SMAIndicator(period=3)
        sma_5 = SMAIndicator(period=5)
        sma_10 = SMAIndicator(period=10)

        target = datetime(2024, 1, 10)
        sma_3.update(target, data)
        sma_5.update(target, data)
        sma_10.update(target, data)

        # SMA_3 of last 3: mean(8, 9, 10) = 9.0
        assert sma_3.value == pytest.approx(9.0)
        # SMA_5 of last 5: mean(6..10) = 8.0
        assert sma_5.value == pytest.approx(8.0)
        # SMA_10 of all: mean(1..10) = 5.5
        assert sma_10.value == pytest.approx(5.5)
