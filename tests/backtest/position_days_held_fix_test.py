#!/usr/bin/env python3
"""
Unit tests for the enhanced Position.get_days_held method.
Tests the fix for date-only comparison in trading day calculations.
"""

import unittest
from datetime import datetime, timezone
from algo_trading_engine.backtest.models import Position, StrategyType
from algo_trading_engine.common.models import Option, OptionType


class TestPositionDaysHeldFix(unittest.TestCase):
    """Test cases for the enhanced get_days_held method."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test position
        self.test_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 9, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=666.0,
            entry_date=datetime(2025, 9, 22, 17, 6, 46),  # 5:06 PM on 2025-09-22
            entry_price=2.35,
            spread_options=[
                Option(
                    ticker="O:SPY250930P00666000",
                    symbol="SPY",
                    strike=666.0,
                    expiration="2025-09-30",
                    option_type=OptionType.PUT,
                    last_price=5.18
                ),
                Option(
                    ticker="O:SPY250930P00657000",
                    symbol="SPY",
                    strike=657.0,
                    expiration="2025-09-30",
                    option_type=OptionType.PUT,
                    last_price=1.93
                )
            ]
        )

    def test_days_held_same_day_entry(self):
        """Test that same day entry shows 0 days held."""
        # Same day as entry
        same_day = datetime(2025, 9, 22, 20, 0, 0)  # 8:00 PM same day
        days_held = self.test_position.get_days_held(same_day)
        self.assertEqual(days_held, 0)

    def test_days_held_next_day(self):
        """Test that next day shows 1 day held."""
        # Next day
        next_day = datetime(2025, 9, 23, 9, 30, 0)  # 9:30 AM next day
        days_held = self.test_position.get_days_held(next_day)
        self.assertEqual(days_held, 1)

    def test_days_held_multiple_days(self):
        """Test that multiple days are calculated correctly."""
        # 3 days later
        three_days_later = datetime(2025, 9, 25, 15, 0, 0)  # 3:00 PM, 3 days later
        days_held = self.test_position.get_days_held(three_days_later)
        self.assertEqual(days_held, 3)

    def test_days_held_with_timezone_aware_entry(self):
        """Test that timezone-aware entry dates work correctly."""
        # Create position with timezone-aware entry date
        timezone_aware_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 9, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=666.0,
            entry_date=datetime(2025, 9, 22, 17, 6, 46, tzinfo=timezone.utc),  # UTC timezone
            entry_price=2.35,
            spread_options=self.test_position.spread_options
        )
        
        # Check next day (timezone-naive)
        next_day = datetime(2025, 9, 23, 9, 30, 0)
        days_held = timezone_aware_position.get_days_held(next_day)
        self.assertEqual(days_held, 1)

    def test_days_held_with_timezone_aware_current(self):
        """Test that timezone-aware current dates work correctly."""
        # Check with timezone-aware current date
        timezone_aware_current = datetime(2025, 9, 23, 9, 30, 0, tzinfo=timezone.utc)
        days_held = self.test_position.get_days_held(timezone_aware_current)
        self.assertEqual(days_held, 1)

    def test_days_held_both_timezone_aware(self):
        """Test that both timezone-aware entry and current dates work correctly."""
        # Create position with timezone-aware entry date
        timezone_aware_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 9, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=666.0,
            entry_date=datetime(2025, 9, 22, 17, 6, 46, tzinfo=timezone.utc),
            entry_price=2.35,
            spread_options=self.test_position.spread_options
        )
        
        # Check with timezone-aware current date
        timezone_aware_current = datetime(2025, 9, 23, 9, 30, 0, tzinfo=timezone.utc)
        days_held = timezone_aware_position.get_days_held(timezone_aware_current)
        self.assertEqual(days_held, 1)

    def test_days_held_negative_result(self):
        """Test that negative results are handled correctly."""
        # Check date before entry (should be negative)
        before_entry = datetime(2025, 9, 21, 15, 0, 0)  # Day before entry
        days_held = self.test_position.get_days_held(before_entry)
        self.assertEqual(days_held, -1)

    def test_days_held_midnight_boundary(self):
        """Test that midnight boundaries work correctly."""
        # Just before midnight on entry day
        just_before_midnight = datetime(2025, 9, 22, 23, 59, 59)
        days_held = self.test_position.get_days_held(just_before_midnight)
        self.assertEqual(days_held, 0)
        
        # Just after midnight on next day
        just_after_midnight = datetime(2025, 9, 23, 0, 0, 1)
        days_held = self.test_position.get_days_held(just_after_midnight)
        self.assertEqual(days_held, 1)

    def test_days_held_weekend_crossing(self):
        """Test that weekend crossings are handled correctly."""
        # Entry on Friday, check on Monday
        friday_entry = datetime(2025, 9, 19, 17, 6, 46)  # Friday 5:06 PM
        monday_check = datetime(2025, 9, 22, 9, 30, 0)  # Monday 9:30 AM
        
        weekend_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 9, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=666.0,
            entry_date=friday_entry,
            entry_price=2.35,
            spread_options=self.test_position.spread_options
        )
        
        days_held = weekend_position.get_days_held(monday_check)
        self.assertEqual(days_held, 3)  # Friday to Monday = 3 calendar days

    def test_days_held_no_entry_date_raises_error(self):
        """Test that missing entry date raises ValueError."""
        # Create position without entry date
        no_entry_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 9, 30),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=666.0,
            entry_date=None,  # No entry date
            entry_price=2.35,
            spread_options=self.test_position.spread_options
        )
        
        with self.assertRaises(ValueError) as context:
            no_entry_position.get_days_held(datetime(2025, 9, 23))
        
        self.assertIn("Entry date is not set", str(context.exception))

    def test_days_held_large_time_difference(self):
        """Test that large time differences work correctly."""
        # 30 days later
        thirty_days_later = datetime(2025, 10, 22, 15, 0, 0)
        days_held = self.test_position.get_days_held(thirty_days_later)
        self.assertEqual(days_held, 30)

    def test_days_held_edge_case_same_date_different_times(self):
        """Test edge case where same date but different times."""
        # Same date, different time (should be 0 days)
        same_date_different_time = datetime(2025, 9, 22, 23, 0, 0)  # 11:00 PM same day
        days_held = self.test_position.get_days_held(same_date_different_time)
        self.assertEqual(days_held, 0)


if __name__ == '__main__':
    unittest.main()

