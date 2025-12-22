#!/usr/bin/env python3
"""
Unit tests for the enhanced Position.get_days_held method.
Tests the fix for date-only comparison in trading day calculations.
"""

import unittest
from datetime import datetime, timezone
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


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


class TestLongPutReturnCalculations(unittest.TestCase):
    """Test cases for LONG_PUT return calculations."""
    
    def setUp(self):
        """Set up test fixtures for long put positions."""
        # Create a simple long put option
        self.put_option = Option(
            ticker="O:SPY250315P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-03-15",
            option_type=OptionType.PUT,
            last_price=3.0
        )
        
        # Create long put position
        self.long_put_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 3, 15),
            strategy_type=StrategyType.LONG_PUT,
            strike_price=500.0,
            entry_date=datetime(2025, 3, 1),
            entry_price=3.0,  # Buy put at $3.00
            spread_options=[self.put_option]
        )
        self.long_put_position.set_quantity(2)  # 2 contracts
    
    def test_long_put_return_calculation_profit(self):
        """Test that LONG_PUT returns are calculated correctly for profitable trades."""
        # Buy put at $3.00, sell at $5.00 = 67% profit
        exit_price = 5.0
        return_pct = self.long_put_position._get_return(exit_price)
        
        # Expected: (5.00 - 3.00) / 3.00 = 0.6667 (67%)
        expected_return = (5.0 - 3.0) / 3.0
        self.assertAlmostEqual(return_pct, expected_return, places=4)
        self.assertGreater(return_pct, 0, "Profitable long put should have positive return")
    
    def test_long_put_return_calculation_loss(self):
        """Test that LONG_PUT returns are calculated correctly for losing trades."""
        # Buy put at $3.00, sell at $1.00 = -67% loss
        exit_price = 1.0
        return_pct = self.long_put_position._get_return(exit_price)
        
        # Expected: (1.00 - 3.00) / 3.00 = -0.6667 (-67%)
        expected_return = (1.0 - 3.0) / 3.0
        self.assertAlmostEqual(return_pct, expected_return, places=4)
        self.assertLess(return_pct, 0, "Losing long put should have negative return")
    
    def test_long_put_return_calculation_breakeven(self):
        """Test that LONG_PUT returns are calculated correctly at breakeven."""
        # Buy put at $3.00, sell at $3.00 = 0% return
        exit_price = 3.0
        return_pct = self.long_put_position._get_return(exit_price)
        
        # Expected: (3.00 - 3.00) / 3.00 = 0.0
        self.assertAlmostEqual(return_pct, 0.0, places=4)
    
    def test_long_put_profit_target_hit_when_above_target(self):
        """Test that profit_target_hit returns True when target is exceeded."""
        profit_target = 0.5  # 50% profit target
        exit_price = 4.5  # (4.5 - 3.0) / 3.0 = 50%
        
        hit = self.long_put_position.profit_target_hit(profit_target, exit_price)
        self.assertTrue(hit, "Should hit profit target when return equals target")
    
    def test_long_put_profit_target_not_hit(self):
        """Test that profit_target_hit returns False when target is not reached."""
        profit_target = 0.5  # 50% profit target
        exit_price = 4.0  # (4.0 - 3.0) / 3.0 = 33%
        
        hit = self.long_put_position.profit_target_hit(profit_target, exit_price)
        self.assertFalse(hit, "Should not hit profit target when return is below target")
    
    def test_long_put_stop_loss_hit_when_below_stop(self):
        """Test that stop_loss_hit returns True when stop loss is exceeded."""
        stop_loss = 0.5  # 50% stop loss
        exit_price = 1.5  # (1.5 - 3.0) / 3.0 = -50%
        
        hit = self.long_put_position.stop_loss_hit(stop_loss, exit_price)
        self.assertTrue(hit, "Should hit stop loss when loss equals stop")
    
    def test_long_put_stop_loss_not_hit(self):
        """Test that stop_loss_hit returns False when stop loss is not exceeded."""
        stop_loss = 0.5  # 50% stop loss
        exit_price = 2.0  # (2.0 - 3.0) / 3.0 = -33%
        
        hit = self.long_put_position.stop_loss_hit(stop_loss, exit_price)
        self.assertFalse(hit, "Should not hit stop loss when loss is less than stop")
    
    def test_long_put_get_return_dollars_profit(self):
        """Test dollar return calculation for profitable long put."""
        exit_price = 5.0  # Buy at $3, sell at $5
        return_dollars = self.long_put_position.get_return_dollars(exit_price)
        
        # Expected: (5.0 - 3.0) * 2 contracts * 100 = $400
        expected_dollars = (5.0 - 3.0) * 2 * 100
        self.assertAlmostEqual(return_dollars, expected_dollars, places=2)
        self.assertEqual(return_dollars, 400.0)
    
    def test_long_put_get_return_dollars_loss(self):
        """Test dollar return calculation for losing long put."""
        exit_price = 1.0  # Buy at $3, sell at $1
        return_dollars = self.long_put_position.get_return_dollars(exit_price)
        
        # Expected: (1.0 - 3.0) * 2 contracts * 100 = -$400
        expected_dollars = (1.0 - 3.0) * 2 * 100
        self.assertAlmostEqual(return_dollars, expected_dollars, places=2)
        self.assertEqual(return_dollars, -400.0)
    
    def test_long_put_total_loss_scenario(self):
        """Test long put when option expires worthless (100% loss)."""
        exit_price = 0.0  # Option expires worthless
        return_pct = self.long_put_position._get_return(exit_price)
        
        # Expected: (0.0 - 3.0) / 3.0 = -1.0 (-100%)
        self.assertAlmostEqual(return_pct, -1.0, places=4)
        
        return_dollars = self.long_put_position.get_return_dollars(exit_price)
        # Expected: (0.0 - 3.0) * 2 * 100 = -$600 (total premium paid)
        self.assertEqual(return_dollars, -600.0)
    
    def test_long_put_large_profit_scenario(self):
        """Test long put with large profit (e.g., stock crashes)."""
        exit_price = 15.0  # Put gains significant value
        return_pct = self.long_put_position._get_return(exit_price)
        
        # Expected: (15.0 - 3.0) / 3.0 = 4.0 (400% profit)
        expected_return = (15.0 - 3.0) / 3.0
        self.assertAlmostEqual(return_pct, expected_return, places=4)
        self.assertGreater(return_pct, 3.0, "Large profit should exceed 300%")
    
    def test_long_put_vs_short_put_return_difference(self):
        """Test that long put and short put have opposite return calculations."""
        # Create a short put with same parameters
        short_put_position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 3, 15),
            strategy_type=StrategyType.SHORT_PUT,
            strike_price=500.0,
            entry_date=datetime(2025, 3, 1),
            entry_price=3.0,  # Sell put at $3.00
            spread_options=[self.put_option]
        )
        short_put_position.set_quantity(2)
        
        exit_price = 5.0
        
        # Long put: buy at $3, sell at $5 = profit
        long_return = self.long_put_position._get_return(exit_price)
        # Short put: sell at $3, buy back at $5 = loss
        short_return = short_put_position._get_return(exit_price)
        
        # They should have opposite signs
        self.assertGreater(long_return, 0, "Long put should profit when price increases")
        self.assertLess(short_return, 0, "Short put should lose when price increases")


if __name__ == '__main__':
    unittest.main()

