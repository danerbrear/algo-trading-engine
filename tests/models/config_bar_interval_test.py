"""
Unit tests for BacktestConfig and PaperTradingConfig bar_interval support.
"""
import pytest
from datetime import datetime, timedelta

from algo_trading_engine.models.config import BacktestConfig, PaperTradingConfig
from algo_trading_engine.enums import BarTimeInterval


class TestBacktestConfigBarInterval:
    """Test cases for BacktestConfig bar_interval parameter."""
    
    def test_default_bar_interval_is_daily(self):
        """Test that bar_interval defaults to DAY."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol='SPY',
            strategy_type='velocity_signal_momentum'
        )
        
        assert config.bar_interval == BarTimeInterval.DAY
    
    def test_hourly_bar_interval(self):
        """Test creating config with hourly bars."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.HOUR
        )
        
        assert config.bar_interval == BarTimeInterval.HOUR
    
    def test_minute_bar_interval(self):
        """Test creating config with minute bars."""
        config = BacktestConfig(
            initial_capital=100000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.MINUTE
        )
        
        assert config.bar_interval == BarTimeInterval.MINUTE
    
    def test_hourly_bars_within_limit(self):
        """Test hourly bars with date range within 729 day limit."""
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=729)  # Exactly at limit
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=start_date,
            end_date=end_date,
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.HOUR
        )
        
        assert config.bar_interval == BarTimeInterval.HOUR
    
    def test_hourly_bars_exceeds_limit_raises_error(self):
        """Test hourly bars with date range exceeding 729 days raises ValueError."""
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=730)  # Exceeds limit
        
        with pytest.raises(ValueError, match="Hourly bars are limited to 729 days"):
            BacktestConfig(
                initial_capital=100000,
                start_date=start_date,
                end_date=end_date,
                symbol='SPY',
                strategy_type='velocity_signal_momentum',
                bar_interval=BarTimeInterval.HOUR
            )
    
    def test_minute_bars_within_limit(self):
        """Test minute bars with date range within 59 day limit."""
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=59)  # Exactly at limit
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=start_date,
            end_date=end_date,
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.MINUTE
        )
        
        assert config.bar_interval == BarTimeInterval.MINUTE
    
    def test_minute_bars_exceeds_limit_raises_error(self):
        """Test minute bars with date range exceeding 59 days raises ValueError."""
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=60)  # Exceeds limit
        
        with pytest.raises(ValueError, match="Minute bars are limited to 59 days"):
            BacktestConfig(
                initial_capital=100000,
                start_date=start_date,
                end_date=end_date,
                symbol='SPY',
                strategy_type='velocity_signal_momentum',
                bar_interval=BarTimeInterval.MINUTE
            )
    
    def test_daily_bars_no_date_range_limit(self):
        """Test daily bars have no date range limit."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)  # 5 years - no limit
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=start_date,
            end_date=end_date,
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.DAY
        )
        
        assert config.bar_interval == BarTimeInterval.DAY
    
    def test_error_message_includes_date_range(self):
        """Test error message includes the requested date range."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 1)  # 60 days, exceeds minute limit
        
        with pytest.raises(ValueError) as exc_info:
            BacktestConfig(
                initial_capital=100000,
                start_date=start_date,
                end_date=end_date,
                symbol='SPY',
                strategy_type='velocity_signal_momentum',
                bar_interval=BarTimeInterval.MINUTE
            )
        
        error_message = str(exc_info.value)
        assert "60 days" in error_message
        assert "2024-01-01" in error_message
        assert "2024-03-01" in error_message


class TestPaperTradingConfigBarInterval:
    """Test cases for PaperTradingConfig bar_interval parameter."""
    
    def test_default_bar_interval_is_daily(self):
        """Test that bar_interval defaults to DAY."""
        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='velocity_signal_momentum'
        )
        
        assert config.bar_interval == BarTimeInterval.DAY
    
    def test_hourly_bar_interval(self):
        """Test creating config with hourly bars."""
        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.HOUR
        )
        
        assert config.bar_interval == BarTimeInterval.HOUR
    
    def test_minute_bar_interval(self):
        """Test creating config with minute bars."""
        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.MINUTE
        )
        
        assert config.bar_interval == BarTimeInterval.MINUTE
    
    def test_daily_bar_interval(self):
        """Test creating config with daily bars explicitly."""
        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.DAY
        )
        
        assert config.bar_interval == BarTimeInterval.DAY
    
    def test_no_date_range_validation_for_paper_trading(self):
        """Test PaperTradingConfig doesn't validate date ranges (no start/end dates)."""
        # This should not raise any errors regardless of bar_interval
        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='velocity_signal_momentum',
            bar_interval=BarTimeInterval.MINUTE  # Would have date limit in BacktestConfig
        )
        
        assert config.bar_interval == BarTimeInterval.MINUTE
