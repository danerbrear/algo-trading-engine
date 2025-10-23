"""
Tests for Upward Trend Reversal Strategy.

Following TDD principles: write tests first, then implement to pass them.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.backtest.models import StrategyType, Position
from src.common.models import Option, OptionType, OptionChain
from src.strategies.upward_trend_reversal_strategy import (
    UpwardTrendReversalStrategy,
    TrendInfo
)


class TestTrendDetection:
    """Test trend detection logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock options handler
        self.options_handler = MagicMock()
        self.options_handler.symbol = 'SPY'
        
        # Create strategy instance
        self.strategy = UpwardTrendReversalStrategy(
            options_handler=self.options_handler,
            min_trend_duration=3,
            max_trend_duration=4,
            start_date_offset=60
        )
    
    def test_detect_upward_trend_3_days(self):
        """Test detection of a 3-day upward trend."""
        # Create test data with a 3-day upward trend
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 102]  # 3-day up, then down
        data = pd.DataFrame({'Close': prices + [101, 100, 99, 98, 97]}, index=dates)
        
        self.strategy.set_data(data, None)
        
        # Detect trends
        trends = self.strategy._detect_upward_trends(data)
        
        # Should detect one 3-day upward trend
        # Note: trend detection starts from day 2 (index 1) because we need previous day for returns
        assert len(trends) >= 1
        trend = trends[0]
        assert trend.duration == 3
        assert trend.start_price == 101  # Starts from day 2
        assert trend.end_price == 103
        assert trend.net_return > 0
    
    def test_detect_upward_trend_4_days(self):
        """Test detection of a 4-day upward trend."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 104, 103]  # 4-day up, then down
        data = pd.DataFrame({'Close': prices + [102, 101, 100, 99]}, index=dates)
        
        self.strategy.set_data(data, None)
        trends = self.strategy._detect_upward_trends(data)
        
        # Should detect one 4-day upward trend
        assert len(trends) >= 1
        trend = trends[0]
        assert trend.duration == 4
        assert trend.start_price == 101  # Starts from day 2
        assert trend.end_price == 104
    
    def test_ignore_trend_too_short(self):
        """Test that trends < 3 days are ignored."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 101]  # Only 2 days up
        data = pd.DataFrame({'Close': prices + [100, 99, 98, 97, 96, 95]}, index=dates)
        
        self.strategy.set_data(data, None)
        trends = self.strategy._detect_upward_trends(data)
        
        # Should not detect trends < 3 days
        if len(trends) > 0:
            assert all(t.duration >= 3 for t in trends)
    
    def test_ignore_trend_too_long(self):
        """Test that trends > 4 days are ignored."""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 105]  # 6 days up
        data = pd.DataFrame({'Close': prices + [104, 103, 102, 101, 100, 99, 98]}, index=dates)
        
        self.strategy.set_data(data, None)
        trends = self.strategy._detect_upward_trends(data)
        
        # Should not detect trends > 4 days (with default settings)
        if len(trends) > 0:
            assert all(t.duration <= 4 for t in trends)
    
    def test_calculate_reversal_drawdown(self):
        """Test calculation of reversal drawdown."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 101]  # Up 3 days, then down to 101
        data = pd.DataFrame({'Close': prices + [100, 99, 98, 97, 96]}, index=dates)
        
        self.strategy.set_data(data, None)
        trends = self.strategy._detect_upward_trends(data)
        
        if len(trends) > 0:
            trend = trends[0]
            # Drawdown = (101 - 103) / 103 = -1.94%
            expected_drawdown = (101 - 103) / 103
            assert abs(trend.reversal_drawdown - expected_drawdown) < 0.001


class TestMarketRegimeFiltering:
    """Test market regime filtering logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.options_handler = MagicMock()
        self.options_handler.symbol = 'SPY'
        
        self.strategy = UpwardTrendReversalStrategy(
            options_handler=self.options_handler,
            start_date_offset=60
        )
    
    def test_filter_momentum_uptrend_regime(self):
        """Test that positions are not opened during momentum uptrend regime."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({'Close': np.linspace(100, 120, 100)}, index=dates)
        
        # Add market state column (1 = MOMENTUM_UPTREND)
        data['Market_State'] = 1
        
        self.strategy.set_data(data, None)
        
        # Check if momentum uptrend regime
        test_date = dates[50]
        is_momentum = self.strategy._is_momentum_uptrend_regime(test_date)
        
        assert is_momentum == True
    
    def test_allow_other_regimes(self):
        """Test that positions can be opened in non-momentum regimes."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({'Close': np.linspace(100, 120, 100)}, index=dates)
        
        # Add market state column (0 = LOW_VOLATILITY_UPTREND)
        data['Market_State'] = 0
        
        self.strategy.set_data(data, None)
        
        test_date = dates[50]
        is_momentum = self.strategy._is_momentum_uptrend_regime(test_date)
        
        assert is_momentum == False


class TestPutDebitSpreadSelection:
    """Test put debit spread selection logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.options_handler = MagicMock()
        self.options_handler.symbol = 'SPY'
        
        self.strategy = UpwardTrendReversalStrategy(
            options_handler=self.options_handler,
            max_spread_width=6.0,
            min_dte=5,
            max_dte=10,
            start_date_offset=60
        )
    
    def test_find_put_debit_spread_valid(self):
        """Test finding a valid put debit spread."""
        # This test validates that the function doesn't crash with proper DTO objects
        # Full integration testing will be done in backtest
        current_price = 580.0
        test_date = datetime(2024, 12, 1)
        
        # Mock to return empty list (no options available)
        self.options_handler.get_contract_list_for_date.return_value = []
        
        # Find spread
        spread_info = self.strategy._find_put_debit_spread(test_date, current_price)
        
        # Should return None when no options available
        assert spread_info is None
    
    def test_reject_spread_too_wide(self):
        """Test that spreads wider than max_spread_width are rejected."""
        current_price = 580.0
        test_date = datetime(2024, 12, 1)
        
        # Create options with width > 6.0
        atm_put = Option(
            ticker='O:SPY241210P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=8.00,
            volume=1000
        )
        
        otm_put = Option(
            ticker='O:SPY241210P00573000',
            symbol='SPY',
            strike=573.0,  # Width = 7.0 (too wide)
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=1000
        )
        
        self.options_handler.get_contract_list_for_date.return_value = [atm_put, otm_put]
        
        spread_info = self.strategy._find_put_debit_spread(test_date, current_price)
        
        # Should not find a spread (too wide)
        # Implementation will need to handle this
        assert spread_info is None or spread_info['width'] <= 6.0


class TestPositionManagement:
    """Test position sizing and management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.options_handler = MagicMock()
        self.options_handler.symbol = 'SPY'
        
        self.strategy = UpwardTrendReversalStrategy(
            options_handler=self.options_handler,
            max_risk_per_trade=0.20,
            start_date_offset=60
        )
    
    def test_calculate_position_size(self):
        """Test position size calculation based on max risk."""
        # Create a mock position
        atm_put = Option(
            ticker='O:SPY241210P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=5.00,
            volume=1000
        )
        
        otm_put = Option(
            ticker='O:SPY241210P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=1000
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 12, 10),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=580.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.0,  # Net debit
            spread_options=[atm_put, otm_put]
        )
        
        # Calculate position size with $10,000 capital
        capital = 10000
        max_risk_pct = 0.20
        
        # Max risk = net debit = 2.0 * 100 = $200 per contract
        # Max risk dollars = 10000 * 0.20 = $2000
        # Quantity = 2000 / 200 = 10 contracts
        
        quantity = self.strategy._calculate_position_size(position, max_risk_pct, capital)
        
        assert quantity == 10
    
    def test_minimum_one_contract(self):
        """Test that at least 1 contract is traded."""
        atm_put = Option(
            ticker='O:SPY241210P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=50.00,  # Very expensive
            volume=1000
        )
        
        otm_put = Option(
            ticker='O:SPY241210P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-10',
            option_type=OptionType.PUT,
            last_price=45.00,
            volume=1000
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 12, 10),
            strategy_type=StrategyType.PUT_DEBIT_SPREAD,
            strike_price=580.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=5.0,  # Net debit = 5.0
            spread_options=[atm_put, otm_put]
        )
        
        # Small capital
        capital = 1000
        max_risk_pct = 0.20
        
        # Max risk = 500 * 100 = $500 per contract
        # Max risk dollars = 1000 * 0.20 = $200
        # Would be 0.4 contracts, but should round to at least 1
        
        quantity = self.strategy._calculate_position_size(position, max_risk_pct, capital)
        
        assert quantity >= 1


class TestStrategyIntegration:
    """Test full strategy integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.options_handler = MagicMock()
        self.options_handler.symbol = 'SPY'
        
        self.strategy = UpwardTrendReversalStrategy(
            options_handler=self.options_handler,
            start_date_offset=60
        )
    
    def test_validate_data_with_required_columns(self):
        """Test data validation with all required columns."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.linspace(100, 120, 100),
            'Market_State': np.random.randint(0, 5, 100)
        }, index=dates)
        
        self.strategy.set_data(data, None)
        
        is_valid = self.strategy.validate_data(data)
        assert is_valid is True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing Close column."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.linspace(100, 120, 100)
            # Missing Close (required)
        }, index=dates)
        
        is_valid = self.strategy.validate_data(data)
        assert is_valid == False
    
    def test_on_new_date_no_open_positions(self):
        """Test on_new_date when no positions should be opened."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.linspace(100, 120, 100),
            'Market_State': np.ones(100)  # All momentum uptrend
        }, index=dates)
        
        self.strategy.set_data(data, None)
        
        # Mock functions
        add_position = MagicMock()
        remove_position = MagicMock()
        
        # Should not add position during momentum uptrend
        self.strategy.on_new_date(dates[70], tuple(), add_position, remove_position)
        
        # Verify no position was added
        add_position.assert_not_called()

