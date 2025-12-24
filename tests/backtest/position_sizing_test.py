"""
Unit tests for position sizing calculations and max risk validation.

These tests verify that position sizing correctly prevents catastrophic losses
by properly calculating max risk and rejecting invalid positions.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
import pandas as pd

from src.backtest.main import BacktestEngine
from src.backtest.models import Strategy, Position, StrategyType
from src.common.models import Option, OptionType


class MockStrategy(Strategy):
    """Minimal mock strategy for testing position sizing."""
    
    def __init__(self):
        super().__init__()
        dates = pd.date_range('2024-01-01', periods=3)
        self.data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [102.0, 103.0, 104.0],
            'Low': [98.0, 99.0, 100.0],
            'Close': [100.0, 101.0, 102.0],
            'Volume': [1000000, 1100000, 1200000],
        }, index=dates)
    
    def on_new_date(self, date, positions, add_position, remove_position):
        """Mock implementation."""
        pass
    
    def on_end(self, positions, remove_position, date):
        """Mock implementation."""
        pass
    
    def validate_data(self, data):
        """Mock implementation."""
        return True


class TestPositionMaxRiskCalculation:
    """Tests for Position.get_max_risk() calculation."""
    
    def test_get_max_risk_uses_entry_price(self):
        """Test that get_max_risk uses the stored entry_price, not recalculated credit."""
        # Create options with specific prices
        atm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=3.00,  # These prices would give a different credit
            volume=100
        )
        
        otm_option = Option(
            ticker="SPY250101P00490000",
            symbol="SPY",
            strike=490.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=1.50,  # Credit from last_price would be 3.00 - 1.50 = 1.50
            volume=100
        )
        
        # Create position with different entry_price (actual credit received)
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,  # Actual credit received (different from last_price calculation)
            spread_options=[atm_option, otm_option]
        )
        
        # Calculate expected max risk
        width = 500.0 - 490.0  # 10
        expected_max_risk = (width - 2.00) * 100  # (10 - 2.00) * 100 = 800
        
        # Verify get_max_risk uses entry_price, not last_price
        assert position.get_max_risk() == expected_max_risk
        
        # Verify it's NOT using the recalculated credit
        recalculated_credit = atm_option.last_price - otm_option.last_price  # 1.50
        wrong_max_risk = (width - recalculated_credit) * 100  # 850
        assert position.get_max_risk() != wrong_max_risk
    
    def test_get_max_risk_with_wide_spread(self):
        """Test max risk calculation with a wide spread and small credit."""
        atm_option = Option(
            ticker="SPY250401P00556000",
            symbol="SPY",
            strike=556.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=50.00,
            volume=10
        )
        
        otm_option = Option(
            ticker="SPY250401P00508000",
            symbol="SPY",
            strike=508.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=47.39,
            volume=10
        )
        
        # This simulates the catastrophic April trade
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 4, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=556.0,
            entry_date=datetime(2025, 4, 1),
            entry_price=2.61,  # Small credit for wide spread
            spread_options=[atm_option, otm_option]
        )
        
        # Max risk should be huge
        width = 556.0 - 508.0  # 48
        expected_max_risk = (width - 2.61) * 100  # (48 - 2.61) * 100 = 4,539
        
        assert position.get_max_risk() == expected_max_risk
        assert position.get_max_risk() > 4500  # Very high risk
    
    def test_get_max_risk_requires_spread_options(self):
        """Test that get_max_risk raises error when spread_options are missing."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,
            spread_options=[]  # Empty spread_options
        )
        
        with pytest.raises(ValueError, match="Credit spread requires at least 2 options"):
            position.get_max_risk()
    
    def test_get_max_risk_with_one_option(self):
        """Test that get_max_risk raises error with only one option."""
        option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=100
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,
            spread_options=[option]  # Only one option
        )
        
        with pytest.raises(ValueError, match="Credit spread requires at least 2 options"):
            position.get_max_risk()


class TestPositionSizing:
    """Tests for BacktestEngine._get_position_size() calculation."""
    
    def test_normal_position_sizing(self):
        """Test normal position sizing with sufficient capital."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.15  # 15% of capital
        )
        
        # Create a normal position
        atm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=100
        )
        
        otm_option = Option(
            ticker="SPY250101P00490000",
            symbol="SPY",
            strike=490.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=1.00,
            volume=100
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,  # $2.00 credit
            spread_options=[atm_option, otm_option]
        )
        
        # Calculate expected position size
        # Max risk = (10 - 2.00) * 100 = $800 per contract
        # Max position capital = $10,000 * 0.15 = $1,500
        # Position size = $1,500 / $800 = 1.875 = 1 contract (int)
        
        position_size = engine._get_position_size(position)
        assert position_size == 1
    
    def test_position_sizing_prevents_oversized_positions(self):
        """Test that position sizing prevents positions larger than capital."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=3000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.15  # 15% of capital = $450
        )
        
        # Create a position with very high risk (like the April catastrophic trade)
        atm_option = Option(
            ticker="SPY250401P00556000",
            symbol="SPY",
            strike=556.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=50.00,
            volume=10
        )
        
        otm_option = Option(
            ticker="SPY250401P00508000",
            symbol="SPY",
            strike=508.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=47.39,
            volume=10
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 4, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=556.0,
            entry_date=datetime(2025, 4, 1),
            entry_price=2.61,  # Small credit for wide spread
            spread_options=[atm_option, otm_option]
        )
        
        # Max risk = (48 - 2.61) * 100 = $4,539 per contract
        # Max position capital = $3,000 * 0.15 = $450
        # Position size = $450 / $4,539 = 0.099 = 0 contracts
        
        position_size = engine._get_position_size(position)
        assert position_size == 0  # Should reject this position
    
    def test_position_sizing_with_max_risk_exceeding_capital(self):
        """Test position sizing when max_risk per contract exceeds total capital."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=1000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.50  # 50% of capital = $500
        )
        
        # Create a position where max_risk exceeds capital
        atm_option = Option(
            ticker="SPY250101P00520000",
            symbol="SPY",
            strike=520.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=15.00,
            volume=100
        )
        
        otm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=13.00,
            volume=100
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=520.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,  # $2 credit
            spread_options=[atm_option, otm_option]
        )
        
        # Max risk = (20 - 2) * 100 = $1,800 per contract
        # This exceeds total capital of $1,000
        
        position_size = engine._get_position_size(position)
        assert position_size == 0  # Should reject due to safety check
    
    def test_position_sizing_with_no_max_position_size(self):
        """Test that position sizing returns 1 when max_position_size is None."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=None  # No position sizing
        )
        
        atm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=100
        )
        
        otm_option = Option(
            ticker="SPY250101P00490000",
            symbol="SPY",
            strike=490.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=1.00,
            volume=100
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,
            spread_options=[atm_option, otm_option]
        )
        
        position_size = engine._get_position_size(position)
        assert position_size == 1
    
    def test_position_sizing_with_invalid_position(self):
        """Test position sizing handles invalid positions gracefully."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.15
        )
        
        # Create position with missing spread_options
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=2.00,
            spread_options=[]  # Invalid: empty spread_options
        )
        
        # Should return 0 and handle the error gracefully
        position_size = engine._get_position_size(position)
        assert position_size == 0
    
    def test_position_sizing_multiple_contracts(self):
        """Test position sizing with sufficient capital for multiple contracts."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=100000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.20  # 20% = $20,000
        )
        
        # Create a low-risk position
        atm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=1000
        )
        
        otm_option = Option(
            ticker="SPY250101P00495000",
            symbol="SPY",
            strike=495.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=1.00,
            volume=1000
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=1.50,  # $1.50 credit
            spread_options=[atm_option, otm_option]
        )
        
        # Max risk = (5 - 1.50) * 100 = $350 per contract
        # Max position capital = $100,000 * 0.20 = $20,000
        # Position size = $20,000 / $350 = 57.14 = 57 contracts
        
        position_size = engine._get_position_size(position)
        assert position_size == 57


class TestCatastrophicLossPrevention:
    """Tests verifying the fix prevents catastrophic losses."""
    
    def test_april_catastrophic_trade_prevented(self):
        """Test that the catastrophic April 2025 trade would now be prevented."""
        strategy = MockStrategy()
        
        # Simulate the exact conditions before the catastrophic trade
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=2559.0,  # Capital before the trade
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.15  # 15% = $383.85
        )
        
        # Recreate the exact catastrophic position
        # This is a 48-point spread (556 - 508) with only $2.61 credit
        atm_option = Option(
            ticker="SPY250401P00556000",
            symbol="SPY",
            strike=556.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=50.00,
            volume=10
        )
        
        otm_option = Option(
            ticker="SPY250401P00508000",
            symbol="SPY",
            strike=508.0,
            expiration="2025-04-01",
            option_type=OptionType.PUT,
            last_price=47.39,
            volume=10
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 4, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=556.0,
            entry_date=datetime(2025, 4, 1),
            entry_price=2.61,
            spread_options=[atm_option, otm_option]
        )
        
        # Verify spread width is 48 points
        spread_width = abs(atm_option.strike - otm_option.strike)
        assert spread_width == 48.0, f"Expected 48-point spread, got {spread_width}"
        
        # Verify max_risk is calculated correctly using entry_price
        expected_max_risk = (spread_width - 2.61) * 100  # (48 - 2.61) * 100 = $4,539
        actual_max_risk = position.get_max_risk()
        assert abs(actual_max_risk - expected_max_risk) < 0.01, \
            f"Expected max_risk ${expected_max_risk:.2f}, got ${actual_max_risk:.2f}"
        
        # Verify the max_risk is huge compared to capital
        assert actual_max_risk > engine.capital, \
            f"Max risk ${actual_max_risk:.2f} should exceed capital ${engine.capital:.2f}"
        
        # Verify position is rejected due to insufficient capital
        # With capital=$2,559, max_position_size=15%, we have $383.85 to risk
        # But max_risk=$4,539, so position_size = int($383.85 / $4,539) = 0
        position_size = engine._get_position_size(position)
        assert position_size == 0, \
            f"Catastrophic trade should be prevented (position_size should be 0, got {position_size})"
    
    def test_negative_capital_impossible(self):
        """Test that positions cannot drive capital negative."""
        strategy = MockStrategy()
        engine = BacktestEngine(
            data=strategy.data,
            strategy=strategy,
            initial_capital=5000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            max_position_size=0.15
        )
        
        # Create a very risky position
        atm_option = Option(
            ticker="SPY250101P00600000",
            symbol="SPY",
            strike=600.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=100.00,
            volume=10
        )
        
        otm_option = Option(
            ticker="SPY250101P00500000",
            symbol="SPY",
            strike=500.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=95.00,
            volume=10
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=600.0,
            entry_date=datetime(2024, 1, 1),
            entry_price=5.00,  # $5 credit for $100 wide spread
            spread_options=[atm_option, otm_option]
        )
        
        # Max risk = (100 - 5) * 100 = $9,500 per contract
        # This exceeds total capital, so should be rejected
        
        position_size = engine._get_position_size(position)
        assert position_size == 0, "Position with max_risk > capital should be rejected"

