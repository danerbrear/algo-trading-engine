"""
Unit tests for Position max_profit, max_loss, and risk_reward methods.

Tests the new methods added to the Position class for calculating
maximum profit, maximum loss, risk/reward ratio, and expected value.
"""

import pytest
from datetime import datetime
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestPositionMaxProfitLoss:
    """Test suite for Position max profit/loss calculations."""
    
    @pytest.fixture
    def call_option_100(self):
        """Create a call option at strike 100."""
        return Option(
            ticker="O:SPY250101C00100000",
            symbol="SPY",
            strike=100.0,
            expiration="2025-01-01",
            option_type=OptionType.CALL,
            last_price=3.00
        )
    
    @pytest.fixture
    def call_option_105(self):
        """Create a call option at strike 105."""
        return Option(
            ticker="O:SPY250101C00105000",
            symbol="SPY",
            strike=105.0,
            expiration="2025-01-01",
            option_type=OptionType.CALL,
            last_price=0.50
        )
    
    @pytest.fixture
    def put_option_100(self):
        """Create a put option at strike 100."""
        return Option(
            ticker="O:SPY250101P00100000",
            symbol="SPY",
            strike=100.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=3.00
        )
    
    @pytest.fixture
    def put_option_95(self):
        """Create a put option at strike 95."""
        return Option(
            ticker="O:SPY250101P00095000",
            symbol="SPY",
            strike=95.0,
            expiration="2025-01-01",
            option_type=OptionType.PUT,
            last_price=0.50
        )
    
    # ==================== Credit Spread Tests ====================
    
    def test_call_credit_spread_max_profit(self, call_option_100, call_option_105):
        """Test max profit for call credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,  # Net credit
            spread_options=[call_option_100, call_option_105]
        )
        
        assert position.max_profit() == 2.50
    
    def test_call_credit_spread_max_loss(self, call_option_100, call_option_105):
        """Test max loss for call credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        # Spread width = 5, credit = 2.50, max loss = 5 - 2.50 = 2.50
        assert position.max_loss() == 2.50
    
    def test_call_credit_spread_risk_reward(self, call_option_100, call_option_105):
        """Test risk/reward ratio for call credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        # Risk/reward = 2.50 / 2.50 = 1.0
        assert position.risk_reward_ratio() == 1.0
    
    def test_put_credit_spread_max_profit(self, put_option_100, put_option_95):
        """Test max profit for put credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100, put_option_95]
        )
        
        assert position.max_profit() == 2.50
    
    def test_put_credit_spread_max_loss(self, put_option_100, put_option_95):
        """Test max loss for put credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100, put_option_95]
        )
        
        # Spread width = 5, credit = 2.50, max loss = 5 - 2.50 = 2.50
        assert position.max_loss() == 2.50
    
    def test_wider_credit_spread_worse_risk_reward(self, call_option_100):
        """Test that wider spreads have worse risk/reward ratios."""
        # 5-point spread
        call_105 = Option(
            ticker="O:SPY250101C00105000",
            symbol="SPY",
            strike=105.0,
            expiration="2025-01-01",
            option_type=OptionType.CALL,
            last_price=0.50
        )
        
        position_5pt = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_105]
        )
        
        # 10-point spread (less credit per point width)
        call_110 = Option(
            ticker="O:SPY250101C00110000",
            symbol="SPY",
            strike=110.0,
            expiration="2025-01-01",
            option_type=OptionType.CALL,
            last_price=0.20
        )
        
        position_10pt = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.80,  # Only slightly more credit
            spread_options=[call_option_100, call_110]
        )
        
        # Verify wider spread has worse risk/reward
        assert position_10pt.risk_reward_ratio() > position_5pt.risk_reward_ratio()
    
    # ==================== Long Option Tests ====================
    
    def test_long_call_max_profit(self, call_option_100):
        """Test max profit for long call (unlimited)."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,  # Premium paid
            spread_options=[call_option_100]
        )
        
        assert position.max_profit() is None  # Unlimited
    
    def test_long_call_max_loss(self, call_option_100):
        """Test max loss for long call."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.max_loss() == 2.50
    
    def test_long_call_risk_reward_none(self, call_option_100):
        """Test risk/reward ratio for long call (None due to unlimited profit)."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.risk_reward_ratio() is None
    
    def test_long_put_max_profit(self, put_option_100):
        """Test max profit for long put."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_PUT,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100]
        )
        
        # Max profit if stock goes to 0: 100 - 2.50 = 97.50
        assert position.max_profit() == 97.50
    
    def test_long_put_max_loss(self, put_option_100):
        """Test max loss for long put."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_PUT,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100]
        )
        
        assert position.max_loss() == 2.50
    
    # ==================== Short Option Tests ====================
    
    def test_short_call_max_profit(self, call_option_100):
        """Test max profit for short call."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.SHORT_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,  # Premium received
            spread_options=[call_option_100]
        )
        
        assert position.max_profit() == 2.50
    
    def test_short_call_max_loss(self, call_option_100):
        """Test max loss for short call (unlimited)."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.SHORT_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.max_loss() is None  # Unlimited
    
    def test_short_call_risk_reward_none(self, call_option_100):
        """Test risk/reward ratio for short call (None due to unlimited loss)."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.SHORT_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.risk_reward_ratio() is None
    
    def test_short_put_max_profit(self, put_option_100):
        """Test max profit for short put."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.SHORT_PUT,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100]
        )
        
        assert position.max_profit() == 2.50
    
    def test_short_put_max_loss(self, put_option_100):
        """Test max loss for short put."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.SHORT_PUT,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[put_option_100]
        )
        
        # Max loss if stock goes to 0: 100 - 2.50 = 97.50
        assert position.max_loss() == 97.50
    
    # ==================== Expected Value Tests ====================
    
    def test_expected_value_positive(self, call_option_100, call_option_105):
        """Test expected value calculation with positive EV."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        # Max profit = 2.50, max loss = 2.50, PoP = 0.70
        # EV = (2.50 × 0.70) - (2.50 × 0.30) = 1.75 - 0.75 = 1.00
        assert position.expected_value(0.70) == pytest.approx(1.00)
    
    def test_expected_value_negative(self, call_option_100, call_option_105):
        """Test expected value calculation with negative EV."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        # Max profit = 2.50, max loss = 2.50, PoP = 0.30
        # EV = (2.50 × 0.30) - (2.50 × 0.70) = 0.75 - 1.75 = -1.00
        assert position.expected_value(0.30) == -1.00
    
    def test_expected_value_invalid_probability(self, call_option_100, call_option_105):
        """Test expected value with invalid probability raises error."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        with pytest.raises(ValueError, match="probability_of_profit must be between 0 and 1"):
            position.expected_value(1.5)
    
    def test_expected_value_unlimited_profit(self, call_option_100):
        """Test expected value returns None for unlimited profit."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.expected_value(0.70) is None
    
    # ==================== Spread Width Tests ====================
    
    def test_spread_width_credit_spread(self, call_option_100, call_option_105):
        """Test spread width calculation for credit spread."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100, call_option_105]
        )
        
        assert position.spread_width() == 5.0
    
    def test_spread_width_single_leg(self, call_option_100):
        """Test spread width returns None for single leg strategies."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_CALL,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[call_option_100]
        )
        
        assert position.spread_width() is None
    
    # ==================== Error Handling Tests ====================
    
    def test_credit_spread_missing_options_raises_error(self):
        """Test that credit spread without spread_options raises error."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[]
        )
        
        with pytest.raises(ValueError, match="Credit spread requires 2 options"):
            position.max_loss()
    
    def test_long_put_missing_options_raises_error(self):
        """Test that long put without spread_options raises error."""
        position = Position(
            symbol="SPY",
            expiration_date=datetime(2025, 1, 1),
            strategy_type=StrategyType.LONG_PUT,
            strike_price=100.0,
            entry_date=datetime(2024, 12, 1),
            entry_price=2.50,
            spread_options=[]
        )
        
        with pytest.raises(ValueError, match="Long put requires option data"):
            position.max_profit()

