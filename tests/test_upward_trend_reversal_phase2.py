"""
Tests for Phase 2: Data Structures (TrendInfo and SpreadInfo)
"""

import pytest
from datetime import datetime
from dataclasses import FrozenInstanceError

from src.common.trend_detector import TrendInfo
from src.strategies.upward_trend_reversal_strategy import SpreadInfo
from src.common.models import Option, OptionType


class TestTrendInfoDataClass:
    """Test TrendInfo dataclass for Phase 2."""
    
    def test_trend_info_creation_with_all_fields(self):
        """Test creating TrendInfo with all fields."""
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03,
            reversal_drawdown=-0.01,
            reversal_date=datetime(2024, 1, 5)
        )
        
        assert trend.start_date == datetime(2024, 1, 1)
        assert trend.end_date == datetime(2024, 1, 4)
        assert trend.duration == 3
        assert trend.start_price == 100.0
        assert trend.end_price == 103.0
        assert trend.net_return == 0.03
        assert trend.reversal_drawdown == -0.01
        assert trend.reversal_date == datetime(2024, 1, 5)
    
    def test_trend_info_optional_reversal_fields(self):
        """Test TrendInfo with optional reversal fields as defaults."""
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03
        )
        
        assert trend.reversal_drawdown == 0.0
        assert trend.reversal_date is None
    
    def test_trend_info_immutability(self):
        """Test that TrendInfo is immutable."""
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03
        )
        
        with pytest.raises((FrozenInstanceError, AttributeError)):
            trend.duration = 5


class TestSpreadInfoDataClass:
    """Test SpreadInfo dataclass for Phase 2."""
    
    def setup_method(self):
        """Create sample options for testing."""
        self.atm_put = Option(
            ticker='O:SPY241231P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=5.00,
            volume=1000
        )
        
        self.otm_put = Option(
            ticker='O:SPY241231P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=1000
        )
    
    def test_spread_info_creation(self):
        """Test creating SpreadInfo with all fields."""
        spread = SpreadInfo(
            atm_put=self.atm_put,
            otm_put=self.otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        assert spread.atm_put == self.atm_put
        assert spread.otm_put == self.otm_put
        assert spread.net_debit == 2.00
        assert spread.width == 5.0
        assert spread.max_risk == 200.0
        assert spread.max_reward == 300.0
        assert spread.risk_reward_ratio == 1.5
        assert spread.dte == 7
    
    def test_spread_info_option_attributes(self):
        """Test accessing option attributes through SpreadInfo."""
        spread = SpreadInfo(
            atm_put=self.atm_put,
            otm_put=self.otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        # Verify we can access option properties
        assert spread.atm_put.strike == 580.0
        assert spread.otm_put.strike == 575.0
        assert spread.atm_put.last_price == 5.00
        assert spread.otm_put.last_price == 3.00
        assert spread.atm_put.expiration == '2024-12-31'
        assert spread.otm_put.expiration == '2024-12-31'
    
    def test_spread_info_risk_reward_calculations(self):
        """Test that SpreadInfo correctly represents risk/reward."""
        net_debit = 2.00
        width = 5.0
        max_risk = net_debit * 100
        max_reward = (width - net_debit) * 100
        risk_reward_ratio = max_reward / max_risk
        
        spread = SpreadInfo(
            atm_put=self.atm_put,
            otm_put=self.otm_put,
            net_debit=net_debit,
            width=width,
            max_risk=max_risk,
            max_reward=max_reward,
            risk_reward_ratio=risk_reward_ratio,
            dte=7
        )
        
        assert spread.max_risk == 200.0
        assert spread.max_reward == 300.0
        assert spread.risk_reward_ratio == 1.5
        
        # Verify the calculation makes sense
        assert spread.max_reward / spread.max_risk == spread.risk_reward_ratio
    
    def test_spread_info_immutability(self):
        """Test that SpreadInfo is immutable."""
        spread = SpreadInfo(
            atm_put=self.atm_put,
            otm_put=self.otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        with pytest.raises((FrozenInstanceError, AttributeError)):
            spread.net_debit = 3.00
    
    def test_spread_info_with_different_strikes(self):
        """Test SpreadInfo with various strike widths."""
        widths = [3.0, 5.0, 6.0]
        
        for width in widths:
            otm_strike = self.atm_put.strike - width
            otm_put = Option(
                ticker=f'O:SPY241231P00{int(otm_strike*1000):07d}',
                symbol='SPY',
                strike=otm_strike,
                expiration='2024-12-31',
                option_type=OptionType.PUT,
                last_price=self.atm_put.last_price - 2.00,
                volume=1000
            )
            
            spread = SpreadInfo(
                atm_put=self.atm_put,
                otm_put=otm_put,
                net_debit=2.00,
                width=width,
                max_risk=200.0,
                max_reward=(width - 2.00) * 100,
                risk_reward_ratio=((width - 2.00) * 100) / 200.0,
                dte=7
            )
            
            assert spread.width == width
            assert spread.atm_put.strike - spread.otm_put.strike == width


class TestDataStructuresIntegration:
    """Test integration between TrendInfo and SpreadInfo."""
    
    def test_combined_signal_generation(self):
        """Test that TrendInfo and SpreadInfo work together for signal generation."""
        # Create a trend that triggered a signal
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=575.0,
            end_price=580.0,
            net_return=0.0087,  # ~0.87% return
            reversal_drawdown=-0.005,  # -0.5% drawdown
            reversal_date=datetime(2024, 1, 5)
        )
        
        # Create a spread for the signal
        atm_put = Option(
            ticker='O:SPY241212P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-12',
            option_type=OptionType.PUT,
            last_price=5.00,
            volume=1000
        )
        
        otm_put = Option(
            ticker='O:SPY241212P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-12',
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=1000
        )
        
        spread = SpreadInfo(
            atm_put=atm_put,
            otm_put=otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        # Verify the signal makes sense
        assert trend.reversal_date == datetime(2024, 1, 5)
        assert trend.duration == 3
        assert spread.dte == 7
        assert spread.risk_reward_ratio > 1.0  # Favorable risk/reward
        
        # Verify the spread strike aligns with trend end price
        assert abs(spread.atm_put.strike - trend.end_price) < 1.0


class TestPhase2Completeness:
    """Verify that Phase 2 is fully implemented."""
    
    def test_trend_info_has_all_required_fields(self):
        """Verify TrendInfo has all fields specified in Phase 2."""
        required_fields = [
            'start_date', 'end_date', 'duration', 
            'start_price', 'end_price', 'net_return',
            'reversal_drawdown', 'reversal_date'
        ]
        
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03
        )
        
        for field in required_fields:
            assert hasattr(trend, field)
    
    def test_spread_info_has_all_required_fields(self):
        """Verify SpreadInfo has all fields specified in Phase 2."""
        required_fields = [
            'atm_put', 'otm_put', 'net_debit', 'width',
            'max_risk', 'max_reward', 'risk_reward_ratio', 'dte'
        ]
        
        atm_put = Option(
            ticker='O:SPY241231P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=5.00
        )
        
        otm_put = Option(
            ticker='O:SPY241231P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=3.00
        )
        
        spread = SpreadInfo(
            atm_put=atm_put,
            otm_put=otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        for field in required_fields:
            assert hasattr(spread, field)
    
    def test_both_dataclasses_are_frozen(self):
        """Verify both dataclasses are immutable (frozen)."""
        # TrendInfo should be immutable
        trend = TrendInfo(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
            duration=3,
            start_price=100.0,
            end_price=103.0,
            net_return=0.03
        )
        
        with pytest.raises((FrozenInstanceError, AttributeError)):
            trend.duration = 5
        
        # SpreadInfo should be immutable
        atm_put = Option(
            ticker='O:SPY241231P00580000',
            symbol='SPY',
            strike=580.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=5.00
        )
        
        otm_put = Option(
            ticker='O:SPY241231P00575000',
            symbol='SPY',
            strike=575.0,
            expiration='2024-12-31',
            option_type=OptionType.PUT,
            last_price=3.00
        )
        
        spread = SpreadInfo(
            atm_put=atm_put,
            otm_put=otm_put,
            net_debit=2.00,
            width=5.0,
            max_risk=200.0,
            max_reward=300.0,
            risk_reward_ratio=1.5,
            dte=7
        )
        
        with pytest.raises((FrozenInstanceError, AttributeError)):
            spread.dte = 10

